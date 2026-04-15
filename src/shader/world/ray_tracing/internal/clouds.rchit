#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "util/random.glsl"
#include "util/ray.glsl"
#include "util/util.glsl"
#include "common/shared.hpp"

// ─── Parameters ────────────────────────────────────────────────────────────────
// Vanilla cloud layer: 4 blocks tall at cloudHeight (192).
// Geometry is per-cell 12×4×12 boxes — the shader only does volumetric lighting.
#define CLOUD_MIN_HEIGHT     192.0
#define CLOUD_THICKNESS      4.0
#define CLOUD_MAX_HEIGHT     (CLOUD_MIN_HEIGHT + CLOUD_THICKNESS)

#define CLOUD_MARCH_STEPS    8
#define SUN_BRIGHTNESS       3.0

// ─── Noise FBM parameters ──────────────────────────────────────────────────────
#define NOISE_BASE_SCALE     0.03   // world→UV; features ~33 blocks wide
#define NOISE_OCTAVES        3
#define NOISE_LACUNARITY     2.3
#define NOISE_GAIN           0.45
#define NOISE_STRENGTH       0.65   // 0 = no noise, 1 = full noise modulation

// ─── Multi-scattering parameters ───────────────────────────────────────────────
#define MS_OCTAVES           6      // scattering orders to approximate
#define MS_ATTENUATION       0.35   // extinction reduction per octave
#define MS_CONTRIBUTION      0.45   // energy contribution per octave
#define MS_PHASE_DECAY       0.50   // anisotropy decay per octave

// ─── Descriptor sets ───────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 0, binding = 2) uniform samplerCube skyFull;
layout(set = 0, binding = 3) uniform sampler3D noiseTexture3D;  // kept for pipeline compatibility

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
} blasOffsets;

layout(set = 1, binding = 2) readonly buffer VertexBufferAddr {
    uint64_t addrs[];
} vertexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
} indexBufferAddrs;

layout(set = 1, binding = 4) readonly buffer LastVertexBufferAddr {
    uint64_t addrs[];
} lastVertexBufferAddrs;

layout(set = 1, binding = 5) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
} lastIndexBufferAddrs;

layout(set = 1, binding = 10) readonly buffer LastObjToWorldMat {
    mat4 mat[];
} lastObjToWorldMats;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 3, binding = 1, rgba8) uniform image2D diffuseAlbedoImage;
layout(set = 3, binding = 2, rgba8) uniform image2D specularAlbedoImage;
layout(set = 3, binding = 3, rgba16f) uniform image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg16f) uniform image2D motionVectorImage;
layout(set = 3, binding = 5, r16f) uniform image2D linearDepthImage;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    PBRVertex vertices[];
} vertexBuffer;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
} indexBuffer;

layout(location = 0) rayPayloadInEXT MainRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

// ─── Henyey-Greenstein phase function ──────────────────────────────────────────
float hgPhase(float cosTheta, float g) {
    float g2 = g * g;
    return 0.25 * ((1.0 - g2) * pow(1.0 + g2 - 2.0 * g * cosTheta, -1.5));
}

float cloudPhase(float cosTheta, float anisotropy) {
    float g1 =  0.8 * anisotropy;   // forward scattering lobe
    float g2 = -0.5 * anisotropy;   // back scattering lobe
    float lobe1 = hgPhase(cosTheta, g1);
    float lobe2 = hgPhase(cosTheta, g2);
    return mix(lobe2, lobe1, 0.6);
}

// ─── 3D noise FBM for cloud density ────────────────────────────────────────────
float cloudNoiseFBM(vec3 worldPos) {
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxAmp = 0.0;
    for (int i = 0; i < NOISE_OCTAVES; i++) {
        value += texture(noiseTexture3D, worldPos * NOISE_BASE_SCALE * frequency).r * amplitude;
        maxAmp += amplitude;
        frequency *= NOISE_LACUNARITY;
        amplitude *= NOISE_GAIN;
    }
    return value / maxAmp;
}

float noiseDensity(vec3 worldPos, float baseDensity) {
    float noise = cloudNoiseFBM(worldPos);
    float noiseMod = smoothstep(0.2, 0.7, noise);
    return baseDensity * mix(1.0, noiseMod, NOISE_STRENGTH);
}

// ─── Beer-powder ───────────────────────────────────────────────────────────────
float powder(float od) {
    return 1.0 - exp2(-od * 2.0);
}

float scatterIntegral(float od, float coeff) {
    float a = -coeff / log(2.0);
    float b = -1.0 / coeff;
    float c =  1.0 / coeff;
    return exp2(a * od) * b + c;
}

// ─── Analytical self-shadow for uniform-density box ────────────────────────────
// Returns the distance from point p to the slab exit along lightDir.
float lightPathDistance(vec3 p, vec3 lightDir) {
    float distToExit;
    if (lightDir.y > 0.001) {
        distToExit = (CLOUD_MAX_HEIGHT - p.y) / lightDir.y;
    } else if (lightDir.y < -0.001) {
        distToExit = (CLOUD_MIN_HEIGHT - p.y) / lightDir.y;
    } else {
        distToExit = 20.0;
    }
    return max(distToExit, 0.0);
}

float selfShadow(float lightDist, float opacity) {
    return exp2(-lightDist * opacity * 2.0);
}

// ─── Multi-scattering approximation (Schneider 2015 / Hillaire 2020) ──────────
// Approximates higher-order scattering by progressively reducing extinction and
// isotropizing the phase function across multiple octaves.
vec3 multiScatterStep(float od, float lightDist, float opacity,
                      float cosTheta, float anisotropy,
                      vec3 sunColor, vec3 skyLight) {
    vec3 totalScatter = vec3(0.0);
    float attMul = 1.0;
    float contMul = 1.0;
    float curAnisotropy = anisotropy;

    for (int n = 0; n < MS_OCTAVES; n++) {
        float effOD = od * max(attMul, 0.001);
        float effShadow = exp2(-lightDist * opacity * 2.0 * attMul);
        float integral = scatterIntegral(effOD, 1.11);
        float bp = powder(effOD * log(2.0));
        float curPhase = cloudPhase(cosTheta, curAnisotropy);

        vec3 sunLit = sunColor * effShadow * bp * curPhase * (PI * 0.5) * SUN_BRIGHTNESS;
        vec3 skyLit = skyLight * 0.25 * INV_PI;
        totalScatter += (sunLit + skyLit) * integral * PI * contMul;

        attMul *= MS_ATTENUATION;
        contMul *= MS_CONTRIBUTION;
        curAnisotropy *= MS_PHASE_DECAY;
    }
    return totalScatter;
}

// ─── Bayer dithering ───────────────────────────────────────────────────────────
float bayer2(vec2 a) {
    a = floor(a);
    return fract(dot(a, vec2(0.5, a.y * 0.75)));
}

#define bayer4(a)  (bayer2(0.5  * (a)) * 0.25 + bayer2(a))
#define bayer8(a)  (bayer4(0.5  * (a)) * 0.25 + bayer2(a))
#define bayer16(a) (bayer8(0.5  * (a)) * 0.25 + bayer2(a))

// ─── Pass-through ──────────────────────────────────────────────────────────────
void passThrough() {
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    mainRay.origin = worldPos + mainRay.direction * 0.001;
    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;
    raySetContinue(mainRay, true);
    raySetStop(mainRay, false);
}

// ─── Main ──────────────────────────────────────────────────────────────────────
void main() {
    vec3 rayOrigin = gl_WorldRayOriginEXT;
    vec3 rayDir    = gl_WorldRayDirectionEXT;
    vec3 camWorldPos = vec3(worldUbo.cameraPos.xyz);
    vec3 worldOrigin = camWorldPos + rayOrigin;

    // The ray hit a vanilla cloud cell box face at gl_HitTEXT.
    // Compute entry/exit through the cloud Y-slab from the hit point onward.
    float tEntry = gl_HitTEXT;

    float tExit;
    if (abs(rayDir.y) < 1e-6) {
        // Near-horizontal ray inside cloud → march up to 20 blocks
        tExit = tEntry + 20.0;
    } else {
        // Find where ray exits the Y-slab
        float tBot = (CLOUD_MIN_HEIGHT - worldOrigin.y) / rayDir.y;
        float tTop = (CLOUD_MAX_HEIGHT - worldOrigin.y) / rayDir.y;
        tExit = max(tBot, tTop);  // the farther slab boundary
    }

    tExit = max(tExit, tEntry + 0.01);
    float marchDist = tExit - tEntry;

    // ── Bayer dithering ──
    vec2 pixelCoord = vec2(gl_LaunchIDEXT.xy);
    float dither = bayer16(pixelCoord);

    // ── Volume ray march through the box ──
    const int steps = CLOUD_MARCH_STEPS;
    float stepLength = marchDist / float(steps);
    vec3 increment = rayDir * stepLength;
    vec3 marchPos = worldOrigin + rayDir * (tEntry + stepLength * dither);

    vec3  scattering    = vec3(0.0);
    float transmittance = 1.0;

    // Cloud parameters from UBO (Java sliders)
    float densityGrad = skyUBO.cloudDensityGradient;  // 0..1: vertical profile strength
    float opacity     = skyUBO.cloudOpacity;          // 0..1: base density multiplier
    float anisotropy  = skyUBO.cloudAnisotropy;       // 0..1: phase function asymmetry

    // Sun direction (ensure upward for lighting)
    vec3 sunDir = normalize(skyUBO.sunDirection);
    vec3 lightDir = sunDir;
    if (lightDir.y < 0.0) lightDir = -lightDir;

    float lDotW = dot(lightDir, rayDir);

    // Sky ambient
    vec3 skyLight = texture(skyFull, vec3(0.0, 1.0, 0.0)).rgb;

    // Sun color via shadow ray from cloud top
    vec3 cloudTopPos = worldOrigin + rayDir * tEntry;
    shadowRay.radiance = vec3(0.0);
    shadowRay.throughput = vec3(1.0);
    traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                WORLD_MASK, 0, 0, 0,
                cloudTopPos - camWorldPos, 0.001, lightDir, 10000.0, 1);
    vec3 sunColor = shadowRay.radiance;

    for (int i = 0; i < steps; i++, marchPos += increment) {
        // Vertical density profile: smoothstep fade at top/bottom of cloud slab
        float heightFrac = clamp((marchPos.y - CLOUD_MIN_HEIGHT) / CLOUD_THICKNESS, 0.0, 1.0);
        float vertProfile = smoothstep(0.0, 0.5, heightFrac) * smoothstep(1.0, 0.5, heightFrac);
        float baseDensity = mix(1.0, vertProfile, densityGrad) * opacity;

        // Modulate with 3D noise FBM for internal cloud structure
        float density = noiseDensity(marchPos, baseDensity);

        float od = stepLength * density;
        if (od <= 0.0) continue;

        // Multi-scattering: replaces single-scatter with octave-based approximation
        float lightDist = lightPathDistance(marchPos, lightDir);
        vec3 stepScatter = multiScatterStep(od, lightDist, opacity,
                                            lDotW, anisotropy, sunColor, skyLight);

        scattering += stepScatter * transmittance;
        transmittance *= exp2(-od);

        if (transmittance < 0.01) break;
    }

    float cloudAlpha = 1.0 - transmittance;

    if (cloudAlpha < 0.005) {
        passThrough();
        return;
    }

    // Rain darkening
    scattering *= mix(1.0, 0.3, skyUBO.rainGradient);

    mainRay.radiance += scattering * mainRay.throughput;
    mainRay.hitT = gl_HitTEXT;

    // G-buffer normal — cloud tops face up
    mainRay.normal = vec3(0.0, 1.0, 0.0);

    rayStoreMaterial(mainRay,
        vec4(1.0, 1.0, 1.0, cloudAlpha),
        vec3(0.04),   // f0
        1.0,          // roughness
        0.0,          // metallic
        0.0,          // transmission
        1.5,          // ior
        0.0           // emission
    );
    mainRay.directLightRadiance = scattering;
    raySetNoisy(mainRay, true);
    raySetSkipFog(mainRay, true);
    mainRay.hasPrevScenePos = 0u;

    // Transparency handling
    if (transmittance > 0.3) {
        passThrough();
        mainRay.throughput *= transmittance;
    } else {
        raySetStop(mainRay, true);
    }
}
