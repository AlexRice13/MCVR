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
#define CLOUD_MIN_HEIGHT     192.0
#define CLOUD_THICKNESS      4.0
#define CLOUD_MAX_HEIGHT     (CLOUD_MIN_HEIGHT + CLOUD_THICKNESS)

#define CLOUD_MARCH_STEPS    8
#define SUN_BRIGHTNESS       3.0

#define CLOUD_CELL_SIZE      12.0
#define CLOUD_GRID_SIZE      256       // cells per axis (must match Java)
#define CLOUD_DENSITY_SCALE  5.0       // extinction multiplier so clouds actually block light

// ─── Multi-scattering parameters ───────────────────────────────────────────────
#define MS_OCTAVES           6
#define MS_ATTENUATION       0.35
#define MS_CONTRIBUTION      0.45
#define MS_PHASE_DECAY       0.50

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

// Cloud coverage SSBO — 256×256 bytes packed as uint32
layout(std430, set = 2, binding = 3) readonly buffer CloudCoverage {
    uint cells[];
} cloudCoverage;

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

// ─── Cloud coverage sampling ───────────────────────────────────────────────────
// Read one byte from the SSBO (grid cell occupancy: 0 or 255)
float readCloudCell(int cx, int cz) {
    cx &= (CLOUD_GRID_SIZE - 1);
    cz &= (CLOUD_GRID_SIZE - 1);
    int idx = cz * CLOUD_GRID_SIZE + cx;
    uint packed = cloudCoverage.cells[idx >> 2];
    uint byteVal = (packed >> ((idx & 3) * 8)) & 0xFFu;
    return float(byteVal) / 255.0;
}

// Distance-based edge falloff: full density inside, smooth falloff toward air neighbors.
// Avoids the hollowness/cross artifacts of smoothstep-rounding + bilinear on binary data.
float sampleCloudCoverage(vec2 worldXZ, float edgeSoftness) {
    vec2 gridPos = worldXZ / CLOUD_CELL_SIZE;
    ivec2 cell = ivec2(floor(gridPos));
    vec2 localPos = fract(gridPos);  // [0,1) within cell

    // If this cell is air, no cloud
    float center = readCloudCell(cell.x, cell.y);
    if (center < 0.5) return 0.0;

    // Check 4 direct neighbors
    float nXn = readCloudCell(cell.x - 1, cell.y);
    float nXp = readCloudCell(cell.x + 1, cell.y);
    float nZn = readCloudCell(cell.x, cell.y - 1);
    float nZp = readCloudCell(cell.x, cell.y + 1);

    // Distance to each edge; apply falloff only toward air neighbors
    float edge = 1.0;
    float soft = max(edgeSoftness, 0.02);  // avoid division by zero
    if (nXn < 0.5) edge *= smoothstep(0.0, soft, localPos.x);
    if (nXp < 0.5) edge *= smoothstep(0.0, soft, 1.0 - localPos.x);
    if (nZn < 0.5) edge *= smoothstep(0.0, soft, localPos.y);
    if (nZp < 0.5) edge *= smoothstep(0.0, soft, 1.0 - localPos.y);

    return edge;
}

// ─── Henyey-Greenstein phase function ──────────────────────────────────────────
float hgPhase(float cosTheta, float g) {
    float g2 = g * g;
    return 0.25 * ((1.0 - g2) * pow(1.0 + g2 - 2.0 * g * cosTheta, -1.5));
}

float cloudPhase(float cosTheta, float anisotropy) {
    float g1 =  0.8 * anisotropy;
    float g2 = -0.5 * anisotropy;
    float lobe1 = hgPhase(cosTheta, g1);
    float lobe2 = hgPhase(cosTheta, g2);
    return mix(lobe2, lobe1, 0.6);
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

// ─── Self-shadow ───────────────────────────────────────────────────────────────
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
vec3 multiScatterStep(float od, float lightDist, float opacity,
                      float localCoverage,
                      float cosTheta, float anisotropy,
                      vec3 sunColor, vec3 skyLight) {
    vec3 totalScatter = vec3(0.0);
    float attMul = 1.0;
    float contMul = 1.0;
    float curAnisotropy = anisotropy;

    // Light path OD accounts for local coverage so dense regions properly self-shadow
    float baseLightOD = lightDist * opacity * CLOUD_DENSITY_SCALE * localCoverage;

    for (int n = 0; n < MS_OCTAVES; n++) {
        float effOD = od * max(attMul, 0.001);
        float effShadow = exp2(-baseLightOD * attMul);
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

    float tEntry = gl_HitTEXT;

    float tExit;
    if (abs(rayDir.y) < 1e-6) {
        tExit = tEntry + 20.0;
    } else {
        float tBot = (CLOUD_MIN_HEIGHT - worldOrigin.y) / rayDir.y;
        float tTop = (CLOUD_MAX_HEIGHT - worldOrigin.y) / rayDir.y;
        tExit = max(tBot, tTop);
    }

    tExit = max(tExit, tEntry + 0.01);
    float marchDist = tExit - tEntry;

    // ── Bayer dithering ──
    vec2 pixelCoord = vec2(gl_LaunchIDEXT.xy);
    float dither = bayer16(pixelCoord);

    // ── Volume ray march ──
    const int steps = CLOUD_MARCH_STEPS;
    float stepLength = marchDist / float(steps);
    vec3 increment = rayDir * stepLength;
    vec3 marchPos = worldOrigin + rayDir * (tEntry + stepLength * dither);

    vec3  scattering    = vec3(0.0);
    float transmittance = 1.0;

    // Cloud parameters from UBO
    float densityGrad = skyUBO.cloudDensityGradient;
    float opacity     = skyUBO.cloudOpacity;
    float anisotropy  = skyUBO.cloudAnisotropy;
    float edgeSoftness = skyUBO.cloudEdgeSoftness;

    // Wind offsets for grid lookup
    vec2 windOffset = vec2(skyUBO.cloudWindOffsetX, skyUBO.cloudWindOffsetZ);

    // Sun direction
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
        // Vertical density profile
        float heightFrac = clamp((marchPos.y - CLOUD_MIN_HEIGHT) / CLOUD_THICKNESS, 0.0, 1.0);
        float vertProfile = smoothstep(0.0, 0.5, heightFrac) * smoothstep(1.0, 0.5, heightFrac);

        // Sample 2D cloud coverage map using wind-shifted world position
        vec2 shiftedXZ = marchPos.xz + windOffset;
        float coverage = sampleCloudCoverage(shiftedXZ, edgeSoftness);

        // Combine: coverage × vertical profile × opacity × density scale
        float density = coverage * mix(1.0, vertProfile, densityGrad) * opacity * CLOUD_DENSITY_SCALE;

        float od = stepLength * density;
        if (od <= 0.0) continue;

        // Multi-scattering (coverage passed for light-path attenuation)
        float lightDist = lightPathDistance(marchPos, lightDir);
        vec3 stepScatter = multiScatterStep(od, lightDist, opacity,
                                            coverage,
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

    mainRay.normal = vec3(0.0, 1.0, 0.0);

    rayStoreMaterial(mainRay,
        vec4(1.0, 1.0, 1.0, cloudAlpha),
        vec3(0.04),
        1.0,
        0.0,
        0.0,
        1.5,
        0.0
    );
    mainRay.directLightRadiance = scattering;
    raySetNoisy(mainRay, true);
    raySetSkipFog(mainRay, true);
    mainRay.hasPrevScenePos = 0u;

    if (transmittance > 0.3) {
        passThrough();
        mainRay.throughput *= transmittance;
    } else {
        raySetStop(mainRay, true);
    }
}
