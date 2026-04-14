#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common/shared.hpp"
#include "util/ray.glsl"
#include "util/util.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 0, binding = 1) uniform sampler2D transLUT;
layout(set = 0, binding = 2) uniform samplerCube skyFull;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 0) rayPayloadInEXT MainRay mainRay;

vec2 transmittanceUv(float r, float mu, SkyUBO ubo) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - ubo.Rg) / (ubo.Rt - ubo.Rg), 0.0, 1.0);
    return vec2(u, v);
}

vec3 sampleTransmittance(float r, float mu) {
    vec2 uv = transmittanceUv(r, mu, skyUBO);
    return sampleTexture(transLUT, uv, false).rgb;
}

bool intersectSphere(vec3 ro, vec3 rd, float R, out float tNear, out float tFar) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - R * R;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    tNear = -b - h;
    tFar = -b + h;
    return true;
}

void makeBasis(in vec3 n, out vec3 t, out vec3 b) {
    // Frisvad 2012, Building an Orthonormal Basis, Revisited
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float k = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * k, -s * n.x);
    b = vec3(k, s + n.y * n.y * a, -n.y);
    t = normalize(t);
    b = normalize(b);
}

vec4 sampleTextureLod0(sampler2D tex, vec2 uv) {
    ivec2 texSize = textureSize(tex, 0);
    if (texSize.x <= 0 || texSize.y <= 0) return vec4(0.0);

    vec2 halfTexel = 0.5 / vec2(texSize);
    vec2 clampedUv = clamp(uv, halfTexel, vec2(1.0) - halfTexel);
    return sampleTexture(tex, clampedUv, 0.0, false);
}

vec4 sampleAtlasLod0(sampler2D tex, vec2 uv01, uvec2 tileCount, uvec2 tile) {
    ivec2 texSize = textureSize(tex, 0);
    if (texSize.x <= 0 || texSize.y <= 0) return vec4(0.0);

    vec2 invTileCount = 1.0 / vec2(tileCount);
    vec2 tileMin = vec2(tile) * invTileCount;
    vec2 tileMax = tileMin + invTileCount;

    vec2 halfTexel = 0.5 / vec2(texSize);
    vec2 minUv = tileMin + halfTexel;
    vec2 maxUv = tileMax - halfTexel;
    vec2 atlasUv = mix(minUv, maxUv, clamp(uv01, 0.0, 1.0));
    return sampleTexture(tex, atlasUv, 0.0, false);
}

vec3 observerPlanetPos() {
    float cameraHeight = worldUBO.cameraViewMatInv[3].y;
    return vec3(0.0, skyUBO.Rg + cameraHeight + 70.0, 0.0);
}

vec4 evalSunBillboard(vec3 rd) {
    vec3 sunDir = normalize(skyUBO.sunDirection);
    rd = normalize(rd);

    float z = dot(rd, sunDir);
    if (z <= 0.0) return vec4(0.0);

    vec3 right, up;
    makeBasis(sunDir, right, up);

    vec2 p = vec2(dot(rd, right), dot(rd, up));
    vec2 q = p / max(z, 1e-4);

    float tanHalf = tan(skyUBO.sunAngularRadius); // sun half-angle from UBO

    vec2 a = abs(q);
    if (a.x > tanHalf || a.y > tanHalf) return vec4(0.0);

    vec2 uv = q / tanHalf * 0.5 + 0.5;
    return sampleTextureLod0(textures[nonuniformEXT(skyUBO.sunTextureID)], uv);
}

vec4 evalMoonBillboard(vec3 rd) {
    vec3 moonDir = normalize(-skyUBO.sunDirection);
    rd = normalize(rd);

    float z = dot(rd, moonDir);
    if (z <= 0.0) return vec4(0.0);

    vec3 right, up;
    makeBasis(moonDir, right, up);

    vec2 p = vec2(dot(rd, right), dot(rd, up));
    vec2 q = p / max(z, 1e-4);

    float tanHalf = tan(skyUBO.sunAngularRadius * 1.667); // moon proportionally larger

    vec2 a = abs(q);
    if (a.x > tanHalf || a.y > tanHalf) return vec4(0.0);

    vec2 uv = q / tanHalf * 0.5 + 0.5;
    uvec2 tileCount = uvec2(4u, 2u);
    uvec2 tile = uvec2(skyUBO.moonPhase % tileCount.x, (skyUBO.moonPhase / tileCount.x) % tileCount.y);
    return sampleAtlasLod0(textures[nonuniformEXT(skyUBO.moonTextureID)], uv, tileCount, tile);
}

void main() {
    vec3 rayDir = normalize(mainRay.direction);

    if (skyUBO.cameraSubmersionType == 0 /*LAVA*/ || skyUBO.cameraSubmersionType == 2 /*POWDER_SNOW*/ ||
        skyUBO.hasBlindnessOrDarkness > 0) {
        raySetStop(mainRay, true);
        mainRay.hitT = INF_DISTANCE;
    } else {
        switch (worldUBO.skyType) {
            case 0: // NONE
                raySetStop(mainRay, true);
                mainRay.hitT = INF_DISTANCE;
                return;
            case 2: // END
                raySetStop(mainRay, true);
                mainRay.hitT = INF_DISTANCE;
                return;
            case 1: // NORMAL
            default: break;
        }

        vec3 rd = normalize(gl_WorldRayDirectionEXT);
        vec3 sunDir = normalize(skyUBO.sunDirection);

        float progress = skyUBO.rainGradient;
        vec3 rainyRadiance = mix(vec3(0.0), vec3(0.1), smoothstep(-0.3, 0.3, sunDir.y));
        vec3 sunnyRadiance = texture(skyFull, rayDir).rgb;
        mainRay.radiance += mix(sunnyRadiance, rainyRadiance, progress) * mainRay.throughput;

        if (worldUBO.skyType == 1) {
            vec3 pPlanet = observerPlanetPos();
            float r = clamp(length(pPlanet), skyUBO.Rg, skyUBO.Rt);
            vec3 up = pPlanet / max(r, 1e-6);

            {
                vec4 sunSample = evalSunBillboard(rd);
                if (sunSample.a > 1e-4) {
                    float tG0, tG1;
                    bool hitGround = intersectSphere(pPlanet, rd, skyUBO.Rg, tG0, tG1);
                    bool blocked = hitGround && (tG1 > 1e-3);
                    if (!blocked) {
                        float mu = clamp(dot(up, rd), -1.0, 1.0);
                        vec3 T = sampleTransmittance(r, mu);
                        vec3 sunRadiance = (sunSample.rgb * skyUBO.sunRadiance * T * sunSample.a);
                        mainRay.radiance += mix(sunRadiance, vec3(0.0), progress) * mainRay.throughput;
                    }
                }
            }

            {
                vec4 moonSample = evalMoonBillboard(rd);

                if (moonSample.a > 1e-4) {
                    float tG0, tG1;
                    bool hitGround = intersectSphere(pPlanet, rd, skyUBO.Rg, tG0, tG1);
                    bool blocked = hitGround && (tG1 > 1e-3);
                    if (!blocked) {
                        float mu = clamp(dot(up, rd), -1.0, 1.0);
                        vec3 T = sampleTransmittance(r, mu);
                        T = max(T, vec3(0.03));
                        vec3 moonRadiance = (moonSample.rgb * skyUBO.moonRadiance);
                        moonRadiance *= T;
                        mainRay.radiance += mix(moonRadiance, vec3(0.0), progress) * mainRay.throughput;
                    }
                }
            }
        }

        raySetStop(mainRay, true);
        mainRay.hitT = INF_DISTANCE;
    }
}
