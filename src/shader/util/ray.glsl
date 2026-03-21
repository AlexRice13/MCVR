#ifndef RAY_GLSL
#define RAY_GLSL

#include "util/ray_payloads.glsl"

const uint rayBounceMask = 0xFFu;
const uint rayInsideBoatBit = 1u << 8u;
const uint rayStopBit = 1u << 9u;
const uint rayContinueBit = 1u << 10u;
const uint rayNoisyBit = 1u << 11u;
const uint rayLobeShift = 12u;
const uint rayLobeMask = 0x3u << rayLobeShift;
const uint raySkipFogBit = 1u << 14u;

void raySetBounce(inout MainRay ray, uint bounce) {
    ray.stateBits = (ray.stateBits & ~rayBounceMask) | (bounce & rayBounceMask);
}

uint rayBounce(MainRay ray) {
    return ray.stateBits & rayBounceMask;
}

void raySetInsideBoat(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayInsideBoatBit) : (ray.stateBits & ~rayInsideBoatBit);
}

bool rayInsideBoat(MainRay ray) {
    return (ray.stateBits & rayInsideBoatBit) != 0u;
}

void raySetStop(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayStopBit) : (ray.stateBits & ~rayStopBit);
}

bool rayShouldStop(MainRay ray) {
    return (ray.stateBits & rayStopBit) != 0u;
}

void raySetContinue(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayContinueBit) : (ray.stateBits & ~rayContinueBit);
}

bool rayShouldContinue(MainRay ray) {
    return (ray.stateBits & rayContinueBit) != 0u;
}

void raySetNoisy(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayNoisyBit) : (ray.stateBits & ~rayNoisyBit);
}

bool rayIsNoisy(MainRay ray) {
    return (ray.stateBits & rayNoisyBit) != 0u;
}

void raySetSkipFog(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | raySkipFogBit) : (ray.stateBits & ~raySkipFogBit);
}

bool raySkipFog(MainRay ray) {
    return (ray.stateBits & raySkipFogBit) != 0u;
}

void raySetLobeType(inout MainRay ray, uint lobeType) {
    ray.stateBits = (ray.stateBits & ~rayLobeMask) | ((lobeType & 0x3u) << rayLobeShift);
}

uint rayLobeType(MainRay ray) {
    return (ray.stateBits & rayLobeMask) >> rayLobeShift;
}

void rayClearMaterial(inout MainRay ray) {
    ray.materialPacked0 = 0u;
    ray.materialPacked1 = 0u;
    ray.materialPacked2 = 0u;
    ray.materialPacked3 = 0u;
    ray.materialPacked4 = 0u;
    ray.materialPacked5 = 0u;
}

void rayStoreMaterial(inout MainRay ray,
                       vec4 albedoValue,
                       vec3 f0,
                       float roughness,
                       float metallic,
                       float transmission,
                       float ior,
                       float emission) {
    ray.materialPacked0 = packHalf2x16(albedoValue.rg);
    ray.materialPacked1 = packHalf2x16(vec2(albedoValue.b, albedoValue.a));
    ray.materialPacked2 = packHalf2x16(f0.rg);
    ray.materialPacked3 = packHalf2x16(vec2(f0.b, roughness));
    ray.materialPacked4 = packHalf2x16(vec2(metallic, transmission));
    ray.materialPacked5 = packHalf2x16(vec2(ior, emission));
}

MaterialInfo rayLoadMaterial(MainRay ray) {
    MaterialInfo mat;
    vec2 albedoRG = unpackHalf2x16(ray.materialPacked0);
    vec2 albedoBA = unpackHalf2x16(ray.materialPacked1);
    vec2 f0RG = unpackHalf2x16(ray.materialPacked2);
    vec2 f0BRoughness = unpackHalf2x16(ray.materialPacked3);
    vec2 material01 = unpackHalf2x16(ray.materialPacked4);
    vec2 material23 = unpackHalf2x16(ray.materialPacked5);

    mat.albedoValue = vec4(albedoRG.x, albedoRG.y, albedoBA.x, albedoBA.y);
    mat.f0 = vec3(f0RG.x, f0RG.y, f0BRoughness.x);
    mat.roughness = f0BRoughness.y;
    mat.metallic = material01.x;
    mat.transmission = material01.y;
    mat.ior = material23.x;
    mat.emission = material23.y;
    return mat;
}

#endif
