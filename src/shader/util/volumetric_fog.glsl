#ifndef VOLUMETRIC_FOG_GLSL
#define VOLUMETRIC_FOG_GLSL

float radianceHgPhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4);
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

float radianceFogStepJitter(ivec2 pixel, uint seed) {
    uint x = uint(pixel.x) * 1973u;
    uint y = uint(pixel.y) * 9277u;
    uint h = x ^ y ^ seed ^ 0x68bc21ebu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return float(h & 1023u) / 1024.0;
}

float radianceLocalLightAttenuation(float distanceToLight, float lightRadius) {
    float normalized = clamp(1.0 - distanceToLight / max(lightRadius, 1e-3), 0.0, 1.0);
    return normalized * normalized;
}

#endif
