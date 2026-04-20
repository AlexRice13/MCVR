#ifndef VOLUMETRIC_FOG_GLSL
#define VOLUMETRIC_FOG_GLSL

float radianceHgPhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4);
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

// Stable per-pixel hash; combined with the R2 temporal offset below this gives a
// well-distributed, low-flicker jitter without requiring the blue-noise descriptor set.
float radianceFogStepJitter(ivec2 pixel, uint seed) {
    uint x = uint(pixel.x) * 1973u;
    uint y = uint(pixel.y) * 9277u;
    uint h = x ^ y ^ 0x68bc21ebu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    float spatial = float(h & 1023u) / 1024.0;

    // R2 / generalized golden-ratio low-discrepancy temporal sequence
    // (Roberts 2018). Phi_2 ≈ 1.32471795724474602596 → reciprocal 0.7548776662466927.
    float temporal = fract(float(seed & 65535u) * 0.7548776662466927);
    return fract(spatial + temporal);
}

float radianceLocalLightAttenuation(float distanceToLight, float lightRadius) {
    float normalized = clamp(1.0 - distanceToLight / max(lightRadius, 1e-3), 0.0, 1.0);
    return normalized * normalized;
}

// Per-step firefly clamp: the single-sample local-light contribution can spike when a
// shadow ray happens to hit the unoccluded side of a small emitter. Soft-clamp luminance
// to a ceiling driven by the running mean of accumulated lighting so far.
vec3 radianceFogFireflyClamp(vec3 contribution, vec3 runningMean, float maxRatio) {
    float lum = max(max(contribution.r, contribution.g), contribution.b);
    float meanLum = max(max(runningMean.r, runningMean.g), runningMean.b);
    float ceiling = max(meanLum * maxRatio, 0.5);
    if (lum <= ceiling) { return contribution; }
    return contribution * (ceiling / max(lum, 1e-4));
}

// Adaptive step count: short rays don't need 12 steps, long rays in deep fog do.
int radianceFogAdaptiveStepCount(float marchDistance, float fogRampDistance) {
    if (marchDistance <= fogRampDistance * 0.5) { return 4; }
    if (marchDistance <= fogRampDistance * 1.5) { return 8; }
    return 12;
}

#endif
