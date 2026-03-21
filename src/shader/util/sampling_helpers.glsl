#ifndef SAMPLING_HELPERS_GLSL
#define SAMPLING_HELPERS_GLSL

#include "disney.glsl"
#include "color_space.glsl"

vec3 sampleCosineHemisphere(inout uint seed, vec3 N) {
    vec3 T, B;
    Onb(N, T, B);
    vec3 localL = CosineSampleHemisphere(rand(seed), rand(seed));
    return ToWorld(T, B, N, localL);
}

// Overload for Xi (blue noise)
vec3 sampleCosineHemisphereXi(vec2 xi, vec3 N) {
    vec3 T, B;
    Onb(N, T, B);
    vec3 localL = CosineSampleHemisphere(xi.x, xi.y);
    return ToWorld(T, B, N, localL);
}

vec3 sampleGGX(inout uint seed, vec3 N, float roughness) {
    vec3 T, B;
    Onb(N, T, B);
    
    float r1 = rand(seed);
    float r2 = rand(seed);
    float a = roughness * roughness;
    float phi = 2.0 * PI * r1;
    float cosTheta2 = (1.0 - r2) / (r2 * (a * a - 1.0) + 1.0);
    float cosTheta = sqrt(clamp(cosTheta2, 0.0, 1.0));
    float sinTheta = sqrt(clamp(1.0 - cosTheta2, 0.0, 1.0));
    
    vec3 localH = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    
    vec3 H = ToWorld(T, B, N, localH);
    return H;
}

vec3 sampleGGXXi(vec2 xi, vec3 N, float roughness) {
    vec3 T, B;
    Onb(N, T, B);
    
    float r1 = xi.x;
    float r2 = xi.y;
    float a = roughness * roughness;
    float phi = 2.0 * PI * r1;
    float cosTheta2 = (1.0 - r2) / (r2 * (a * a - 1.0) + 1.0);
    float cosTheta = sqrt(clamp(cosTheta2, 0.0, 1.0));
    float sinTheta = sqrt(clamp(1.0 - cosTheta2, 0.0, 1.0));
    
    vec3 localH = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    
    vec3 H = ToWorld(T, B, N, localH);
    return H;
}

vec3 sampleVMF(inout uint seed, vec3 mu, float kappa) {
    return SampleVMF(seed, mu, kappa);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * SchlickWeight(cosTheta);
}

ivec2 clampTexelCoord(ivec2 coord, ivec2 size) {
    return clamp(coord, ivec2(0), max(size - ivec2(1), ivec2(0)));
}

vec2 clampUvToRect(vec2 uv, vec2 uvMin, vec2 uvMax, ivec2 size) {
    vec2 minUv = min(uvMin, uvMax);
    vec2 maxUv = max(uvMin, uvMax);
    vec2 halfTexel = 0.5 / vec2(size);

    minUv += halfTexel;
    maxUv -= halfTexel;
    maxUv = max(maxUv, minUv);

    return clamp(uv, minUv, maxUv);
}

vec4 sampleNearest(sampler2D tex, vec2 uv, int lod, bool isSRGB) {
    ivec2 size = textureSize(tex, lod);
    if (size.x <= 0 || size.y <= 0) { return vec4(0.0); }
    ivec2 texel = ivec2(uv * vec2(size));
    return sampleTexture(tex, clampTexelCoord(texel, size), lod, isSRGB);
}

vec4 sampleBilinear(sampler2D tex, vec2 uv, int lod, bool isSRGB) {
    ivec2 size = textureSize(tex, lod);
    if (size.x <= 0 || size.y <= 0) { return vec4(0.0); }

    vec2 pixelCoord = uv * vec2(size) - vec2(0.5);
    ivec2 p0 = ivec2(floor(pixelCoord));
    ivec2 p1 = p0 + ivec2(1);
    vec2 fracPart = fract(pixelCoord);

    ivec2 t00 = clampTexelCoord(ivec2(p0.x, p0.y), size);
    ivec2 t10 = clampTexelCoord(ivec2(p1.x, p0.y), size);
    ivec2 t01 = clampTexelCoord(ivec2(p0.x, p1.y), size);
    ivec2 t11 = clampTexelCoord(ivec2(p1.x, p1.y), size);

    vec4 c00 = sampleTexture(tex, t00, lod, isSRGB);
    vec4 c10 = sampleTexture(tex, t10, lod, isSRGB);
    vec4 c01 = sampleTexture(tex, t01, lod, isSRGB);
    vec4 c11 = sampleTexture(tex, t11, lod, isSRGB);

    vec4 c0 = mix(c00, c10, fracPart.x);
    vec4 c1 = mix(c01, c11, fracPart.x);
    return mix(c0, c1, fracPart.y);
}

vec4 samplePBRTexture(sampler2D tex,
                      vec2 uv,
                      vec2 atlasUvMin,
                      vec2 atlasUvMax,
                      float lod,
                      uint samplingMode) {
    int lodLevel = clamp(int(floor(lod)), 0, max(textureQueryLevels(tex) - 1, 0));
    ivec2 size = textureSize(tex, lodLevel);
    if (size.x <= 0 || size.y <= 0) { return vec4(0.0); }

    vec2 clampedUv = clampUvToRect(uv, atlasUvMin, atlasUvMax, size);
    if (samplingMode == 0u) {
        return sampleNearest(tex, clampedUv, lodLevel, false);
    } else {
        return sampleBilinear(tex, clampedUv, lodLevel, false);
    }
}

#endif
