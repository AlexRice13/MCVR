#ifndef ADVANCE_ALEX_RAIN_GLSL
#define ADVANCE_ALEX_RAIN_GLSL

const uint ALEX_RAIN_EXPOSED_MATERIAL_BIT = 1u << 17u;
const uint ALEX_RAIN_PRECIPITATION_MATERIAL_BIT = 1u << 18u;
const uint ALEX_RAIN_SPLASH_MATERIAL_BIT = 1u << 19u;

#ifndef ADV_ALEX_RAIN_WETNESS_THRESHOLD
#    define ADV_ALEX_RAIN_WETNESS_THRESHOLD 0.15
#endif

bool alexHasRainExposedMaterial(uint packedData) {
    return (packedData & ALEX_RAIN_EXPOSED_MATERIAL_BIT) != 0u;
}

bool alexHasRainPrecipitationMaterial(uint packedData) {
    return (packedData & ALEX_RAIN_PRECIPITATION_MATERIAL_BIT) != 0u;
}

bool alexHasRainSplashMaterial(uint packedData) {
    return (packedData & ALEX_RAIN_SPLASH_MATERIAL_BIT) != 0u;
}

bool alexHasRainAnisotropicMaterial(uint packedData) {
    return alexHasRainPrecipitationMaterial(packedData) || alexHasRainSplashMaterial(packedData);
}

#ifndef ADVANCE_ALEX_RAIN_FLAGS_ONLY

float alexComputeRainWetnessFactor(float rainGradient, uint packedData) {
    if (!alexHasRainExposedMaterial(packedData)) { return 0.0; }
    float startThreshold = clamp(float(ADV_ALEX_RAIN_WETNESS_THRESHOLD), 0.0, 1.0);
    float endThreshold = min(startThreshold + 0.35, 1.0);
    if (endThreshold <= startThreshold) { endThreshold = min(startThreshold + 0.01, 1.0); }
    return smoothstep(startThreshold, endThreshold, rainGradient);
}

void alexApplyRainWetness(inout LabPBRMat mat, float wetness) {
    if (wetness <= EPS) { return; }
    float wetRoughness = clamp(mat.roughness * 0.28, 0.02, 0.35);
    mat.roughness = mix(mat.roughness, wetRoughness, wetness);
}

uint alexPrecipitationSeed(vec3 worldPos, vec2 uv, uint textureID) {
    return xxhash32(uvec3(floatBitsToUint(worldPos.x * 0.25 + uv.x * 17.0 + float(textureID) * 0.011),
                          floatBitsToUint(worldPos.y * 0.5 + uv.y * 31.0 + float(textureID) * 0.007),
                          floatBitsToUint(worldPos.z * 0.25 + uv.x * 13.0 + uv.y * 7.0 +
                                          float(textureID) * 0.003)));
}

void alexApplyRainMaterial(uint packedData,
                           float rainGradient,
                           vec3 worldPos,
                           vec2 uv,
                           uint textureID,
                           inout LabPBRMat mat,
                           inout vec4 albedoValue) {
    alexApplyRainWetness(mat, alexComputeRainWetnessFactor(rainGradient, packedData));

    if (!alexHasRainAnisotropicMaterial(packedData)) { return; }

    uint rainSeed = alexPrecipitationSeed(worldPos, uv, textureID);
    float rainIor = mix(1.16, 1.52, rand(rainSeed));
    float rainF0 = pow((rainIor - 1.0) / max(rainIor + 1.0, 1e-4), 2.0);

    albedoValue.rgb = vec3(1.0);
    albedoValue.a = min(albedoValue.a, mix(0.035, 0.085, rand(rainSeed)));
    mat.albedo = vec3(1.0);
    mat.f0 = vec3(rainF0);
    mat.roughness = mix(0.002, 0.018, rand(rainSeed));
    mat.metallic = 0.0;
    mat.transmission = 1.0;
    mat.ior = rainIor;
    mat.emission = 0.0;
}

vec3 alexBuildPrecipitationNormal(vec3 baseNormal, vec3 viewDir, vec3 worldPos, vec2 uv, uint textureID) {
    vec3 tangent, bitangent;
    Onb(baseNormal, tangent, bitangent);

    uint seed = alexPrecipitationSeed(worldPos, uv, textureID);
    vec2 centeredUv = uv * 2.0 - 1.0;
    float crossMask = pow(max(1.0 - abs(centeredUv.x), 0.0), 1.35);
    float lengthMask = pow(max(1.0 - abs(centeredUv.y), 0.0), 0.45);
    float dropletMask = max(crossMask * (0.55 + 0.45 * lengthMask), 0.2);

    float swirl = rand(seed) * 2.0 - 1.0;
    float tilt = rand(seed) * 2.0 - 1.0;
    float microX = rand(seed) * 2.0 - 1.0;
    float microY = rand(seed) * 2.0 - 1.0;

    vec2 slope = vec2(centeredUv.x * 1.4 + swirl * 0.5, centeredUv.y * 0.18 + tilt * 0.28);
    slope += vec2(microX, microY * 0.4) * mix(0.12, 0.55, dropletMask);
    slope *= mix(0.2, 0.95, dropletMask);

    vec3 perturbed =
        normalize(tangent * slope.x + bitangent * slope.y + baseNormal * max(0.35, 1.0 - dot(slope, slope) * 0.22));
    if (dot(perturbed, viewDir) < 0.05) { perturbed = normalize(mix(baseNormal, perturbed, 0.4)); }
    return dot(perturbed, viewDir) > 0.0 ? perturbed : baseNormal;
}

#endif

#endif
