#ifndef ALPHA_MODE_GLSL
#define ALPHA_MODE_GLSL

const uint ALPHA_MODE_OPAQUE = 0u;
const uint ALPHA_MODE_CUTOUT = 1u;
const uint ALPHA_MODE_TRANSPARENT = 2u;
const uint MATERIAL_FLAG_RAIN_EXPOSED = 1u << 4u;
const uint MATERIAL_FLAG_RAIN_PRECIPITATION = 1u << 5u;
const uint MATERIAL_FLAG_RAIN_SPLASH = 1u << 6u;

const float CUTOUT_ALPHA_THRESHOLD = 0.5;

uint decodeAlphaMode(uint encodedMaterialData) {
    return encodedMaterialData & 0xFu;
}

bool hasRainExposedMaterial(uint encodedMaterialData) {
    return (encodedMaterialData & MATERIAL_FLAG_RAIN_EXPOSED) != 0u;
}

bool hasPrecipitationMaterial(uint encodedMaterialData) {
    return (encodedMaterialData & MATERIAL_FLAG_RAIN_PRECIPITATION) != 0u;
}

bool hasRainSplashMaterial(uint encodedMaterialData) {
    return (encodedMaterialData & MATERIAL_FLAG_RAIN_SPLASH) != 0u;
}

bool hasRainAnisotropicMaterial(uint encodedMaterialData) {
    return hasPrecipitationMaterial(encodedMaterialData)
        || hasRainSplashMaterial(encodedMaterialData);
}

float resolveSurfaceAlpha(float alpha, uint alphaMode) {
    alpha = clamp(alpha, 0.0, 1.0);

    if (alphaMode == ALPHA_MODE_TRANSPARENT) {
        return alpha;
    }

    if (alphaMode == ALPHA_MODE_CUTOUT) {
        return alpha >= CUTOUT_ALPHA_THRESHOLD ? 1.0 : 0.0;
    }

    return 1.0;
}

#endif
