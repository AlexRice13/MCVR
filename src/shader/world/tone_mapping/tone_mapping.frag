#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D HDR;

layout(set = 0, binding = 2) readonly buffer ExposureBuffer {
    float exposure;
    float avgLogLum;
    float padding0;
    float padding1;
}
expData;

layout(push_constant) uniform PushConstant {
    float log2Min;
    float log2Max;
    float epsilon;
    float lowPercent;
    float highPercent;
    float middleGrey;
    float dt;
    float speedUp;
    float speedDown;
    float minExposure;
    float maxExposure;
    float manualExposure;
    float exposureBias;
    float whitePoint;
    float saturation;
    float contrast;
    float gradingGamma;
    float lift;
    float gain;
    float temperature;
    float tint;
    int toneMappingMethod;
    int autoExposure;
    int clampOutput;
    int exposureMeteringMode;
    float centerMeteringPercent;
    int hdrActive;
    float hdrMinLuminance;
    float hdrMaxLuminance;
    float hdrRollOff;
    float padding0;
    float padding1;
}
pc;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

// https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral
vec3 pbrNeutralToneMap(vec3 color) {
    float startCompression = 0.76;
    float desaturation = 0.01;

    float x = min(color.r, min(color.g, color.b));
    float offset = (x < 0.08) ? (x - 6.25 * x * x) : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, newPeak * vec3(1.0), g);
}

// all following
// https://64.github.io/tonemapping/
vec3 reinhardToneMap(vec3 color) {
    return color / (1.0 + color);
}

vec3 reinhardWhitePointToneMap(vec3 color, float whitePoint) {
    float w2 = max(whitePoint * whitePoint, 1e-6);
    return (color * (1.0 + color / w2)) / (1.0 + color);
}

vec3 acesFittedRaw(vec3 color) {
    vec3 a = color * (2.51 * color + 0.03);
    vec3 b = color * (2.43 * color + 0.59) + 0.14;
    return a / max(b, vec3(1e-6));
}

vec3 acesFittedToneMap(vec3 color) {
    return clamp(acesFittedRaw(color), 0.0, 1.0);
}

vec3 acesFittedWhitePointToneMap(vec3 color, float whitePoint) {
    float whiteScale = 1.0 / max(acesFittedRaw(vec3(whitePoint)).r, 1e-6);
    return clamp(acesFittedRaw(color) * whiteScale, 0.0, 1.0);
}

vec3 uncharted2Partial(vec3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 uncharted2ToneMap(vec3 color, float whitePoint) {
    vec3 mapped = uncharted2Partial(color);
    float whiteScale = 1.0 / max(uncharted2Partial(vec3(whitePoint)).r, 1e-6);
    return mapped * whiteScale;
}

vec3 applyToneMapping(vec3 color) {
    switch (pc.toneMappingMethod) {
        case 1: return reinhardToneMap(color);
        case 2: return reinhardWhitePointToneMap(color, pc.whitePoint);
        case 3: return acesFittedToneMap(color);
        case 4: return acesFittedWhitePointToneMap(color, pc.whitePoint);
        case 5: return uncharted2ToneMap(color, pc.whitePoint);
        case 0:
        default: return pbrNeutralToneMap(color);
    }
}

vec3 applySaturation(vec3 color, float saturation) {
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luma), color, saturation);
}

vec3 applyContrast(vec3 color, float contrast) {
    return max(((color - 0.5) * contrast) + 0.5, vec3(0.0));
}

vec3 applyLiftGain(vec3 color, float lift, float gain) {
    return max((color + vec3(lift)) * vec3(gain), vec3(0.0));
}

vec3 applyTemperatureTint(vec3 color, float temperature, float tint) {
    vec3 temperatureScale = vec3(1.0 + temperature * 0.15, 1.0, 1.0 - temperature * 0.15);
    vec3 tintScale = vec3(1.0 + tint * 0.05, 1.0 - tint * 0.10, 1.0 + tint * 0.05);
    return max(color * temperatureScale * tintScale, vec3(0.0));
}

vec3 applyCalibrationGamma(vec3 color, float gammaValue) {
    return pow(max(color, vec3(0.0)), vec3(1.0 / max(gammaValue, 1e-6)));
}

float applyHdrReinhardRollOff(float value, float rollOff) {
    float clampedValue = clamp(value, 0.0, 1.0);
    float shoulder = max(rollOff, 1e-3);
    return clamp(clampedValue / max(clampedValue + shoulder * (1.0 - clampedValue), 1e-6), 0.0, 1.0);
}

vec3 applyHdrReinhardRollOff(vec3 value, float rollOff) {
    return vec3(
        applyHdrReinhardRollOff(value.r, rollOff),
        applyHdrReinhardRollOff(value.g, rollOff),
        applyHdrReinhardRollOff(value.b, rollOff)
    );
}

void main() {
    const float SDR_REFERENCE_WHITE_NITS = 80.0;

    vec3 hdr = texture(HDR, texCoord).rgb;

    float exposure = (pc.autoExposure != 0) ? expData.exposure : pc.manualExposure;
    if (isnan(exposure) || isinf(exposure) || exposure <= 0.0) { exposure = max(pc.manualExposure, 1e-6); }
    exposure *= exp2(pc.exposureBias);

    vec3 expColor = max(hdr * max(exposure, 0.0), vec3(0.0));
    vec3 mapped = applyToneMapping(expColor);
    mapped = max(mapped, vec3(0.0));
    mapped = applyTemperatureTint(mapped, pc.temperature, pc.tint);
    mapped = applyLiftGain(mapped, pc.lift, pc.gain);
    mapped = applyContrast(mapped, max(pc.contrast, 0.0));
    mapped = applySaturation(mapped, max(pc.saturation, 0.0));

    if (pc.hdrActive != 0) {
        float minOutput = max(pc.hdrMinLuminance, 0.0) / SDR_REFERENCE_WHITE_NITS;
        float maxOutput = max(pc.hdrMaxLuminance, pc.hdrMinLuminance + 1e-3) / SDR_REFERENCE_WHITE_NITS;

        mapped = applyHdrReinhardRollOff(mapped, pc.hdrRollOff);
        vec3 hdrOutput = mix(vec3(minOutput), vec3(maxOutput), clamp(mapped, vec3(0.0), vec3(1.0)));

        if (pc.clampOutput != 0) hdrOutput = clamp(hdrOutput, vec3(0.0), vec3(maxOutput));
        fragColor = vec4(hdrOutput, 1.0);
        return;
    }

    mapped = applyCalibrationGamma(mapped, pc.gradingGamma);
    mapped = pow(mapped, vec3(1.0 / 2.2));
    if (pc.clampOutput != 0) mapped = clamp(mapped, vec3(0.0), vec3(1.0));

    fragColor = vec4(mapped, 1.0);
}
