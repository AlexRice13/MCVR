#ifndef ADVANCE_ALEX_BLOCKY_CLOUD_SHADOW_GLSL
#define ADVANCE_ALEX_BLOCKY_CLOUD_SHADOW_GLSL

#define ALEX_CLOUD_MIN_HEIGHT     192.0
#define ALEX_CLOUD_THICKNESS      4.0
#define ALEX_CLOUD_MAX_HEIGHT     (ALEX_CLOUD_MIN_HEIGHT + ALEX_CLOUD_THICKNESS)
#define ALEX_CLOUD_CELL_SIZE      12.0
#define ALEX_CLOUD_GRID_SIZE      256
#define ALEX_CLOUD_DENSITY_SCALE  5.0

layout(std430, set = 2, binding = 3) readonly buffer AlexCloudCoverage {
    uint cells[];
} alexCloudCoverage;

float alexReadCloudCell(int cx, int cz) {
    cx &= (ALEX_CLOUD_GRID_SIZE - 1);
    cz &= (ALEX_CLOUD_GRID_SIZE - 1);
    int idx = cz * ALEX_CLOUD_GRID_SIZE + cx;
    uint packed = alexCloudCoverage.cells[idx >> 2];
    uint byteVal = (packed >> ((idx & 3) * 8)) & 0xFFu;
    return float(byteVal) / 255.0;
}

float alexSampleCloudCoverage(vec2 worldXZ, float edgeSoftness) {
    vec2 gridPos = worldXZ / ALEX_CLOUD_CELL_SIZE;
    vec2 samplePos = gridPos - 0.5;
    ivec2 base = ivec2(floor(samplePos));
    vec2 f = fract(samplePos);

    float v00 = alexReadCloudCell(base.x, base.y);
    float v10 = alexReadCloudCell(base.x + 1, base.y);
    float v01 = alexReadCloudCell(base.x, base.y + 1);
    float v11 = alexReadCloudCell(base.x + 1, base.y + 1);
    float raw = mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
    return smoothstep(0.0, max(edgeSoftness, 1e-3), raw);
}

float alexBlockyCloudLightVisibility(vec3 worldPos, vec3 lightDir) {
    if (ADV_CLOUD_MODE != 2u) { return 1.0; }

    float t0;
    float t1;
    if (abs(lightDir.y) < 1e-5) {
        if (worldPos.y < ALEX_CLOUD_MIN_HEIGHT || worldPos.y > ALEX_CLOUD_MAX_HEIGHT) { return 1.0; }
        t0 = 0.0;
        t1 = ALEX_CLOUD_CELL_SIZE * 3.0;
    } else {
        float tMin = (ALEX_CLOUD_MIN_HEIGHT - worldPos.y) / lightDir.y;
        float tMax = (ALEX_CLOUD_MAX_HEIGHT - worldPos.y) / lightDir.y;
        t0 = max(min(tMin, tMax), 0.0);
        t1 = max(tMin, tMax);
        if (t1 <= t0) { return 1.0; }
    }

    float marchDistance = min(t1 - t0, ALEX_CLOUD_CELL_SIZE * 3.0);
    const int LIGHT_STEPS = 5;
    float stepLen = marchDistance / float(LIGHT_STEPS);
    vec3 pos = worldPos + lightDir * (t0 + stepLen * 0.5);
    vec2 windOffset = vec2(skyUBO.cloudWindOffsetX, skyUBO.cloudWindOffsetZ);

    float totalOD = 0.0;
    for (int i = 0; i < LIGHT_STEPS; ++i) {
        float hFrac = clamp((pos.y - ALEX_CLOUD_MIN_HEIGHT) / ALEX_CLOUD_THICKNESS, 0.0, 1.0);
        float vertProfile = smoothstep(0.0, 0.5, hFrac) * smoothstep(1.0, 0.5, hFrac);
        float coverage = alexSampleCloudCoverage(pos.xz + windOffset, skyUBO.cloudEdgeSoftness);
        totalOD += coverage * mix(1.0, vertProfile, skyUBO.cloudDensityGradient) * skyUBO.cloudOpacity *
                   ALEX_CLOUD_DENSITY_SCALE * stepLen;
        pos += lightDir * stepLen;
    }

    return clamp(exp(-totalOD), 0.0, 1.0);
}

#endif
