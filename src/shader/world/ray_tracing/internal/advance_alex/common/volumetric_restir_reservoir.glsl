#ifndef ADV_VOLUMETRIC_RESTIR_RESERVOIR_GLSL
#define ADV_VOLUMETRIC_RESTIR_RESERVOIR_GLSL

#ifndef ADV_VOLUMETRIC_RESTIR_GRID_DEPTH
#    define ADV_VOLUMETRIC_RESTIR_GRID_DEPTH 12
#endif
#ifndef ADV_VOLUMETRIC_RESTIR_DOWNSAMPLE_FACTOR
#    define ADV_VOLUMETRIC_RESTIR_DOWNSAMPLE_FACTOR 4
#endif
#ifndef ADV_VOLUMETRIC_RESTIR_INITIAL_SAMPLES
#    define ADV_VOLUMETRIC_RESTIR_INITIAL_SAMPLES 4
#endif
#ifndef ADV_VOLUMETRIC_RESTIR_SPATIAL_SAMPLES
#    define ADV_VOLUMETRIC_RESTIR_SPATIAL_SAMPLES 8
#endif
#ifndef ADV_VOLUMETRIC_RESTIR_SPATIAL_RADIUS
#    define ADV_VOLUMETRIC_RESTIR_SPATIAL_RADIUS 2
#endif
#ifndef ADV_VOLUMETRIC_RESTIR_M_CAP
#    define ADV_VOLUMETRIC_RESTIR_M_CAP 32.0
#endif
#ifndef ADV_VOLUMETRIC_RESTIR_TEMPORAL_REUSE
#    define ADV_VOLUMETRIC_RESTIR_TEMPORAL_REUSE 1
#endif
#ifndef ADV_EMISSIVE_VOLUME_LIGHT_FADE_DISTANCE
#    define ADV_EMISSIVE_VOLUME_LIGHT_FADE_DISTANCE 32.0
#endif
#ifndef ADV_EMISSIVE_VOLUME_LIGHT_STRENGTH
#    define ADV_EMISSIVE_VOLUME_LIGHT_STRENGTH 0.08
#endif
#ifndef ADV_DIRECT_LIGHT_STRENGTH
#    define ADV_DIRECT_LIGHT_STRENGTH 1.0
#endif
#ifndef ADV_VOLUMETRIC_LIGHT_DOWNSAMPLE_FACTOR
#    define ADV_VOLUMETRIC_LIGHT_DOWNSAMPLE_FACTOR 2
#endif
#ifndef ADV_VOLUMETRIC_LIGHT_MAX_DISTANCE
#    define ADV_VOLUMETRIC_LIGHT_MAX_DISTANCE 128.0
#endif
#ifndef ADV_ATMOSPHERE_MIE_G
#    define ADV_ATMOSPHERE_MIE_G 0.8
#endif

struct VolumetricRestirReservoir {
    uvec4 source;        // xy: source ID, z: source light index, w: source chunk index
    vec4 sampleParamsM;  // xy: area sample xi, z: selected triangle flag, w: effective sample count M
    vec4 weightTarget;   // x: unbiased reservoir weight W, y: encoded target p_hat, zw: reserved
};

struct VolumetricRestirSample {
    vec3 point;
    vec3 normal;
    vec3 emission;
    uvec2 sourceID;
    uint sourceLightIndex;
    uint sourceChunkIndex;
    vec2 sampleXi;
    float triangleSelector;
    float targetFunction;
    float contributionWeight;
    float M;
    bool valid;
};

struct VolumetricRestirRisState {
    VolumetricRestirSample sampleOut;
    float weightSum;
    bool hasSampleOut;
};

layout(std430, set = 5, binding = 39) buffer VolumetricRestirCandidateBuffer {
    VolumetricRestirReservoir volumetricRestirCandidates[];
};

layout(std430, set = 5, binding = 40) buffer VolumetricRestirCurrentBuffer {
    VolumetricRestirReservoir volumetricRestirCurrent[];
};

layout(std430, set = 5, binding = 41) buffer VolumetricRestirHistoryBuffer {
    VolumetricRestirReservoir volumetricRestirHistory[];
};

float volumetricRestirMax3(vec3 value) {
    return max(value.r, max(value.g, value.b));
}

float volumetricRestirFinitePositive(float value) {
    return (!isnan(value) && !isinf(value) && value > 0.0) ? value : 0.0;
}

vec3 volumetricRestirSafeNormalize(vec3 value, vec3 fallback) {
    float len2 = dot(value, value);
    if (isnan(len2) || isinf(len2) || len2 <= 1e-12) { return fallback; }
    return value * inversesqrt(len2);
}

float volumetricRestirHenyeyGreenstein(float cosTheta, float g) {
    float gg = g * g;
    float denom = max(1.0 + gg - 2.0 * g * cosTheta, 1e-4);
    return (1.0 - gg) * INV_4_PI / (denom * sqrt(denom));
}

VolumetricRestirSample makeEmptyVolumetricRestirSample() {
    VolumetricRestirSample lightSample;
    lightSample.point = vec3(0.0);
    lightSample.normal = vec3(0.0, 1.0, 0.0);
    lightSample.emission = vec3(0.0);
    lightSample.sourceID = uvec2(0u);
    lightSample.sourceLightIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    lightSample.sourceChunkIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    lightSample.sampleXi = vec2(0.0);
    lightSample.triangleSelector = 0.0;
    lightSample.targetFunction = 0.0;
    lightSample.contributionWeight = 0.0;
    lightSample.M = 0.0;
    lightSample.valid = false;
    return lightSample;
}

VolumetricRestirReservoir makeEmptyVolumetricRestirReservoir() {
    VolumetricRestirReservoir reservoir;
    reservoir.source = uvec4(0u, 0u, ADV_INVALID_AREA_LIGHT_SOURCE_INDEX, ADV_INVALID_AREA_LIGHT_SOURCE_INDEX);
    reservoir.sampleParamsM = vec4(0.0);
    reservoir.weightTarget = vec4(0.0);
    return reservoir;
}

VolumetricRestirRisState makeVolumetricRestirRisState() {
    VolumetricRestirRisState ris;
    ris.sampleOut = makeEmptyVolumetricRestirSample();
    ris.weightSum = 0.0;
    ris.hasSampleOut = false;
    return ris;
}

bool volumetricRestirSampleValid(VolumetricRestirSample lightSample) {
    return lightSample.valid && lightSample.M > 1e-8 && lightSample.targetFunction > 1e-8 &&
           lightSample.contributionWeight > 1e-8 && volumetricRestirMax3(lightSample.emission) > 1e-8 &&
           lightSample.sourceLightIndex != ADV_INVALID_AREA_LIGHT_SOURCE_INDEX &&
           lightSample.sourceChunkIndex != ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
}

float volumetricRestirTriangleArea(vec3 a, vec3 b, vec3 c) {
    return 0.5 * length(cross(b - a, c - a));
}

vec3 volumetricRestirSampleTriangle(vec3 a, vec3 b, vec3 c, vec2 xi) {
    float sqrtXi = sqrt(clamp(xi.x, 0.0, 1.0));
    float bary0 = 1.0 - sqrtXi;
    float bary1 = xi.y * sqrtXi;
    float bary2 = 1.0 - bary0 - bary1;
    return a * bary0 + b * bary1 + c * bary2;
}

bool volumetricRestirLoadAddressableChunkLight(uint sourceChunkIndex,
                                               uint sourceLightIndex,
                                               out ChunkPackedLight chunkLight) {
    chunkLight =
        ChunkPackedLight(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));

    ivec3 sectionCoord = ivec3(0);
    ivec3 chunkOrigin = ivec3(0);
    if (!tryGetSectionCoordinateFromChunkIndex(sourceChunkIndex, worldUBO, sectionCoord, chunkOrigin)) {
        return false;
    }

    ChunkLightData sourceChunk = chunkPackedData[sourceChunkIndex];
    if (!chunkPackedDataMatchesOrigin(sourceChunk, chunkOrigin) || !chunkPackedDataHasLights(sourceChunk) ||
        sourceLightIndex >= sourceChunk.lightCount) {
        return false;
    }

    ChunkPackedLightBuffer lightBuffer = ChunkPackedLightBuffer(sourceChunk.lightBufferAddress);
    chunkLight = lightBuffer.lights[sourceLightIndex];
    return true;
}

bool volumetricRestirReconstructSampleGeometry(inout VolumetricRestirSample lightSample) {
    ChunkPackedLight chunkLight;
    if (!volumetricRestirLoadAddressableChunkLight(lightSample.sourceChunkIndex, lightSample.sourceLightIndex, chunkLight)) {
        return false;
    }

    vec3 p0 = chunkLight.p0Area.xyz;
    vec3 p1 = chunkLight.p1.xyz;
    vec3 p2 = chunkLight.p2.xyz;
    vec3 p3 = chunkLight.p3.xyz;
    bool firstTriangle = lightSample.triangleSelector < 0.5;
    vec2 xi = clamp(lightSample.sampleXi, vec2(0.0), vec2(1.0));
    if (firstTriangle) {
        lightSample.point = volumetricRestirSampleTriangle(p0, p1, p2, xi);
        lightSample.normal = volumetricRestirSafeNormalize(cross(p1 - p0, p2 - p0), chunkLight.normal.xyz);
    } else {
        lightSample.point = volumetricRestirSampleTriangle(p0, p2, p3, xi);
        lightSample.normal = volumetricRestirSafeNormalize(cross(p2 - p0, p3 - p0), chunkLight.normal.xyz);
    }

    lightSample.emission = max(chunkLight.radiance.rgb, vec3(0.0));
    return volumetricRestirMax3(lightSample.emission) > 1e-8;
}

VolumetricRestirReservoir encodeVolumetricRestirReservoir(VolumetricRestirSample lightSample) {
    if (!volumetricRestirSampleValid(lightSample)) { return makeEmptyVolumetricRestirReservoir(); }

    VolumetricRestirReservoir reservoir;
    reservoir.source = uvec4(lightSample.sourceID, lightSample.sourceLightIndex, lightSample.sourceChunkIndex);
    reservoir.sampleParamsM = vec4(clamp(lightSample.sampleXi, vec2(0.0), vec2(1.0)),
                                   lightSample.triangleSelector < 0.5 ? 0.0 : 1.0, lightSample.M);
    reservoir.weightTarget = vec4(lightSample.contributionWeight, lightSample.targetFunction, 0.0, 0.0);
    return reservoir;
}

VolumetricRestirSample decodeVolumetricRestirReservoir(VolumetricRestirReservoir reservoir) {
    VolumetricRestirSample lightSample = makeEmptyVolumetricRestirSample();
    lightSample.M = reservoir.sampleParamsM.w;
    if (!(lightSample.M > 1e-8)) { return lightSample; }

    lightSample.sourceID = reservoir.source.xy;
    lightSample.sourceLightIndex = reservoir.source.z;
    lightSample.sourceChunkIndex = reservoir.source.w;
    lightSample.sampleXi = reservoir.sampleParamsM.xy;
    lightSample.triangleSelector = reservoir.sampleParamsM.z;
    lightSample.targetFunction = max(reservoir.weightTarget.y, 1e-8);
    lightSample.contributionWeight = reservoir.weightTarget.x;
    if (!volumetricRestirReconstructSampleGeometry(lightSample)) { return makeEmptyVolumetricRestirSample(); }
    lightSample.valid = true;
    lightSample.valid = volumetricRestirSampleValid(lightSample);
    return lightSample;
}

uint volumetricRestirLinearIndex(uvec3 coord, uvec3 gridSize) {
    return (coord.z * gridSize.y + coord.y) * gridSize.x + coord.x;
}

uint volumetricRestirCellCount(uvec3 gridSize) {
    return gridSize.x * gridSize.y * gridSize.z;
}

VolumetricRestirReservoir loadVolumetricRestirCandidate(uvec3 coord, uvec3 gridSize) {
    return volumetricRestirCandidates[volumetricRestirLinearIndex(coord, gridSize)];
}

void storeVolumetricRestirCandidate(uvec3 coord, uvec3 gridSize, VolumetricRestirReservoir reservoir) {
    volumetricRestirCandidates[volumetricRestirLinearIndex(coord, gridSize)] = reservoir;
}

VolumetricRestirReservoir loadVolumetricRestirCurrent(uvec3 coord, uvec3 gridSize) {
    return volumetricRestirCurrent[volumetricRestirLinearIndex(coord, gridSize)];
}

void storeVolumetricRestirCurrent(uvec3 coord, uvec3 gridSize, VolumetricRestirReservoir reservoir) {
    volumetricRestirCurrent[volumetricRestirLinearIndex(coord, gridSize)] = reservoir;
}

VolumetricRestirReservoir loadVolumetricRestirHistory(uvec3 coord, uvec3 gridSize, uint layer) {
    return volumetricRestirHistory[volumetricRestirCellCount(gridSize) * layer + volumetricRestirLinearIndex(coord, gridSize)];
}

void storeVolumetricRestirHistory(uvec3 coord, uvec3 gridSize, uint layer, VolumetricRestirReservoir reservoir) {
    volumetricRestirHistory[volumetricRestirCellCount(gridSize) * layer + volumetricRestirLinearIndex(coord, gridSize)] =
        reservoir;
}

uvec3 volumetricRestirGridSizeFromFullResolution(ivec2 fullResolution) {
    uint downsample = max(uint(ADV_VOLUMETRIC_RESTIR_DOWNSAMPLE_FACTOR), 1u);
    return uvec3((uint(fullResolution.x) + downsample - 1u) / downsample,
                 (uint(fullResolution.y) + downsample - 1u) / downsample,
                 uint(max(ADV_VOLUMETRIC_RESTIR_GRID_DEPTH, 1)));
}

float volumetricRestirFroxelDistance(uint z, uint depth) {
    float slice = (float(z) + 0.5) / max(float(depth), 1.0);
    float nearBiasedSlice = pow(clamp(slice, 0.0, 1.0), 1.65);
    return max(0.05, clamp(float(ADV_VOLUMETRIC_LIGHT_MAX_DISTANCE), 8.0, 320.0) * nearBiasedSlice);
}

uint volumetricRestirFroxelZFromDistance(float distanceValue, uint depth) {
    float maxDistance = clamp(float(ADV_VOLUMETRIC_LIGHT_MAX_DISTANCE), 8.0, 320.0);
    float normalized = pow(clamp(distanceValue / maxDistance, 0.0, 0.999999), 1.0 / 1.65);
    return min(uint(floor(normalized * float(depth))), depth - 1u);
}

void volumetricRestirBuildCameraRay(ivec2 lowPixel,
                                    ivec2 fullResolution,
                                    out vec3 origin,
                                    out vec3 direction) {
    int downsample = max(int(ADV_VOLUMETRIC_RESTIR_DOWNSAMPLE_FACTOR), 1);
    ivec2 fullPixel = min(lowPixel * downsample + ivec2(downsample / 2),
                          fullResolution - ivec2(1));
    vec2 pixelCenter = vec2(fullPixel) + vec2(0.5) + worldUBO.cameraJitter;
    vec2 ndc = pixelCenter / vec2(fullResolution) * 2.0 - 1.0;
    vec4 nearPoint = vec4(ndc, 0.0, 1.0);
    vec4 viewNear = worldUBO.cameraProjMatInv * nearPoint;
    viewNear /= viewNear.w;

    origin = vec3(worldUBO.cameraEffectedViewMatInv * vec4(0.0, 0.0, 0.0, 1.0));
    direction = volumetricRestirSafeNormalize(vec3(worldUBO.cameraEffectedViewMatInv * vec4(viewNear.xyz, 0.0)),
                                              vec3(0.0, 0.0, -1.0));
}

vec3 volumetricRestirCellCenterScenePosition(uvec3 coord, uvec3 gridSize, ivec2 fullResolution) {
    vec3 origin;
    vec3 direction;
    volumetricRestirBuildCameraRay(ivec2(coord.xy), fullResolution, origin, direction);
    return origin + direction * volumetricRestirFroxelDistance(coord.z, gridSize.z);
}

bool volumetricRestirCoordFromScenePosition(vec3 scenePosition,
                                            uvec3 gridSize,
                                            ivec2 fullResolution,
                                            out uvec3 coord) {
    coord = uvec3(0u);
    vec4 clip = worldUBO.cameraProjMat * worldUBO.cameraEffectedViewMat * vec4(scenePosition, 1.0);
    if (abs(clip.w) <= 1e-6) { return false; }

    vec3 ndc = clip.xyz / clip.w;
    if (any(lessThan(ndc.xy, vec2(-1.0))) || any(greaterThan(ndc.xy, vec2(1.0)))) { return false; }

    ivec2 lowPixel = ivec2(floor(((ndc.xy * 0.5 + 0.5) * vec2(fullResolution)) /
                                 float(max(int(ADV_VOLUMETRIC_RESTIR_DOWNSAMPLE_FACTOR), 1))));
    if (any(lessThan(lowPixel, ivec2(0))) || any(greaterThanEqual(lowPixel, ivec2(gridSize.xy)))) { return false; }

    float distanceValue = length(scenePosition);
    coord = uvec3(lowPixel, volumetricRestirFroxelZFromDistance(distanceValue, gridSize.z));
    return true;
}

bool volumetricRestirCoordFromPixelDistance(ivec2 lowPixel, float rayDistance, uvec3 gridSize, out uvec3 coord) {
    coord = uvec3(0u);
    if (any(lessThan(lowPixel, ivec2(0)))) {
        return false;
    }

    uint outputDownsample = max(uint(ADV_VOLUMETRIC_LIGHT_DOWNSAMPLE_FACTOR), 1u);
    uint restirDownsample = max(uint(ADV_VOLUMETRIC_RESTIR_DOWNSAMPLE_FACTOR), 1u);
    uvec2 fullPixel = uvec2(lowPixel) * outputDownsample + uvec2(outputDownsample / 2u);
    uvec2 restirPixel = fullPixel / restirDownsample;
    if (any(greaterThanEqual(restirPixel, gridSize.xy))) { return false; }

    coord = uvec3(restirPixel, volumetricRestirFroxelZFromDistance(rayDistance, gridSize.z));
    return true;
}

bool volumetricRestirPreviousCoordFromScenePosition(vec3 scenePosition,
                                                    uvec3 gridSize,
                                                    ivec2 fullResolution,
                                                    out uvec3 coord) {
    coord = uvec3(0u);
    vec3 worldPosition = scenePosition + vec3(worldUBO.cameraPos.xyz);
    vec3 previousScenePosition = worldPosition - vec3(lastWorldUBO.cameraPos.xyz);
    vec4 clip = lastWorldUBO.cameraProjMat * lastWorldUBO.cameraEffectedViewMat * vec4(previousScenePosition, 1.0);
    if (abs(clip.w) <= 1e-6) { return false; }

    vec3 ndc = clip.xyz / clip.w;
    if (any(lessThan(ndc.xy, vec2(-1.0))) || any(greaterThan(ndc.xy, vec2(1.0)))) { return false; }

    ivec2 lowPixel = ivec2(floor(((ndc.xy * 0.5 + 0.5) * vec2(fullResolution)) /
                                 float(max(int(ADV_VOLUMETRIC_RESTIR_DOWNSAMPLE_FACTOR), 1))));
    if (any(lessThan(lowPixel, ivec2(0))) || any(greaterThanEqual(lowPixel, ivec2(gridSize.xy)))) { return false; }

    coord = uvec3(lowPixel, volumetricRestirFroxelZFromDistance(length(previousScenePosition), gridSize.z));
    return true;
}

bool volumetricRestirTryGetNeighborhoodIndex(vec3 scenePosition, out uint centerChunkIndex) {
    centerChunkIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;

    ivec3 sectionCoordinate = ivec3(floor((scenePosition + vec3(worldUBO.cameraPos.xyz)) / 16.0));
    int resolvedCenterChunkIndex = -1;
    ivec3 centerChunkOrigin = ivec3(0);
    if (!tryGetChunkIndexFromSectionCoordinate(sectionCoordinate, worldUBO, resolvedCenterChunkIndex, centerChunkOrigin)) {
        return false;
    }

    ChunkLightData centerChunk = chunkPackedData[resolvedCenterChunkIndex];
    if (!chunkPackedDataMatchesOrigin(centerChunk, centerChunkOrigin)) { return false; }

    centerChunkIndex = uint(resolvedCenterChunkIndex);
    return chunkLightNeighborhoodCount(chunkLightNeighborhoods[centerChunkIndex]) > 0u;
}

bool volumetricRestirSourceExists(VolumetricRestirSample lightSample) {
    if (!volumetricRestirSampleValid(lightSample)) { return false; }
    uint chunkCount =
        uint(max(worldUBO.chunkGridInfo.x * worldUBO.chunkGridInfo.y * worldUBO.chunkGridInfo.z, 0));
    if (lightSample.sourceChunkIndex >= chunkCount) { return false; }

    ChunkLightData sourceChunk = chunkPackedData[lightSample.sourceChunkIndex];
    if (!chunkPackedDataHasLights(sourceChunk) || lightSample.sourceLightIndex >= sourceChunk.lightCount) {
        return false;
    }

    ChunkPackedLightBuffer lightBuffer = ChunkPackedLightBuffer(sourceChunk.lightBufferAddress);
    ChunkPackedLight chunkLight = lightBuffer.lights[lightSample.sourceLightIndex];
    return chunkPackedLightSourceID(chunkLight) == lightSample.sourceID;
}

bool volumetricRestirSourceAddressable(VolumetricRestirSample lightSample) {
    if (!volumetricRestirSampleValid(lightSample)) { return false; }

    ivec3 sectionCoord = ivec3(0);
    ivec3 chunkOrigin = ivec3(0);
    if (!tryGetSectionCoordinateFromChunkIndex(lightSample.sourceChunkIndex, worldUBO, sectionCoord, chunkOrigin)) {
        return false;
    }

    ChunkLightData sourceChunk = chunkPackedData[lightSample.sourceChunkIndex];
    return chunkPackedDataMatchesOrigin(sourceChunk, chunkOrigin) && chunkPackedDataHasLights(sourceChunk) &&
           lightSample.sourceLightIndex < sourceChunk.lightCount;
}

float volumetricRestirTargetFunction(vec3 scenePosition,
                                     vec3 viewDir,
                                     vec3 point,
                                     vec3 normal,
                                     vec3 emission,
                                     out vec3 dirToLight,
                                     out float dist) {
    dirToLight = vec3(0.0);
    dist = 0.0;

    vec3 sceneLightPoint = point - vec3(worldUBO.cameraPos.xyz);
    vec3 toLight = sceneLightPoint - scenePosition;
    float dist2 = dot(toLight, toLight);
    if (isnan(dist2) || isinf(dist2) || dist2 <= 1e-6) { return 0.0; }

    dist = sqrt(dist2);
    dirToLight = toLight / dist;
    if (any(isnan(dirToLight)) || any(isinf(dirToLight))) { return 0.0; }

    float lightNoL = max(dot(volumetricRestirSafeNormalize(normal, vec3(0.0, 1.0, 0.0)), -dirToLight), 0.0);
    if (lightNoL <= 1e-5) { return 0.0; }

    float influenceRadius = clamp(float(ADV_EMISSIVE_VOLUME_LIGHT_FADE_DISTANCE), 8.0, 128.0);
    float distanceFade = 1.0 - smoothstep(influenceRadius * 0.65, influenceRadius, dist);
    if (distanceFade <= 1e-4) { return 0.0; }

    vec3 positiveEmission = max(emission, vec3(0.0));
    float emissionLuma = volumetricRestirMax3(positiveEmission);
    if (!(emissionLuma > 1e-8)) { return 0.0; }

    float fogRange = max(worldUBO.fogEnd - worldUBO.fogStart, 1.0);
    float sigmaT = max(2.0 / fogRange, 0.0) * 0.02 * 0.45;
    float phase = volumetricRestirHenyeyGreenstein(clamp(dot(viewDir, dirToLight), -1.0, 1.0),
                                                   clamp(ADV_ATMOSPHERE_MIE_G * 0.35, -0.2, 0.35));
    phase = mix(INV_4_PI, min(phase, 4.0), 0.35);

    float attenuation = exp(-sigmaT * dist);
    float target = emissionLuma * lightNoL * phase * attenuation * distanceFade / max(dist2, 1e-6);
    return volumetricRestirFinitePositive(target);
}

vec3 volumetricRestirEvaluateRadiance(vec3 scenePosition,
                                      vec3 viewDir,
                                      VolumetricRestirSample lightSample,
                                      out vec3 dirToLight,
                                      out float dist) {
    float target = volumetricRestirTargetFunction(scenePosition, viewDir, lightSample.point, lightSample.normal,
                                                 lightSample.emission, dirToLight, dist);
    if (!(target > 1e-8) || !(lightSample.contributionWeight > 1e-8)) { return vec3(0.0); }

    float emissionLuma = max(volumetricRestirMax3(lightSample.emission), 1e-6);
    return ADV_DIRECT_LIGHT_STRENGTH * ADV_EMISSIVE_VOLUME_LIGHT_STRENGTH * lightSample.emission *
           (target / emissionLuma) * lightSample.contributionWeight;
}

VolumetricRestirSample volumetricRestirSampleFromChunkLight(ChunkPackedLight chunkLight,
                                                           uint sourceChunkIndex,
                                                           uint sourceLightIndex,
                                                           vec2 triangleXi,
                                                           float trianglePickXi) {
    VolumetricRestirSample lightSample = makeEmptyVolumetricRestirSample();

    vec3 p0 = chunkLight.p0Area.xyz;
    vec3 p1 = chunkLight.p1.xyz;
    vec3 p2 = chunkLight.p2.xyz;
    vec3 p3 = chunkLight.p3.xyz;
    float area0 = volumetricRestirTriangleArea(p0, p1, p2);
    float area1 = volumetricRestirTriangleArea(p0, p2, p3);
    float totalArea = max(area0 + area1, chunkLight.p0Area.w);
    if (!(totalArea > 1e-8)) { return lightSample; }

    bool firstTriangle = trianglePickXi * totalArea < area0;
    lightSample.sampleXi = clamp(triangleXi, vec2(0.0), vec2(1.0));
    lightSample.triangleSelector = firstTriangle ? 0.0 : 1.0;
    if (firstTriangle) {
        lightSample.point = volumetricRestirSampleTriangle(p0, p1, p2, lightSample.sampleXi);
        lightSample.normal = volumetricRestirSafeNormalize(cross(p1 - p0, p2 - p0), chunkLight.normal.xyz);
    } else {
        lightSample.point = volumetricRestirSampleTriangle(p0, p2, p3, lightSample.sampleXi);
        lightSample.normal = volumetricRestirSafeNormalize(cross(p2 - p0, p3 - p0), chunkLight.normal.xyz);
    }

    lightSample.emission = max(chunkLight.radiance.rgb, vec3(0.0));
    lightSample.sourceID = chunkPackedLightSourceID(chunkLight);
    lightSample.sourceLightIndex = sourceLightIndex;
    lightSample.sourceChunkIndex = sourceChunkIndex;
    lightSample.M = 1.0;
    lightSample.valid = true;
    return lightSample;
}

void volumetricRestirRisAddSample(inout VolumetricRestirRisState ris,
                                  VolumetricRestirSample lightSample,
                                  float weight,
                                  inout uint seed) {
    if (!(weight > 1e-8) || !volumetricRestirSampleValid(lightSample)) { return; }
    ris.weightSum += weight;
    if (!ris.hasSampleOut || rand(seed) * ris.weightSum < weight) {
        ris.sampleOut = lightSample;
        ris.hasSampleOut = true;
    }
}

VolumetricRestirSample volumetricRestirReservoirFromRis(VolumetricRestirRisState ris, float M) {
    if (!ris.hasSampleOut || !(M > 1e-8) || !(ris.sampleOut.targetFunction > 1e-8)) {
        return makeEmptyVolumetricRestirSample();
    }

    VolumetricRestirSample outSample = ris.sampleOut;
    outSample.M = min(M, float(ADV_VOLUMETRIC_RESTIR_M_CAP));
    outSample.contributionWeight = ris.weightSum / max(outSample.M * outSample.targetFunction, 1e-8);
    outSample.valid = volumetricRestirSampleValid(outSample);
    return outSample;
}

#endif
