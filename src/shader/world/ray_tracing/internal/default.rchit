#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common/shared.hpp"
#include "util/disney.glsl"
#include "util/alpha_mode.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/sampling_helpers.glsl"
#include "util/util.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 3) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 5) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 10) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 1, binding = 9) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

layout(set = 1, binding = 8) readonly buffer LastPositionBufferAddr {
    uint64_t addrs[];
}
lastPositionBufferAddrs;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    float directLightStrength;
    float indirectLightStrength;
    float basicRadiance;
    uint pbrSamplingMode;
    uint transparentSplitMode;
    float rainWetnessThreshold;
    uint volumetricFogEnabled;
    float volumetricFogStrength;
    uint volumetricFogSamplingMode;
    uint transparentRefractionSamplingMode;
    uint pad1;
}
pc;

layout(location = 0) rayPayloadInEXT MainRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

vec3 calculateNormal(
    vec3 p0, vec3 p1, vec3 p2, vec2 uv0, vec2 uv1, vec2 uv2, vec3 matNormal, vec3 viewDir, out vec3 geoNormal) {
    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 geoNormalObj = normalize(cross(edge1, edge2));

    mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
    geoNormal = normalize(normalMatrix * geoNormalObj);
    if (dot(viewDir, geoNormal) < 0.0) geoNormal = -geoNormal;

    if (any(isnan(matNormal))) { return geoNormal; }

    vec2 deltaUV1 = uv1 - uv0;
    vec2 deltaUV2 = uv2 - uv0;
    float det = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;

    vec3 tangentObj;
    if (abs(det) < 1e-6) {
        tangentObj = (abs(geoNormalObj.x) > 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    } else {
        float f = 1.0 / det;
        tangentObj.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangentObj.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangentObj.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
    }

    vec3 TObj = normalize(tangentObj - geoNormalObj * dot(geoNormalObj, tangentObj));
    vec3 BObj = cross(geoNormalObj, TObj);

    vec3 T = normalize(normalMatrix * TObj);
    vec3 B = normalize(normalMatrix * BObj);
    vec3 N = geoNormal;

    vec3 correctedLocalNormal = matNormal;
    correctedLocalNormal.y = -correctedLocalNormal.y;

    vec3 finalNormal = normalize(T * correctedLocalNormal.x + B * correctedLocalNormal.y + N * correctedLocalNormal.z);
    if (dot(viewDir, finalNormal) < 0.0) return geoNormal;
    return finalNormal;
}

bool loadPreviousScenePos(uint geometryBufferIndex, uint primitiveID, vec3 baryCoords, out vec3 prevScenePos) {
    uint64_t lastPositionAddr = lastPositionBufferAddrs.addrs[geometryBufferIndex];
    uint64_t lastIndexAddr = lastIndexBufferAddrs.addrs[geometryBufferIndex];
    if (lastPositionAddr == 0 || lastIndexAddr == 0) { return false; }

    IndexBuffer lastIndexBuffer = IndexBuffer(lastIndexAddr);
    uint indexBaseID = 3u * primitiveID;
    uint i0 = lastIndexBuffer.indices[indexBaseID];
    uint i1 = lastIndexBuffer.indices[indexBaseID + 1u];
    uint i2 = lastIndexBuffer.indices[indexBaseID + 2u];

    PositionBuffer lastPositionBuffer = PositionBuffer(lastPositionAddr);
    vec3 p0 = lastPositionBuffer.vertices[i0].pos;
    vec3 p1 = lastPositionBuffer.vertices[i1].pos;
    vec3 p2 = lastPositionBuffer.vertices[i2].pos;
    vec3 prevLocalPos = baryCoords.x * p0 + baryCoords.y * p1 + baryCoords.z * p2;

    mat4 lastModelMat = lastObjToWorldMats.mat[gl_InstanceCustomIndexEXT];
    prevScenePos = mat3(lastModelMat) * prevLocalPos + lastModelMat[3].xyz;
    return true;
}

uint precipitationSeed(vec3 worldPos, vec2 uv, uint textureID) {
    return xxhash32(uvec3(floatBitsToUint(worldPos.x * 0.25 + uv.x * 17.0 + float(textureID) * 0.011),
                          floatBitsToUint(worldPos.y * 0.5 + uv.y * 31.0 + float(textureID) * 0.007),
                          floatBitsToUint(worldPos.z * 0.25 + uv.x * 13.0 + uv.y * 7.0 +
                                          float(textureID) * 0.003)));
}

vec3 buildPrecipitationNormal(vec3 baseNormal, vec3 viewDir, vec3 worldPos, vec2 uv, uint textureID) {
    vec3 tangent, bitangent;
    Onb(baseNormal, tangent, bitangent);

    uint seed = precipitationSeed(worldPos, uv, textureID);
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
    if (dot(perturbed, viewDir) < 0.05) {
        perturbed = normalize(mix(baseNormal, perturbed, 0.4));
    }
    return dot(perturbed, viewDir) > 0.0 ? perturbed : baseNormal;
}

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;
    uint geometryBufferIndex = getGeometryBufferIndex(instanceID, geometryID);

    uint i0, i1, i2;
    MaterialVertex m0, m1, m2;
    loadTriangleIndices(geometryBufferIndex, gl_PrimitiveID, i0, i1, i2);
    loadTriangleMaterial(geometryBufferIndex, i0, i1, i2, m0, m1, m2);

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    uint packedData = m0.packedData;
    bool useColorLayer = hasColorLayer(packedData);
    bool useTexture = hasTexture(packedData);
    bool useGlint = hasGlint(packedData);
    bool useOverlay = hasOverlay(packedData);
    vec4 colorLayerValue = useColorLayer ?
                               baryCoords.x * m0.colorLayer + baryCoords.y * m1.colorLayer + baryCoords.z * m2.colorLayer :
                               vec4(1.0);
    vec3 colorLayer = colorLayerValue.rgb;
    uint bounce = rayBounce(mainRay);

    float albedoEmission =
        baryCoords.x * m0.albedoEmission + baryCoords.y * m1.albedoEmission + baryCoords.z * m2.albedoEmission;
    uint textureID = m0.textureID;
    uint alphaMode = getAlphaMode(packedData);

    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    vec2 textureUV = vec2(0.0);
    PositionVertex p0, p1, p2;
    bool hasPositions = false;
    if (useTexture) {
        TextureMapEntry textureMap = mapping.entries[textureID];
        textureUV = baryCoords.x * m0.textureUV + baryCoords.y * m1.textureUV + baryCoords.z * m2.textureUV;
        vec2 atlasUvMin = min(m0.textureUV, min(m1.textureUV, m2.textureUV));
        vec2 atlasUvMax = max(m0.textureUV, max(m1.textureUV, m2.textureUV));

        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        loadTrianglePositions(geometryBufferIndex, i0, i1, i2, p0, p1, p2);
        hasPositions = true;
        vec3 dposdu, dposdv;
        computedposduDv(p0.pos, p1.pos, p2.pos, m0.textureUV, m1.textureUV, m2.textureUV, dposdu, dposdv);
        float lod = lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);

        albedoValue = sampleTexture(textures[nonuniformEXT(textureID)], textureUV, lod, false);
        albedoValue.a = resolveSurfaceAlpha(albedoValue.a * colorLayerValue.a, alphaMode);
        specularValue = textureMap.specular >= 0 ?
                            sampleTexture(textures[nonuniformEXT(textureMap.specular)], textureUV, lod, false) :
                            vec4(0.0);
        normalValue = textureMap.normal >= 0 ?
                          samplePBRTexture(textures[nonuniformEXT(textureMap.normal)], textureUV, atlasUvMin, atlasUvMax,
                                           lod, pc.pbrSamplingMode) :
                          vec4(0.0);
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.0);
    }

    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;
    mainRay.directLightRadiance = vec3(0.0);
    mainRay.hasPrevScenePos = 0u;
    if (bounce == 0u) {
        vec3 prevScenePos;
        if (loadPreviousScenePos(geometryBufferIndex, gl_PrimitiveID, baryCoords, prevScenePos)) {
            mainRay.prevScenePos = prevScenePos;
            mainRay.hasPrevScenePos = 1u;
        }
    }

    vec3 glint = vec3(0.0);
    if (useGlint) {
        vec2 glintUV = baryCoords.x * m0.glintUV + baryCoords.y * m1.glintUV + baryCoords.z * m2.glintUV;
        glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
        glint = sampleTexture(textures[nonuniformEXT(m0.glintTexture)], glintUV, false).rgb;
    }
    glint *= glint;

    vec3 tint = albedoValue.rgb * colorLayer + glint;
    if (useOverlay) {
        vec4 overlayColor = sampleTexture(textures[nonuniformEXT(worldUbo.overlayTextureID)], m0.overlayUV, 0, false);
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);
    applyRainWetness(mat,
                     computeRainWetnessFactor(skyUBO.rainGradient, hasRainExposed(packedData), pc.rainWetnessThreshold));
    if (hasRainPrecipitationMaterial(packedData)) {
        uint rainSeed = precipitationSeed(worldPos, textureUV, textureID);
        float rainIor = mix(1.16, 1.52, rand(rainSeed));
        float rainF0 = pow((rainIor - 1.0) / max(rainIor + 1.0, 1e-4), 2.0);
        albedoValue.rgb = vec3(1.0);
        albedoValue.a = max(albedoValue.a, mix(0.12, 0.32, rand(rainSeed)));
        mat.albedo = vec3(1.0);
        mat.f0 = vec3(rainF0);
        mat.roughness = mix(0.004, 0.045, rand(rainSeed));
        mat.metallic = 0.0;
        mat.transmission = 1.0;
        mat.ior = rainIor;
        mat.emission = 0.0;
    }

    if (!hasPositions) {
        loadTrianglePositions(geometryBufferIndex, i0, i1, i2, p0, p1, p2);
    }
    vec3 geoNormal;
    vec3 normal =
        calculateNormal(p0.pos, p1.pos, p2.pos, m0.textureUV, m1.textureUV, m2.textureUV, mat.normal, viewDir, geoNormal);
    if (hasRainPrecipitationMaterial(packedData)) {
        normal = buildPrecipitationNormal(normal, viewDir, worldPos, textureUV, textureID);
    }

    float factor = bounce == 0u ? pc.directLightStrength : pc.indirectLightStrength;
    vec3 emissionRadiance = factor * tint * mat.emission * mainRay.throughput;
    emissionRadiance += tint * albedoEmission * mainRay.throughput;
    mainRay.radiance += emissionRadiance;

    bool isOpaqueSurface = mat.transmission <= EPS;
    if (worldUbo.skyType == 1) {
        vec3 lightDir = normalize(skyUBO.sunDirection);
        if (lightDir.y < 0.0) { lightDir = -lightDir; }
        vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, 1.0 / max(skyUBO.sunAngularRadius * skyUBO.sunAngularRadius, 1e-6));
        float sampledLightNoL = dot(sampledLightDir, geoNormal);
        if (!isOpaqueSurface || sampledLightNoL > 0.0) {
            float lightPdf;
            vec3 lightBRDF = DisneyEval(mat, viewDir, normal, sampledLightDir, lightPdf);
            if (lightPdf > 1e-6) {
                vec3 shadowRayOrigin = worldPos + (sampledLightNoL > 0.0 ? geoNormal : -geoNormal) * 0.0001;
                shadowRay.radiance = vec3(0.0);
                shadowRay.throughput = vec3(1.0);
                shadowRay.insideBoat = rayInsideBoat(mainRay) ? 1u : 0u;

                traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT, WORLD_MASK | PLAYER_MASK | CLOUD_MASK, 0, 0, 0, shadowRayOrigin,
                            0.0001, sampledLightDir, 1000.0, 1);

                float progress = skyUBO.rainGradient;
                vec3 lightRadiance = shadowRay.radiance * mainRay.throughput * lightBRDF;
                mainRay.directLightRadiance = mix(lightRadiance, vec3(0.0), progress);
                mainRay.radiance += mainRay.directLightRadiance;
            }
        }
    }

    mainRay.normal = normal;
    rayStoreMaterial(mainRay, albedoValue, mat.f0, mat.roughness, mat.metallic, mat.transmission, mat.ior, mat.emission);

    vec3 sampleDir;
    float pdf;
    uint lobeType;
    vec3 bsdf = DisneySample(mat, viewDir, normal, sampleDir, pdf, mainRay.seed, lobeType);

    raySetLobeType(mainRay, lobeType);
    raySetNoisy(mainRay, true);

    if (pdf <= 1e-6) {
        raySetStop(mainRay, true);
        return;
    }

    if (isOpaqueSurface && dot(sampleDir, geoNormal) <= 0.0) {
        raySetStop(mainRay, true);
        return;
    }

    mainRay.throughput *= bsdf / max(pdf, 1e-4);

    vec3 offsetDir = dot(sampleDir, geoNormal) > 0.0 ? geoNormal : -geoNormal;
    mainRay.origin = worldPos + offsetDir * 0.0001;
    mainRay.direction = sampleDir;
    raySetStop(mainRay, false);
}
