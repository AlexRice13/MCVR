#include "common/constants.glsl"
#ifndef ADV_PARALLAX_CONDITION_GLSL
#define ADV_PARALLAX_CONDITION_GLSL

bool shouldTraceRestirParallax(float lod, vec3 hitWorldPos) {
    if (ADV_PARALLAX_QUALITY == 0) { return false; }
    vec3 cameraOrigin = vec3(worldUBO.cameraEffectedViewMatInv * vec4(0.0, 0.0, 0.0, 1.0));
    float cameraDistance = distance(hitWorldPos, cameraOrigin);
    return cameraDistance <= ADV_PARALLAX_CLOSE_DISTANCE && lod <= ADV_PARALLAX_MAX_LOD;
}

#endif
