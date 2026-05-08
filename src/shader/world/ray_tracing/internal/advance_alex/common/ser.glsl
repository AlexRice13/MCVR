#ifndef ADV_SER_GLSL
#define ADV_SER_GLSL

#ifndef ADV_ENABLE_SER
#    define ADV_ENABLE_SER 0
#endif

#ifndef MCVR_DEVICE_RAY_TRACING_INVOCATION_REORDER
#    define MCVR_DEVICE_RAY_TRACING_INVOCATION_REORDER 0
#endif

#ifndef MCVR_COMPILER_RAY_TRACING_INVOCATION_REORDER
#    define MCVR_COMPILER_RAY_TRACING_INVOCATION_REORDER 0
#endif

#ifndef ADV_SER_RAYGEN_STAGE
#    define ADV_SER_RAYGEN_STAGE 0
#endif

#if ADV_SER_RAYGEN_STAGE != 0 && ADV_ENABLE_SER != 0 && MCVR_DEVICE_RAY_TRACING_INVOCATION_REORDER != 0 && MCVR_COMPILER_RAY_TRACING_INVOCATION_REORDER != 0
#extension GL_NV_shader_invocation_reorder : enable
#    define ADV_SER_REORDER_ENABLED 1
#else
#    define ADV_SER_REORDER_ENABLED 0
#endif

void advReorderThreadForPom(bool traceLocalHeight, bool hasFftWaterSurface) {
#if ADV_SER_REORDER_ENABLED != 0
    uint hint = traceLocalHeight ? 1u : (hasFftWaterSurface ? 2u : 0u);
    reorderThreadNV(hint, 0u);
#endif
}

#endif
