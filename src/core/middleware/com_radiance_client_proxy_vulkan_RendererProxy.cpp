#include "com_radiance_client_proxy_vulkan_RendererProxy.h"

#include "core/all_extern.hpp"
#include "core/render/buffers.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/scenario_color_grading.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

#include <vector>

#if defined(_WIN32)
#    include <windows.h>
using DYNLIB_HANDLE = HMODULE;

static DYNLIB_HANDLE try_get_loaded_handle(const wchar_t *wname) {
    return GetModuleHandleW(wname);
}

static FARPROC getproc(DYNLIB_HANDLE h, const char *sym) {
    FARPROC p = GetProcAddress(h, sym);
    if (!p) {
        std::cerr << "GetProcAddress failed: " << sym << std::endl;
        std::abort();
    }
    return p;
}

#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#    include <dlfcn.h>
using DYNLIB_HANDLE = void *;

static DYNLIB_HANDLE try_get_loaded_handle(const char *name) {
    return dlopen(name, RTLD_NOW | RTLD_NOLOAD);
}

static void *getproc(DYNLIB_HANDLE h, const char *sym) {
    void *p = dlsym(h, sym);
    if (!p) {
        std::cerr << "dlsym failed: " << sym << " — " << dlerror() << std::endl;
        std::abort();
    }
    return p;
}

#else
#    error "Unsupported platform"
#endif

static DYNLIB_HANDLE bind_handle_from_candidates(JNIEnv *env, jobjectArray jnames) {
    jsize n = env->GetArrayLength(jnames);
    if (n == 0) return nullptr;
#if defined(_WIN32)
    for (jsize i = 0; i < n; ++i) {
        jstring s = (jstring)env->GetObjectArrayElement(jnames, i);
        const jchar *w = env->GetStringChars(s, nullptr);
        DYNLIB_HANDLE h = try_get_loaded_handle(reinterpret_cast<const wchar_t *>(w));
        env->ReleaseStringChars(s, w);
        env->DeleteLocalRef(s);
        if (h) return h;
    }
#else
    for (jsize i = 0; i < n; ++i) {
        jstring s = (jstring)env->GetObjectArrayElement(jnames, i);
        const char *c = env->GetStringUTFChars(s, nullptr);
        DYNLIB_HANDLE h = try_get_loaded_handle(c);
        env->ReleaseStringUTFChars(s, c);
        env->DeleteLocalRef(s);
        if (h) return h;
    }
#endif
    return nullptr;
}

static void bind_symbols(DYNLIB_HANDLE h) {
#if defined(_WIN32)
    auto gp = [&](const char *sym) { return getproc(h, sym); };
#else
    auto gp = [&](const char *sym) { return getproc(h, sym); };
#endif
    p_glfwInit = reinterpret_cast<PFN_glfwInit>(gp("glfwInit"));
    p_glfwTerminate = reinterpret_cast<PFN_glfwTerminate>(gp("glfwTerminate"));
    p_glfwGetWindowSize = reinterpret_cast<PFN_glfwGetWindowSize>(gp("glfwGetWindowSize"));
    p_glfwCreateWindowSurface = reinterpret_cast<PFN_glfwCreateWindowSurface>(gp("glfwCreateWindowSurface"));
    p_glfwGetRequiredInstanceExtensions =
        reinterpret_cast<PFN_glfwGetRequiredInstanceExtensions>(gp("glfwGetRequiredInstanceExtensions"));
    p_glfwSetWindowTitle = reinterpret_cast<PFN_glfwSetWindowTitle>(gp("glfwSetWindowTitle"));
    p_glfwSetFramebufferSizeCallback =
        reinterpret_cast<PFN_glfwSetFramebufferSizeCallback>(gp("glfwSetFramebufferSizeCallback"));
    p_glfwGetFramebufferSize = reinterpret_cast<PFN_glfwGetFramebufferSize>(gp("glfwGetFramebufferSize"));
    p_glfwWaitEvents = reinterpret_cast<PFN_glfwWaitEvents>(gp("glfwWaitEvents"));
}

static std::u16string JStringToU16(JNIEnv* env, jstring jstr) {
    if (!jstr) return {};
    const jchar* chars = env->GetStringChars(jstr, nullptr);
    jsize len = env->GetStringLength(jstr);
    std::u16string u16(reinterpret_cast<const char16_t*>(chars),
                       reinterpret_cast<const char16_t*>(chars) + len);
    env->ReleaseStringChars(jstr, chars);
    return u16;
}

static std::filesystem::path JStringToPath(JNIEnv* env, jstring jstr) {
    std::u16string u16 = JStringToU16(env, jstr);
    return std::filesystem::path(u16);
}

static std::string JStringToUtf8(JNIEnv *env, jstring jstr) {
    if (jstr == nullptr) return "";
    const char *chars = env->GetStringUTFChars(jstr, nullptr);
    if (chars == nullptr) return "";
    std::string out(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return out;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initFolderPath(JNIEnv *env,
                                                                                          jclass,
                                                                                          jstring folderPath) {
    if (folderPath == NULL) { return; }

    Renderer::folderPath = JStringToU16(env, folderPath);
    if (!ScenarioColorGradingManager::is_initialized()) { ScenarioColorGradingManager::init(); }
    ScenarioColorGradingManager::instance().setConfigPath(Renderer::folderPath / "scenarios_config.ini");
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initRenderer(JNIEnv *env,
                                                                                        jclass,
                                                                                        jobjectArray candidates,
                                                                                        jlong windowHandle) {
    DYNLIB_HANDLE h = bind_handle_from_candidates(env, candidates);
    if (!h) {
        std::cerr << "[GLFW-Bind] Could not find already-loaded GLFW via NOLOAD/GetModuleHandle."
                     " Ensure Java(LWJGL) loads GLFW before JNI and pass correct names/paths."
                  << std::endl;
        std::abort();
    }
    bind_symbols(h);

    GLFWwindow *window = (GLFWwindow *)(intptr_t)windowHandle;
    Renderer::init(window);
    Renderer::instance().framework()->acquireContext();
}

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_maxSupportedTextureSize(JNIEnv *, jclass) {
    auto maxImageSize = Renderer::instance().framework()->physicalDevice()->properties().limits.maxImageDimension2D;
    return maxImageSize;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_acquireContext(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->acquireContext();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_submitCommand(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->submitCommand();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_present(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->present();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_drawOverlay(
    JNIEnv *, jclass, jint vertexId, jint indexId, jint pipelineType, jint indexCount, jint indexType) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto vertexBuffer = Renderer::instance().buffers()->getBuffer(vertexId);
    auto indexBuffer = Renderer::instance().buffers()->getBuffer(indexId);
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->drawIndexed(vertexBuffer, indexBuffer,
                                                  static_cast<OverlayDrawPipelineType>(pipelineType), indexCount,
                                                  static_cast<VkIndexType>(indexType));
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_updateScenarioSceneContext(
    JNIEnv *env,
    jclass,
    jstring dimensionKey,
    jstring biomeKey,
    jlong timeOfDay,
    jboolean raining,
    jboolean thundering,
    jstring submersion,
    jboolean indoors,
    jboolean cave) {
    if (!ScenarioColorGradingManager::is_initialized()) return;
    ScenarioColorGradingManager::instance().updateSceneContext(
        JStringToUtf8(env, dimensionKey),
        JStringToUtf8(env, biomeKey),
        static_cast<uint32_t>(timeOfDay),
        raining == JNI_TRUE,
        thundering == JNI_TRUE,
        JStringToUtf8(env, submersion),
        indoors == JNI_TRUE,
        cave == JNI_TRUE);
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_saveScenarioColorGrading(
    JNIEnv *env,
    jclass,
    jstring scenarioName,
    jint priority,
    jboolean saveWorld,
    jboolean saveTime,
    jboolean saveWeather,
    jboolean saveBiome,
    jboolean saveSubmersion,
    jboolean saveIndoor,
    jboolean saveCave,
    jint timeStart,
    jint timeEnd,
    jobjectArray attributePairs) {
    if (!ScenarioColorGradingManager::is_initialized()) return JNI_FALSE;

    std::vector<std::string> pairs;
    if (attributePairs != nullptr) {
        jsize count = env->GetArrayLength(attributePairs);
        pairs.reserve(static_cast<size_t>(count));
        for (jsize i = 0; i < count; ++i) {
            auto element = static_cast<jstring>(env->GetObjectArrayElement(attributePairs, i));
            pairs.push_back(JStringToUtf8(env, element));
            env->DeleteLocalRef(element);
        }
    }

    ScenarioSaveMetadataSelection selection{};
    selection.world = saveWorld == JNI_TRUE;
    selection.time = saveTime == JNI_TRUE;
    selection.weather = saveWeather == JNI_TRUE;
    selection.biome = saveBiome == JNI_TRUE;
    selection.submersion = saveSubmersion == JNI_TRUE;
    selection.indoor = saveIndoor == JNI_TRUE;
    selection.cave = saveCave == JNI_TRUE;
    selection.timeStart = static_cast<int>(timeStart);
    selection.timeEnd = static_cast<int>(timeEnd);

    bool saved = ScenarioColorGradingManager::instance().saveScenario(
        JStringToUtf8(env, scenarioName),
        priority,
        selection,
        Renderer::options.hdrActive,
        pairs);
    return saved ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_applyPreviewScenarioColorGrading(
    JNIEnv *env, jclass, jobjectArray attributePairs) {
    if (!ScenarioColorGradingManager::is_initialized()) return;

    ToneMappingSettings previewSettings = createDefaultToneMappingSettings();
    std::vector<std::string> pairs;
    if (attributePairs != nullptr) {
        jsize count = env->GetArrayLength(attributePairs);
        pairs.reserve(static_cast<size_t>(count));
        for (jsize i = 0; i < count; ++i) {
            auto element = static_cast<jstring>(env->GetObjectArrayElement(attributePairs, i));
            pairs.push_back(JStringToUtf8(env, element));
            env->DeleteLocalRef(element);
        }
    }

    for (size_t i = 0; i + 1 < pairs.size(); i += 2) {
        applyToneMappingAttributeKV(previewSettings, pairs[i], pairs[i + 1]);
    }

    ScenarioColorGradingManager::instance().setPreviewSettings(previewSettings);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_clearPreviewScenarioColorGrading(
    JNIEnv *, jclass) {
    if (!ScenarioColorGradingManager::is_initialized()) return;
    ScenarioColorGradingManager::instance().clearPreviewSettings();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_setScenarioGradingIsolation(
    JNIEnv *, jclass, jboolean enabled) {
    Renderer::options.scenarioGradingIsolation = enabled == JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_fuseWorld(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->fuseWorld();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_postBlur(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto world = Renderer::instance().world();
    if (world != nullptr && world->shouldRender()) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->postBlur(6);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_close(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    Renderer::instance().close();
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_RendererProxy_shouldRenderWorld(JNIEnv *, jclass, jboolean shouldRenderWorld) {
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->shouldRender() = shouldRenderWorld;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_takeScreenshot(
    JNIEnv *, jclass, jboolean withUI, jint width, jint height, jint channel, jlong pointer) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->takeScreenshot(withUI, width, height, channel, reinterpret_cast<void *>(pointer));
}
