#include "com_radiance_client_profiler_ProfilerOverlay.h"

#include "common/profiler.hpp"

#include <algorithm>
#include <sstream>

namespace {

std::string snapshotText(int maxRows) {
    auto snapshot = mcvr::profiler::Profiler::snapshot();
    maxRows = std::max(1, maxRows);

    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << "Radiance CPU Profiler  frame=" << snapshot.frameId << " dropped=" << snapshot.droppedEvents << '\n';

    for (const auto &counter : snapshot.counters) {
        out << counter.name << ": " << counter.value << '\n';
    }

    const int rows = std::min(maxRows, static_cast<int>(snapshot.zones.size()));
    for (int i = 0; i < rows; ++i) {
        const auto &zone = snapshot.zones[i];
        out << i + 1 << ". " << zone.name << "  total=" << zone.totalMilliseconds
            << "ms avg=" << zone.averageMilliseconds << "ms calls=" << zone.callCount << '\n';
    }
    return out.str();
}

} // namespace

JNIEXPORT void JNICALL Java_com_radiance_client_profiler_ProfilerOverlay_nativeSetEnabled(JNIEnv *, jclass,
                                                                                          jboolean enabled) {
    mcvr::profiler::Profiler::setEnabled(enabled == JNI_TRUE);
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_profiler_ProfilerOverlay_nativeIsEnabled(JNIEnv *, jclass) {
    return mcvr::profiler::Profiler::isEnabled() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_radiance_client_profiler_ProfilerOverlay_nativeReset(JNIEnv *, jclass) {
    mcvr::profiler::Profiler::reset();
}

JNIEXPORT jstring JNICALL Java_com_radiance_client_profiler_ProfilerOverlay_nativeSnapshotText(JNIEnv *env, jclass,
                                                                                               jint maxRows) {
    const std::string text = snapshotText(maxRows);
    return env->NewStringUTF(text.c_str());
}
