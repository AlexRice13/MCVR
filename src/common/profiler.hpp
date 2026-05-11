#pragma once

#ifndef RADIANCE_PROFILER_ENABLED
#    define RADIANCE_PROFILER_ENABLED 1
#endif

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace mcvr::profiler {

struct ZoneStats {
    std::string name;
    uint64_t callCount = 0;
    uint64_t totalTicks = 0;
    uint64_t minTicks = 0;
    uint64_t maxTicks = 0;
    double totalMilliseconds = 0.0;
    double averageMilliseconds = 0.0;
};

struct CounterStats {
    std::string name;
    uint64_t value = 0;
};

struct ProfilerSnapshot {
    uint64_t frameId = 0;
    uint64_t droppedEvents = 0;
    std::vector<ZoneStats> zones;
    std::vector<CounterStats> counters;
};

#if RADIANCE_PROFILER_ENABLED

class Profiler {
  public:
    static std::atomic_bool enabled;

    static void setEnabled(bool value) noexcept;
    static bool isEnabled() noexcept;
    static void refreshCurrentThread() noexcept;
    static void beginFrame() noexcept;
    static void endFrame() noexcept;
    static void flushCurrentThread();
    static void reset();
    static ProfilerSnapshot snapshot();
    static void setCounter(const char *name, uint64_t value);
    static uint64_t frameId() noexcept;
    static void setAggregationFrameInterval(uint32_t frames) noexcept;
    static double ticksToMilliseconds(uint64_t ticks) noexcept;
};

class ScopedZone {
  public:
    explicit ScopedZone(const char *name);
    ~ScopedZone();

    ScopedZone(const ScopedZone &) = delete;
    ScopedZone &operator=(const ScopedZone &) = delete;
    ScopedZone(ScopedZone &&) = delete;
    ScopedZone &operator=(ScopedZone &&) = delete;

  private:
    uint32_t nameId_ = 0;
    uint64_t beginTicks_ = 0;
    bool active_ = false;
};

class FrameScope {
  public:
    FrameScope() noexcept;
    ~FrameScope() noexcept;

    FrameScope(const FrameScope &) = delete;
    FrameScope &operator=(const FrameScope &) = delete;
    FrameScope(FrameScope &&) = delete;
    FrameScope &operator=(FrameScope &&) = delete;
};

#else

class Profiler {
  public:
    inline static std::atomic_bool enabled{false};

    static void setEnabled(bool) noexcept {}
    static bool isEnabled() noexcept { return false; }
    static void refreshCurrentThread() noexcept {}
    static void beginFrame() noexcept {}
    static void endFrame() noexcept {}
    static void flushCurrentThread() {}
    static void reset() {}
    static ProfilerSnapshot snapshot() { return {}; }
    static void setCounter(const char *, uint64_t) {}
    static uint64_t frameId() noexcept { return 0; }
    static void setAggregationFrameInterval(uint32_t) noexcept {}
    static double ticksToMilliseconds(uint64_t) noexcept { return 0.0; }
};

class ScopedZone {
  public:
    explicit ScopedZone(const char *) noexcept {}
};

class FrameScope {
  public:
    FrameScope() noexcept {}
    ~FrameScope() noexcept {}
};

#endif

} // namespace mcvr::profiler

#define RAD_PROFILE_CONCAT_IMPL(a, b) a##b
#define RAD_PROFILE_CONCAT(a, b) RAD_PROFILE_CONCAT_IMPL(a, b)

#if RADIANCE_PROFILER_ENABLED
#    define RAD_PROFILE_THREAD() ::mcvr::profiler::Profiler::refreshCurrentThread()
#    define RAD_PROFILE_COUNTER(name, value) ::mcvr::profiler::Profiler::setCounter(name, static_cast<uint64_t>(value))
#    define RAD_PROFILE_SCOPE(name)                                                                                    \
        ::mcvr::profiler::ScopedZone RAD_PROFILE_CONCAT(__rad_profile_scope_, __LINE__)(name)
#    define RAD_PROFILE_FRAME()                                                                                        \
        ::mcvr::profiler::FrameScope RAD_PROFILE_CONCAT(__rad_profile_frame_, __LINE__)
#else
#    define RAD_PROFILE_THREAD() ((void)0)
#    define RAD_PROFILE_COUNTER(name, value) ((void)0)
#    define RAD_PROFILE_SCOPE(name) ((void)0)
#    define RAD_PROFILE_FRAME() ((void)0)
#endif
