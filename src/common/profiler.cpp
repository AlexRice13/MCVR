#include "common/profiler.hpp"

#if RADIANCE_PROFILER_ENABLED

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <Windows.h>
#endif

namespace mcvr::profiler {

std::atomic_bool Profiler::enabled{false};

namespace {

constexpr uint32_t kMaxZoneNames = 2048;
constexpr uint32_t kMaxThreadEvents = 32768;
constexpr uint32_t kDefaultAggregationFrameInterval = 30;

struct ZoneEvent {
    uint32_t nameId = 0;
    uint64_t beginTicks = 0;
    uint64_t endTicks = 0;
};

struct AggregatedZone {
    uint64_t callCount = 0;
    uint64_t totalTicks = 0;
    uint64_t minTicks = std::numeric_limits<uint64_t>::max();
    uint64_t maxTicks = 0;
};

struct NameRegistry {
    std::mutex mutex;
    std::array<const char *, kMaxZoneNames> names{};
    uint32_t count = 0;
};

struct AggregateRegistry {
    std::mutex mutex;
    std::array<AggregatedZone, kMaxZoneNames> zones{};
    std::array<uint64_t, kMaxZoneNames> counters{};
    std::array<bool, kMaxZoneNames> counterSet{};
    uint64_t droppedEvents = 0;
};

struct ThreadState {
    std::array<ZoneEvent, kMaxThreadEvents> events{};
    uint32_t eventCount = 0;
    uint64_t droppedEvents = 0;
    uint32_t framesSinceFlush = 0;
    bool active = false;

    ~ThreadState();
};

std::atomic_uint64_t gFrameId{0};
std::atomic_uint32_t gAggregationFrameInterval{kDefaultAggregationFrameInterval};

NameRegistry &nameRegistry() {
    static auto *registry = new NameRegistry();
    return *registry;
}

AggregateRegistry &aggregateRegistry() {
    static auto *registry = new AggregateRegistry();
    return *registry;
}

ThreadState &threadState() {
    thread_local ThreadState state;
    return state;
}

uint64_t readTicks() noexcept {
#if defined(_WIN32)
    LARGE_INTEGER value{};
    QueryPerformanceCounter(&value);
    return static_cast<uint64_t>(value.QuadPart);
#else
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
#endif
}

uint64_t ticksFrequency() noexcept {
#if defined(_WIN32)
    static const uint64_t frequency = [] {
        LARGE_INTEGER value{};
        QueryPerformanceFrequency(&value);
        return static_cast<uint64_t>(value.QuadPart);
    }();
    return frequency;
#else
    return 1000000000ull;
#endif
}

uint32_t registerName(const char *name) {
    if (name == nullptr) { name = "<null>"; }

    auto &registry = nameRegistry();
    std::lock_guard lock(registry.mutex);

    for (uint32_t i = 0; i < registry.count; ++i) {
        if (std::strcmp(registry.names[i], name) == 0) { return i; }
    }

    if (registry.count >= kMaxZoneNames) { return kMaxZoneNames - 1; }

    const uint32_t id = registry.count++;
    registry.names[id] = name;
    return id;
}

void recordEvent(ThreadState &state, uint32_t nameId, uint64_t beginTicks, uint64_t endTicks) noexcept {
    if (state.eventCount >= kMaxThreadEvents) {
        ++state.droppedEvents;
        return;
    }

    state.events[state.eventCount++] = ZoneEvent{
        .nameId = nameId,
        .beginTicks = beginTicks,
        .endTicks = endTicks,
    };
}

void flushThread(ThreadState &state) {
    if (state.eventCount == 0 && state.droppedEvents == 0) { return; }

    auto &aggregate = aggregateRegistry();
    std::lock_guard lock(aggregate.mutex);

    for (uint32_t i = 0; i < state.eventCount; ++i) {
        const ZoneEvent &event = state.events[i];
        const uint64_t elapsed = event.endTicks >= event.beginTicks ? event.endTicks - event.beginTicks : 0;
        AggregatedZone &zone = aggregate.zones[event.nameId];
        ++zone.callCount;
        zone.totalTicks += elapsed;
        zone.minTicks = std::min(zone.minTicks, elapsed);
        zone.maxTicks = std::max(zone.maxTicks, elapsed);
    }

    aggregate.droppedEvents += state.droppedEvents;
    state.eventCount = 0;
    state.droppedEvents = 0;
}

ThreadState::~ThreadState() {
    flushThread(*this);
}

} // namespace

void Profiler::setEnabled(bool value) noexcept {
    enabled.store(value, std::memory_order_relaxed);
}

bool Profiler::isEnabled() noexcept {
    return enabled.load(std::memory_order_relaxed);
}

void Profiler::refreshCurrentThread() noexcept {
    threadState().active = enabled.load(std::memory_order_relaxed);
}

void Profiler::beginFrame() noexcept {
    ThreadState &state = threadState();
    state.active = enabled.load(std::memory_order_relaxed);
    gFrameId.fetch_add(1, std::memory_order_relaxed);

    const uint32_t interval = std::max(1u, gAggregationFrameInterval.load(std::memory_order_relaxed));
    if (++state.framesSinceFlush >= interval) {
        state.framesSinceFlush = 0;
        flushThread(state);
    }
}

void Profiler::endFrame() noexcept {}

void Profiler::flushCurrentThread() {
    flushThread(threadState());
}

void Profiler::reset() {
    flushCurrentThread();

    auto &aggregate = aggregateRegistry();
    std::lock_guard lock(aggregate.mutex);
    aggregate.zones = {};
    aggregate.counters = {};
    aggregate.counterSet = {};
    aggregate.droppedEvents = 0;
}

ProfilerSnapshot Profiler::snapshot() {
    ProfilerSnapshot snapshot;
    snapshot.frameId = frameId();

    auto &aggregate = aggregateRegistry();
    auto &names = nameRegistry();
    std::scoped_lock lock(aggregate.mutex, names.mutex);

    snapshot.droppedEvents = aggregate.droppedEvents;
    snapshot.zones.reserve(names.count);

    for (uint32_t i = 0; i < names.count; ++i) {
        const AggregatedZone &zone = aggregate.zones[i];
        const char *name = names.names[i] != nullptr ? names.names[i] : "<unknown>";
        if (zone.callCount != 0) {
            const double totalMs = ticksToMilliseconds(zone.totalTicks);
            snapshot.zones.push_back(ZoneStats{
                .name = name,
                .callCount = zone.callCount,
                .totalTicks = zone.totalTicks,
                .minTicks = zone.minTicks,
                .maxTicks = zone.maxTicks,
                .totalMilliseconds = totalMs,
                .averageMilliseconds = totalMs / static_cast<double>(zone.callCount),
            });
        }
        if (aggregate.counterSet[i]) {
            snapshot.counters.push_back(CounterStats{
                .name = name,
                .value = aggregate.counters[i],
            });
        }
    }

    std::sort(snapshot.zones.begin(), snapshot.zones.end(), [](const ZoneStats &lhs, const ZoneStats &rhs) {
        return lhs.totalTicks > rhs.totalTicks;
    });
    return snapshot;
}

void Profiler::setCounter(const char *name, uint64_t value) {
    if (!enabled.load(std::memory_order_relaxed)) { return; }

    const uint32_t nameId = registerName(name);
    auto &aggregate = aggregateRegistry();
    std::lock_guard lock(aggregate.mutex);
    aggregate.counters[nameId] = value;
    aggregate.counterSet[nameId] = true;
}

uint64_t Profiler::frameId() noexcept {
    return gFrameId.load(std::memory_order_relaxed);
}

void Profiler::setAggregationFrameInterval(uint32_t frames) noexcept {
    gAggregationFrameInterval.store(std::max(1u, frames), std::memory_order_relaxed);
}

double Profiler::ticksToMilliseconds(uint64_t ticks) noexcept {
    return (static_cast<double>(ticks) * 1000.0) / static_cast<double>(ticksFrequency());
}

ScopedZone::ScopedZone(const char *name) {
    ThreadState &state = threadState();
    if (!state.active) { return; }

    nameId_ = registerName(name);
    beginTicks_ = readTicks();
    active_ = true;
}

ScopedZone::~ScopedZone() {
    if (!active_) { return; }

    recordEvent(threadState(), nameId_, beginTicks_, readTicks());
}

FrameScope::FrameScope() noexcept {
    Profiler::beginFrame();
}

FrameScope::~FrameScope() noexcept {
    Profiler::endFrame();
}

} // namespace mcvr::profiler

#endif
