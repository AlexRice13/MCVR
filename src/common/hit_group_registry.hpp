#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcvr {

class HitGroupRegistry {
  public:
    static uint32_t registerName(const std::string &name) {
        auto &state = registryState();
        std::lock_guard lock(state.mutex);
        auto iter = state.nameToId.find(name);
        if (iter != state.nameToId.end()) { return iter->second; }

        const uint32_t id = static_cast<uint32_t>(state.names.size());
        state.names.push_back(name);
        state.nameToId.emplace(state.names.back(), id);
        return id;
    }

    static uint32_t shadowId() {
        static const uint32_t id = registerName("shadow");
        return id;
    }

    static uint32_t defaultId() {
        static const uint32_t id = registerName("default");
        return id;
    }

    static uint32_t entityDefaultId() {
        static const uint32_t id = registerName("Entity");
        return id;
    }

    static std::vector<std::string> namesSnapshot() {
        auto &state = registryState();
        std::lock_guard lock(state.mutex);
        return state.names;
    }

  private:
    struct State {
        std::mutex mutex;
        std::unordered_map<std::string, uint32_t> nameToId;
        std::vector<std::string> names;
    };

    static State &registryState() {
        static State state;
        return state;
    }
};

} // namespace mcvr
