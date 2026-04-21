#pragma once

#include "common/singleton.hpp"
#include "core/render/modules/world/tone_mapping/tone_mapping_module.hpp"

#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct ScenarioSceneContext {
    std::string dimensionKey;
    std::string biomeKey;
    uint32_t timeOfDay = 0;
    bool raining = false;
    bool thundering = false;
    std::string submersion = "air";
    bool indoors = false;
    bool cave = false;
};

struct ScenarioDefinition {
    std::string name;
    bool enabled = true;
    int priority = 0;
    std::vector<std::string> worlds;
    std::vector<std::string> biomes;
    std::string weather = "any";
    int timeStart = -1;
    int timeEnd = -1;
    std::string submersion = "any";
    std::string indoor = "any";
    std::string cave = "any";
    std::map<std::string, std::string> hdrOnValues;
    std::map<std::string, std::string> hdrOffValues;
};

struct ScenarioSaveMetadataSelection {
    bool world = false;
    bool time = false;
    bool weather = false;
    bool biome = false;
    bool submersion = false;
    bool indoor = false;
    bool cave = false;
    int timeStart = -1;
    int timeEnd = -1;
};

class ScenarioColorGradingManager : public Singleton<ScenarioColorGradingManager> {
    friend class Singleton<ScenarioColorGradingManager>;

  public:
    void setConfigPath(const std::filesystem::path &configPath);
    void updateSceneContext(const std::string &dimensionKey,
                            const std::string &biomeKey,
                            uint32_t timeOfDay,
                            bool raining,
                            bool thundering,
                            const std::string &submersion,
                            bool indoors,
                            bool cave);

    ToneMappingSettings resolveSettings(const ToneMappingSettings &baseSettings, bool hdrActive);

    void setPreviewSettings(const ToneMappingSettings &previewSettings);
    void clearPreviewSettings();

    bool saveScenario(const std::string &scenarioName,
                      int priority,
                      const ScenarioSaveMetadataSelection &selection,
                      bool hdrActive,
                      const std::vector<std::string> &attributePairs);

  private:
    ScenarioColorGradingManager() = default;

    void ensureLoadedLocked();
    void loadConfigLocked();
    void writeConfigLocked() const;
    void writeDefaultConfigLocked() const;

    const ScenarioDefinition *findBestMatchLocked() const;
    bool matchesScenario(const ScenarioDefinition &scenario, const ScenarioSceneContext &context) const;

  private:
    mutable std::mutex mutex_;
    std::filesystem::path configPath_;
    bool hasLoadedAtLeastOnce_ = false;
    std::filesystem::file_time_type lastWriteTime_{};
    bool hasKnownWriteTime_ = false;
    ScenarioSceneContext sceneContext_{};
    bool hasPreviewSettings_ = false;
    ToneMappingSettings previewSettings_{};
    std::vector<ScenarioDefinition> scenarios_;
};
