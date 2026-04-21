#include "core/render/scenario_color_grading.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <sstream>
#include <utility>

namespace {

std::string trim(std::string text) {
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    auto start = std::find_if(text.begin(), text.end(), notSpace);
    if (start == text.end()) return "";
    auto end = std::find_if(text.rbegin(), text.rend(), notSpace).base();
    return std::string(start, end);
}

std::string toLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

bool parseBoolValue(const std::string &value, bool fallback) {
    std::string lowered = toLower(trim(value));
    if (lowered == "true" || lowered == "1" || lowered == "yes" || lowered == "on") return true;
    if (lowered == "false" || lowered == "0" || lowered == "no" || lowered == "off") return false;
    return fallback;
}

int parseIntValue(const std::string &value, int fallback) {
    try {
        return std::stoi(trim(value));
    } catch (...) {
        return fallback;
    }
}

std::vector<std::string> splitCsv(const std::string &text) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

std::string joinCsv(const std::vector<std::string> &values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << values[i];
    }
    return out.str();
}

bool listMatches(const std::vector<std::string> &values, const std::string &candidate) {
    if (values.empty()) return true;
    return std::find(values.begin(), values.end(), candidate) != values.end();
}

bool triStateMatches(const std::string &rule, bool value) {
    std::string lowered = toLower(rule);
    if (lowered.empty() || lowered == "any") return true;
    if (lowered == "true") return value;
    if (lowered == "false") return !value;
    return true;
}

bool stringMatches(const std::string &rule, const std::string &value) {
    std::string lowered = toLower(rule);
    return lowered.empty() || lowered == "any" || lowered == toLower(value);
}

bool isValidTickRangeValue(int value) {
    return value >= 0 && value <= 24000;
}

uint32_t normalizeDayTick(uint32_t tick) {
    return tick % 24000u;
}

uint32_t normalizeRangeBound(int tick) {
    if (tick >= 24000) return 0u;
    return static_cast<uint32_t>(tick);
}

bool timeRangeMatches(int startTick, int endTick, uint32_t timeOfDay) {
    if (startTick < 0 || endTick < 0) return true;
    if (!isValidTickRangeValue(startTick) || !isValidTickRangeValue(endTick)) return false;
    if (startTick == 0 && endTick == 24000) return true;

    uint32_t current = normalizeDayTick(timeOfDay);
    uint32_t start = normalizeRangeBound(startTick);
    uint32_t end = normalizeRangeBound(endTick);
    if (startTick == endTick) return current == start;
    if (startTick < endTick) {
        if (endTick == 24000) return current >= start;
        return current >= start && current <= end;
    }
    return current >= start || current <= end;
}

std::string classifyWeather(const ScenarioSceneContext &context) {
    if (context.thundering) return "thunder";
    if (context.raining) return "rain";
    return "clear";
}

std::pair<int, int> legacyTimePeriodToRange(const std::string &timePeriod) {
    std::string lowered = toLower(trim(timePeriod));
    if (lowered.empty() || lowered == "any") return {-1, -1};
    if (lowered == "sunrise") return {23000, 1000};
    if (lowered == "sunset") return {12000, 13000};
    if (lowered == "night") return {13000, 23000};
    if (lowered == "day") return {1000, 12000};
    return {-1, -1};
}

using IniSections = std::map<std::string, std::map<std::string, std::string>>;

IniSections parseIniFile(const std::filesystem::path &path) {
    IniSections sections;
    std::ifstream in(path);
    if (!in.is_open()) return sections;

    std::string currentSection = "global";
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line.front() == '[' && line.back() == ']') {
            currentSection = trim(line.substr(1, line.size() - 2));
            continue;
        }

        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;

        std::string key = trim(line.substr(0, eqPos));
        std::string value = trim(line.substr(eqPos + 1));
        sections[currentSection][key] = value;
    }

    return sections;
}

std::string defaultScenarioConfigText() {
    return R"(; Radiance scenario color grading config
; Sections:
;   [scenario.<name>]         scenario matcher and metadata
;   [scenario.<name>.hdr_on]  overrides when HDR output is active
;   [scenario.<name>.hdr_off] overrides when HDR output is inactive

[global]
version=1

[scenario.example_nether]
enabled=false
priority=5
world=minecraft:the_nether
weather=any
time_start=13000
time_end=23000
submersion=any
indoor=any
cave=any

[scenario.example_nether.hdr_on]
render_pipeline.module.tone_mapping.attribute.saturation=1.10
render_pipeline.module.tone_mapping.attribute.contrast=1.08

[scenario.example_nether.hdr_off]
render_pipeline.module.tone_mapping.attribute.saturation=1.00
render_pipeline.module.tone_mapping.attribute.contrast=1.04
)";
}

} // namespace

void ScenarioColorGradingManager::setConfigPath(const std::filesystem::path &configPath) {
    std::scoped_lock lock(mutex_);
    configPath_ = configPath;
    std::filesystem::create_directories(configPath_.parent_path());
    ensureLoadedLocked();
}

void ScenarioColorGradingManager::updateSceneContext(const std::string &dimensionKey,
                                                     const std::string &biomeKey,
                                                     uint32_t timeOfDay,
                                                     bool raining,
                                                     bool thundering,
                                                     const std::string &submersion,
                                                     bool indoors,
                                                     bool cave) {
    std::scoped_lock lock(mutex_);
    sceneContext_.dimensionKey = dimensionKey;
    sceneContext_.biomeKey = biomeKey;
    sceneContext_.timeOfDay = timeOfDay;
    sceneContext_.raining = raining;
    sceneContext_.thundering = thundering;
    sceneContext_.submersion = submersion.empty() ? "air" : submersion;
    sceneContext_.indoors = indoors;
    sceneContext_.cave = cave;
}

ToneMappingSettings ScenarioColorGradingManager::resolveSettings(const ToneMappingSettings &baseSettings, bool hdrActive) {
    std::scoped_lock lock(mutex_);
    ensureLoadedLocked();

    ToneMappingSettings resolved = baseSettings;
    if (hasPreviewSettings_) return previewSettings_;
    const ScenarioDefinition *scenario = findBestMatchLocked();
    if (scenario == nullptr) return resolved;

    const auto &values = hdrActive ? scenario->hdrOnValues : scenario->hdrOffValues;
    for (const auto &[key, value] : values) { applyToneMappingAttributeKV(resolved, key, value); }
    return resolved;
}

void ScenarioColorGradingManager::setPreviewSettings(const ToneMappingSettings &previewSettings) {
    std::scoped_lock lock(mutex_);
    previewSettings_ = previewSettings;
    hasPreviewSettings_ = true;
}

void ScenarioColorGradingManager::clearPreviewSettings() {
    std::scoped_lock lock(mutex_);
    hasPreviewSettings_ = false;
}

bool ScenarioColorGradingManager::saveScenario(const std::string &scenarioName,
                                               int priority,
                                               const ScenarioSaveMetadataSelection &selection,
                                               bool hdrActive,
                                               const std::vector<std::string> &attributePairs) {
    std::string trimmedName = trim(scenarioName);
    if (trimmedName.empty()) return false;

    std::scoped_lock lock(mutex_);
    ensureLoadedLocked();

    auto iter = std::find_if(scenarios_.begin(), scenarios_.end(), [&](const ScenarioDefinition &scenario) {
        return scenario.name == trimmedName;
    });
    if (iter == scenarios_.end()) {
        scenarios_.push_back(ScenarioDefinition{.name = trimmedName});
        iter = std::prev(scenarios_.end());
    }

    ScenarioDefinition &scenario = *iter;
    scenario.enabled = true;
    scenario.priority = std::clamp(priority, 0, 10);
    scenario.worlds = selection.world && !sceneContext_.dimensionKey.empty()
        ? std::vector<std::string>{sceneContext_.dimensionKey}
        : std::vector<std::string>{};
    scenario.biomes = selection.biome && !sceneContext_.biomeKey.empty()
        ? std::vector<std::string>{sceneContext_.biomeKey}
        : std::vector<std::string>{};
    scenario.weather = selection.weather ? classifyWeather(sceneContext_) : "any";
    if (selection.time) {
        if (!isValidTickRangeValue(selection.timeStart) || !isValidTickRangeValue(selection.timeEnd)) {
            return false;
        }
        scenario.timeStart = selection.timeStart;
        scenario.timeEnd = selection.timeEnd;
    } else {
        scenario.timeStart = -1;
        scenario.timeEnd = -1;
    }
    scenario.submersion = selection.submersion
        ? (sceneContext_.submersion.empty() ? "air" : sceneContext_.submersion)
        : "any";
    scenario.indoor = selection.indoor ? (sceneContext_.indoors ? "true" : "false") : "any";
    scenario.cave = selection.cave ? (sceneContext_.cave ? "true" : "false") : "any";

    auto &branch = hdrActive ? scenario.hdrOnValues : scenario.hdrOffValues;
    branch.clear();
    for (size_t i = 0; i + 1 < attributePairs.size(); i += 2) {
        std::string key = trim(attributePairs[i]);
        if (key.empty()) continue;
        branch[key] = attributePairs[i + 1];
    }

    std::sort(scenarios_.begin(), scenarios_.end(), [](const ScenarioDefinition &left, const ScenarioDefinition &right) {
        if (left.priority != right.priority) return left.priority < right.priority;
        return left.name < right.name;
    });

    writeConfigLocked();
    return true;
}

void ScenarioColorGradingManager::ensureLoadedLocked() {
    if (configPath_.empty()) return;

    if (!std::filesystem::exists(configPath_)) {
        writeDefaultConfigLocked();
        hasKnownWriteTime_ = false;
    }

    std::optional<std::filesystem::file_time_type> currentWriteTime;
    try {
        currentWriteTime = std::filesystem::last_write_time(configPath_);
    } catch (...) {
        currentWriteTime.reset();
    }

    if (!hasLoadedAtLeastOnce_ || !hasKnownWriteTime_ || !currentWriteTime.has_value() || lastWriteTime_ != *currentWriteTime) {
        loadConfigLocked();
        hasLoadedAtLeastOnce_ = true;
        if (currentWriteTime.has_value()) {
            lastWriteTime_ = *currentWriteTime;
            hasKnownWriteTime_ = true;
        } else {
            hasKnownWriteTime_ = false;
        }
    }
}

void ScenarioColorGradingManager::loadConfigLocked() {
    scenarios_.clear();

    IniSections sections = parseIniFile(configPath_);
    for (const auto &[sectionName, values] : sections) {
        constexpr std::string_view scenarioPrefix = "scenario.";
        if (!sectionName.starts_with(scenarioPrefix)) continue;

        std::string scenarioName = sectionName.substr(scenarioPrefix.size());
        bool hdrOn = false;
        bool hdrOff = false;

        if (scenarioName.ends_with(".hdr_on")) {
            scenarioName.erase(scenarioName.size() - std::string(".hdr_on").size());
            hdrOn = true;
        } else if (scenarioName.ends_with(".hdr_off")) {
            scenarioName.erase(scenarioName.size() - std::string(".hdr_off").size());
            hdrOff = true;
        }

        auto iter = std::find_if(scenarios_.begin(), scenarios_.end(), [&](const ScenarioDefinition &scenario) {
            return scenario.name == scenarioName;
        });
        if (iter == scenarios_.end()) {
            scenarios_.push_back(ScenarioDefinition{.name = scenarioName});
            iter = std::prev(scenarios_.end());
        }

        ScenarioDefinition &scenario = *iter;
        if (hdrOn) {
            scenario.hdrOnValues = values;
            continue;
        }
        if (hdrOff) {
            scenario.hdrOffValues = values;
            continue;
        }

        scenario.enabled = parseBoolValue(values.contains("enabled") ? values.at("enabled") : "true", true);
        scenario.priority = parseIntValue(values.contains("priority") ? values.at("priority") : "0", 0);
        scenario.worlds.clear();
        if (values.contains("world")) scenario.worlds.push_back(values.at("world"));
        if (values.contains("worlds")) {
            auto worlds = splitCsv(values.at("worlds"));
            scenario.worlds.insert(scenario.worlds.end(), worlds.begin(), worlds.end());
        }

        scenario.biomes.clear();
        if (values.contains("biome")) scenario.biomes.push_back(values.at("biome"));
        if (values.contains("biomes")) {
            auto biomes = splitCsv(values.at("biomes"));
            scenario.biomes.insert(scenario.biomes.end(), biomes.begin(), biomes.end());
        }

        scenario.priority = std::clamp(scenario.priority, 0, 10);
        scenario.weather = values.contains("weather") ? values.at("weather") : "any";
        scenario.timeStart = -1;
        scenario.timeEnd = -1;
        if (values.contains("time_start") && values.contains("time_end")) {
            int timeStart = parseIntValue(values.at("time_start"), -1);
            int timeEnd = parseIntValue(values.at("time_end"), -1);
            if (isValidTickRangeValue(timeStart) && isValidTickRangeValue(timeEnd)) {
                scenario.timeStart = timeStart;
                scenario.timeEnd = timeEnd;
            }
        } else if (values.contains("time_period")) {
            auto [timeStart, timeEnd] = legacyTimePeriodToRange(values.at("time_period"));
            scenario.timeStart = timeStart;
            scenario.timeEnd = timeEnd;
        }
        scenario.submersion = values.contains("submersion") ? values.at("submersion") : "any";
        scenario.indoor = values.contains("indoor") ? values.at("indoor") : "any";
        scenario.cave = values.contains("cave") ? values.at("cave") : "any";
    }

    std::sort(scenarios_.begin(), scenarios_.end(), [](const ScenarioDefinition &left, const ScenarioDefinition &right) {
        if (left.priority != right.priority) return left.priority < right.priority;
        return left.name < right.name;
    });
}

void ScenarioColorGradingManager::writeConfigLocked() const {
    std::filesystem::create_directories(configPath_.parent_path());
    std::ofstream out(configPath_, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return;

    out << "; Radiance scenario color grading config\n";
    out << "[global]\n";
    out << "version=1\n\n";

    for (const auto &scenario : scenarios_) {
        out << "[scenario." << scenario.name << "]\n";
        out << "enabled=" << (scenario.enabled ? "true" : "false") << '\n';
        out << "priority=" << scenario.priority << '\n';
        if (scenario.worlds.size() == 1) out << "world=" << scenario.worlds.front() << '\n';
        else if (!scenario.worlds.empty()) out << "worlds=" << joinCsv(scenario.worlds) << '\n';
        if (scenario.biomes.size() == 1) out << "biome=" << scenario.biomes.front() << '\n';
        else if (!scenario.biomes.empty()) out << "biomes=" << joinCsv(scenario.biomes) << '\n';
        if (!scenario.weather.empty()) out << "weather=" << scenario.weather << '\n';
        if (scenario.timeStart >= 0 && scenario.timeEnd >= 0) {
            out << "time_start=" << scenario.timeStart << '\n';
            out << "time_end=" << scenario.timeEnd << '\n';
        }
        if (!scenario.submersion.empty()) out << "submersion=" << scenario.submersion << '\n';
        if (!scenario.indoor.empty()) out << "indoor=" << scenario.indoor << '\n';
        if (!scenario.cave.empty()) out << "cave=" << scenario.cave << '\n';
        out << '\n';

        out << "[scenario." << scenario.name << ".hdr_on]\n";
        for (const auto &[key, value] : scenario.hdrOnValues) { out << key << '=' << value << '\n'; }
        out << '\n';

        out << "[scenario." << scenario.name << ".hdr_off]\n";
        for (const auto &[key, value] : scenario.hdrOffValues) { out << key << '=' << value << '\n'; }
        out << "\n\n";
    }
}

void ScenarioColorGradingManager::writeDefaultConfigLocked() const {
    std::filesystem::create_directories(configPath_.parent_path());
    std::ofstream out(configPath_, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return;
    out << defaultScenarioConfigText();
}

const ScenarioDefinition *ScenarioColorGradingManager::findBestMatchLocked() const {
    std::vector<const ScenarioDefinition *> matches;
    for (const auto &scenario : scenarios_) {
        if (matchesScenario(scenario, sceneContext_)) matches.push_back(&scenario);
    }
    if (matches.empty()) return nullptr;

    int bestPriority = 10;
    for (const ScenarioDefinition *scenario : matches) {
        bestPriority = std::min(bestPriority, scenario->priority);
    }

    const ScenarioDefinition *winner = nullptr;
    for (const ScenarioDefinition *scenario : matches) {
        if (scenario->priority != bestPriority) continue;
        if (winner != nullptr) return nullptr;
        winner = scenario;
    }
    return winner;
}

bool ScenarioColorGradingManager::matchesScenario(const ScenarioDefinition &scenario,
                                                  const ScenarioSceneContext &context) const {
    if (!scenario.enabled) return false;
    if (!listMatches(scenario.worlds, context.dimensionKey)) return false;
    if (!listMatches(scenario.biomes, context.biomeKey)) return false;
    if (!stringMatches(scenario.weather, classifyWeather(context))) return false;
    if (!timeRangeMatches(scenario.timeStart, scenario.timeEnd, context.timeOfDay)) return false;
    if (!stringMatches(scenario.submersion, context.submersion)) return false;
    if (!triStateMatches(scenario.indoor, context.indoors)) return false;
    if (!triStateMatches(scenario.cave, context.cave)) return false;
    return true;
}
