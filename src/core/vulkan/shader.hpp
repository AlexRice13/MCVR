#pragma once

#include "core/all_extern.hpp"

#include <filesystem>
#include <shaderc/shaderc.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace vk {
class Device;
shaderc_shader_kind shaderKindFromStage(VkShaderStageFlagBits stage);

class ShaderIncluder : public shaderc::CompileOptions::IncluderInterface {
  public:
    explicit ShaderIncluder(std::vector<std::filesystem::path> includeDirs);

    shaderc_include_result *GetInclude(const char *requested_source,
                                       shaderc_include_type type,
                                       const char *requesting_source,
                                       size_t include_depth) override;
    void ReleaseInclude(shaderc_include_result *data) override;

  private:
    struct IncludeData;

    std::vector<std::filesystem::path> includeDirs_;
};

class Shader : public SharedObject<Shader> {
  public:
    Shader(std::shared_ptr<Device> device, std::string filePath);
    Shader(std::shared_ptr<Device> device,
           std::string sourcePath,
           VkShaderStageFlagBits stage,
           std::unordered_map<std::string, std::string> definitions = {},
           std::vector<std::string> includeDirectories = {});
    ~Shader();

    VkShaderModule &vkShaderModule();

  private:
    std::shared_ptr<Device> device_;

    std::string filePath_;
    VkShaderModule shader_;
};
}; // namespace vk
