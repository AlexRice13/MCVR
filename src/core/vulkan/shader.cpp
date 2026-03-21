#include "core/vulkan/shader.hpp"

#include "core/vulkan/device.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

std::ostream &shaderCout() {
    return std::cout << "[Shader] ";
}

std::ostream &shaderCerr() {
    return std::cerr << "[Shader] ";
}

shaderc_shader_kind vk::shaderKindFromStage(VkShaderStageFlagBits stage) {
    switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: return shaderc_glsl_vertex_shader;
        case VK_SHADER_STAGE_FRAGMENT_BIT: return shaderc_glsl_fragment_shader;
        case VK_SHADER_STAGE_COMPUTE_BIT: return shaderc_glsl_compute_shader;
        case VK_SHADER_STAGE_RAYGEN_BIT_KHR: return shaderc_glsl_raygen_shader;
        case VK_SHADER_STAGE_MISS_BIT_KHR: return shaderc_glsl_miss_shader;
        case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR: return shaderc_glsl_closesthit_shader;
        case VK_SHADER_STAGE_ANY_HIT_BIT_KHR: return shaderc_glsl_anyhit_shader;
        case VK_SHADER_STAGE_INTERSECTION_BIT_KHR: return shaderc_glsl_intersection_shader;
        case VK_SHADER_STAGE_CALLABLE_BIT_KHR: return shaderc_glsl_callable_shader;
        default: shaderCerr() << "unsupported shader stage " << stage << std::endl; exit(EXIT_FAILURE);
    }
}

struct vk::ShaderIncluder::IncludeData {
    shaderc_include_result result{};
    std::string sourceName;
    std::string content;
};

vk::ShaderIncluder::ShaderIncluder(std::vector<std::filesystem::path> includeDirs)
    : includeDirs_(std::move(includeDirs)) {}

shaderc_include_result *vk::ShaderIncluder::GetInclude(const char *requested_source,
                                                       shaderc_include_type type,
                                                       const char *requesting_source,
                                                       size_t include_depth) {
    (void)include_depth;
    std::filesystem::path requestedPath(requested_source);
    std::filesystem::path resolvedPath;

    if (type == shaderc_include_type_relative && requesting_source != nullptr && requesting_source[0] != '\0') {
        std::filesystem::path requestingPath(requesting_source);
        std::filesystem::path candidate = requestingPath.parent_path() / requestedPath;
        if (std::filesystem::exists(candidate) && std::filesystem::is_regular_file(candidate)) {
            resolvedPath = std::filesystem::weakly_canonical(candidate);
        }
    }

    if (resolvedPath.empty()) {
        for (const std::filesystem::path &includeDir : includeDirs_) {
            std::filesystem::path candidate = includeDir / requestedPath;
            if (std::filesystem::exists(candidate) && std::filesystem::is_regular_file(candidate)) {
                resolvedPath = std::filesystem::weakly_canonical(candidate);
                break;
            }
        }
    }

    IncludeData *includeData = new IncludeData{};
    if (resolvedPath.empty()) {
        includeData->content = "Failed to resolve include: " + std::string(requested_source);
        includeData->sourceName = requested_source;
        includeData->result.source_name = includeData->sourceName.c_str();
        includeData->result.source_name_length = includeData->sourceName.size();
        includeData->result.content = includeData->content.c_str();
        includeData->result.content_length = includeData->content.size();
        includeData->result.user_data = includeData;
        return &includeData->result;
    }

    std::ifstream includeFile(resolvedPath, std::ios::binary);
    if (!includeFile.is_open()) {
        includeData->content = "Failed to open include: " + resolvedPath.string();
        includeData->sourceName = resolvedPath.string();
        includeData->result.source_name = includeData->sourceName.c_str();
        includeData->result.source_name_length = includeData->sourceName.size();
        includeData->result.content = includeData->content.c_str();
        includeData->result.content_length = includeData->content.size();
        includeData->result.user_data = includeData;
        return &includeData->result;
    }

    includeData->sourceName = resolvedPath.string();
    includeData->content.assign(std::istreambuf_iterator<char>(includeFile), std::istreambuf_iterator<char>());
    includeData->result.source_name = includeData->sourceName.c_str();
    includeData->result.source_name_length = includeData->sourceName.size();
    includeData->result.content = includeData->content.c_str();
    includeData->result.content_length = includeData->content.size();
    includeData->result.user_data = includeData;
    return &includeData->result;
}

void vk::ShaderIncluder::ReleaseInclude(shaderc_include_result *data) {
    if (data == nullptr || data->user_data == nullptr) { return; }
    delete static_cast<IncludeData *>(data->user_data);
}

vk::Shader::Shader(std::shared_ptr<Device> device, std::string filePath) : device_(device), filePath_(filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        shaderCerr() << "Cannot open file: " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<char> fileBytes(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(fileBytes.data(), fileBytes.size());
    file.close();

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileBytes.size();
    createInfo.pCode = (uint32_t *)fileBytes.data();

    if (vkCreateShaderModule(device->vkDevice(), &createInfo, nullptr, &shader_) != VK_SUCCESS) {
        shaderCerr() << "failed to create shader module for " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG
    shaderCout() << "created shader module for " << filePath << std::endl;
#endif
}

vk::Shader::Shader(std::shared_ptr<Device> device,
                   std::string sourcePath,
                   VkShaderStageFlagBits stage,
                   std::unordered_map<std::string, std::string> definitions,
                   std::vector<std::string> includeDirectories)
    : device_(device), filePath_(sourcePath) {
    std::ifstream sourceFile(sourcePath, std::ios::binary);
    if (!sourceFile.is_open()) {
        shaderCerr() << "Cannot open source file: " << sourcePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string sourceText{std::istreambuf_iterator<char>(sourceFile), std::istreambuf_iterator<char>()};

    std::vector<std::filesystem::path> includeDirs;
    std::filesystem::path sourcePathFs(sourcePath);
    includeDirs.emplace_back(sourcePathFs.parent_path());
    for (const std::string &includeDirectory : includeDirectories) {
        includeDirs.emplace_back(std::filesystem::path(includeDirectory));
    }

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    for (const auto &[name, value] : definitions) { options.AddMacroDefinition(name, value); }
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_4);
    options.SetSourceLanguage(shaderc_source_language_glsl);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetIncluder(std::make_unique<ShaderIncluder>(includeDirs));

    shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(sourceText, shaderKindFromStage(stage), sourcePath.c_str(), options);
    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        shaderCerr() << "failed to compile shader source " << sourcePath << "\n"
                     << module.GetErrorMessage() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<uint32_t> spirv(module.cbegin(), module.cend());
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    if (vkCreateShaderModule(device->vkDevice(), &createInfo, nullptr, &shader_) != VK_SUCCESS) {
        shaderCerr() << "failed to create shader module for " << sourcePath << std::endl;
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG
    shaderCout() << "created runtime shader module for " << sourcePath << std::endl;
#endif
}

vk::Shader::~Shader() {
    vkDestroyShaderModule(device_->vkDevice(), shader_, nullptr);
}

VkShaderModule &vk::Shader::vkShaderModule() {
    return shader_;
}
