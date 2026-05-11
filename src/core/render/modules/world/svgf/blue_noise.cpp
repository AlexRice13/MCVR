#include "blue_noise.hpp"

#include <iostream>

#if __has_include("../../../extern/FidelityFX-SDK/sdk/src/components/sssr/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.cpp")
#    include "../../../extern/FidelityFX-SDK/sdk/src/components/sssr/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.cpp"
#    define RADIANCE_HAS_FFX_BLUE_NOISE_DATA 1
#else
#    define RADIANCE_HAS_FFX_BLUE_NOISE_DATA 0
#endif

namespace {

uint32_t fallbackBlueNoiseValue(size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) + salt;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

} // namespace

std::ostream &blueNoiseCout() {
    return std::cout << "[BlueNoise] ";
}

BlueNoise::BlueNoise(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::VMA> vma) {
    blueNoiseCout() << "Initializing blue noise buffers..." << std::endl;

    // Create Sobol buffer (256*256 uint32 = 256KB)
    m_sobolBuffer = vk::DeviceLocalBuffer::create(
        vma, device, SOBOL_SIZE * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    std::vector<uint32_t> sobolData(SOBOL_SIZE);
    for (size_t i = 0; i < SOBOL_SIZE; i++) {
#if RADIANCE_HAS_FFX_BLUE_NOISE_DATA
        sobolData[i] = static_cast<uint32_t>(sobol_256spp_256d[i]);
#else
        sobolData[i] = fallbackBlueNoiseValue(i, 0x51f15eedu);
#endif
    }
    m_sobolBuffer->uploadToStagingBuffer(sobolData.data());

    // Create scrambling tile buffer (128*128*8 uint32 = 512KB)
    m_scramblingBuffer = vk::DeviceLocalBuffer::create(
        vma, device, SCRAMBLING_SIZE * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    std::vector<uint32_t> scramblingData(SCRAMBLING_SIZE);
    for (size_t i = 0; i < SCRAMBLING_SIZE; i++) {
#if RADIANCE_HAS_FFX_BLUE_NOISE_DATA
        scramblingData[i] = static_cast<uint32_t>(scramblingTile[i]);
#else
        scramblingData[i] = fallbackBlueNoiseValue(i, 0x9e3779b9u);
#endif
    }
    m_scramblingBuffer->uploadToStagingBuffer(scramblingData.data());

    blueNoiseCout() << "Blue noise buffers created: Sobol(" << SOBOL_SIZE * sizeof(uint32_t) / 1024
                    << "KB), Scrambling(" << SCRAMBLING_SIZE * sizeof(uint32_t) / 1024 << "KB)" << std::endl;
}

std::shared_ptr<vk::DeviceLocalBuffer> BlueNoise::sobolBuffer() {
    return m_sobolBuffer;
}

std::shared_ptr<vk::DeviceLocalBuffer> BlueNoise::scramblingBuffer() {
    return m_scramblingBuffer;
}

void BlueNoise::uploadToBuffer(std::shared_ptr<vk::CommandBuffer> cmdBuffer) {
    m_sobolBuffer->uploadToBuffer(cmdBuffer);
    m_scramblingBuffer->uploadToBuffer(cmdBuffer);
}
