#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/render/chunks.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <map>
#include <mutex>
#include <queue>
#include <unordered_map>

class Framework;
class FrameworkContext;
class RayTracingModule;
struct RayTracingModuleContext;
struct Entity;

struct WorldPrepareContext;

class WorldPrepare : public SharedObject<WorldPrepare> {
    friend RayTracingModule;
    friend RayTracingModuleContext;
    friend WorldPrepareContext;

  public:
    WorldPrepare();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<RayTracingModule> rayTracingModule);

    void build();

  private:
    using EntityRenderDataBatch = std::map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>>;

    std::weak_ptr<Framework> framework_;
    std::weak_ptr<RayTracingModule> rayTracingModule_;

    std::queue<EntityRenderDataBatch> previousEntityRenderDataBatches_;
    EntityRenderDataBatch emptyEntityRenderDataBatch_;
    std::recursive_mutex entityRenderDataBatchesMtx_;

    std::vector<std::shared_ptr<WorldPrepareContext>> contexts_;
};

struct WorldPrepareContext : public SharedObject<WorldPrepareContext> {
    std::weak_ptr<FrameworkContext> frameworkContext;
    std::weak_ptr<RayTracingModuleContext> rayTracingModuleContext;
    std::weak_ptr<WorldPrepare> worldPrepare;

    std::shared_ptr<vk::TLAS> tlas;
    std::shared_ptr<vk::TLASBuilder> tlasBuilder;

    std::shared_ptr<vk::DeviceLocalBuffer> blasOffsetsBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> positionBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> materialBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastIndexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastPositionBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastObjToWorldMat;
    std::shared_ptr<vk::DeviceLocalBuffer> tlasInstanceBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> tlasScratchBuffer;
    uint32_t previousTlasInstanceCount = 0;
    std::vector<uint32_t> hitGroupNameIds;
    uint64_t hitGroupNameIdsHash = 0;
    size_t previousHitGroupNameCount = 0;
    size_t previousGeometryAddressCount = 0;
    size_t previousTlasInstanceReserve = 0;

    struct CachedChunkRow {
        std::shared_ptr<Chunk1> chunk;
        int64_t latestVersion = -1;
        int64_t blasVersion = -1;
        uint32_t geometryCount = 0;
        std::shared_ptr<std::vector<VkDeviceAddress>> indexBufferAddresses;
        std::shared_ptr<std::vector<VkDeviceAddress>> positionBufferAddresses;
        std::shared_ptr<std::vector<VkDeviceAddress>> materialBufferAddresses;
        std::shared_ptr<std::vector<uint32_t>> geometryGroupIds;
        std::shared_ptr<vk::BLAS> blas;
        std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> positionBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> materialBuffer;
        std::shared_ptr<std::vector<LightInfo>> lightInfos;
        std::shared_ptr<vk::DeviceLocalBuffer> lightBuffer;
        uint32_t lightCount = 0;
    };
    std::vector<CachedChunkRow> cachedChunkRows;

    struct CachedSbtHitGroupIndices {
        size_t hitGroupCount = 0;
        uint64_t hitGroupHash = 0;
        uint64_t passMapHash = 0;
        uint32_t fallbackHitGroupIndex = 0;
        uint32_t shadowHitGroupIndex = 0;
        std::vector<uint32_t> indices;
    };
    std::unordered_map<const void *, CachedSbtHitGroupIndices> cachedSbtHitGroupIndices;

    WorldPrepareContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<WorldPrepare> worldprepare);

    void uploadBuffer(std::vector<uint32_t> &blasOffsets,
                      std::vector<uint64_t> &indexBufferAddrs,
                      std::vector<uint64_t> &positionBufferAddrs,
                      std::vector<uint64_t> &materialBufferAddrs,
                      std::vector<uint64_t> &lastIndexBufferAddrs,
                      std::vector<uint64_t> &lastPositionBufferAddrs,
                      std::vector<glm::mat4> &lastObjToWorldMats);
    void setupHitGroupSbt(const void *passKey,
                          const std::vector<uint32_t> &hitGroupIdToIndex,
                          uint64_t hitGroupIdToIndexHash,
                          uint32_t fallbackHitGroupIndex,
                          uint32_t shadowHitGroupIndex,
                          std::shared_ptr<vk::CommandBuffer> commandBuffer,
                          std::shared_ptr<vk::SBT> updateSbt,
                          std::shared_ptr<vk::SBT> querySbt);
    CachedChunkRow &refreshCachedChunkRow(size_t index, const std::shared_ptr<Chunk1> &chunk);
    void render();
};
