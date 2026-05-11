#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"

#include "common/hit_group_registry.hpp"
#include "common/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

#include <filesystem>
#include <glm/gtc/type_ptr.hpp>

WorldPrepare::WorldPrepare() {}

void WorldPrepare::init(std::shared_ptr<Framework> framework, std::shared_ptr<RayTracingModule> rayTracingModule) {
    framework_ = framework;
    rayTracingModule_ = rayTracingModule;
}

void WorldPrepare::build() {
    auto framework = framework_.lock();
    auto rayTracingModule = rayTracingModule_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        contexts_[i] = WorldPrepareContext::create(framework->contexts()[i], shared_from_this());
    }
}

WorldPrepareContext::WorldPrepareContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                          std::shared_ptr<WorldPrepare> worldPrepare)
    : frameworkContext(frameworkContext), worldPrepare(worldPrepare) {}

WorldPrepareContext::CachedChunkRow &
WorldPrepareContext::refreshCachedChunkRow(size_t index, const std::shared_ptr<Chunk1> &chunk) {
    if (cachedChunkRows.size() <= index) { cachedChunkRows.resize(index + 1); }

    auto &row = cachedChunkRows[index];
    if (row.chunk == chunk && row.latestVersion == chunk->latestVersion && row.blasVersion == chunk->blasVersion &&
        row.geometryCount == chunk->geometryCount) {
        return row;
    }

    row.chunk = chunk;
    row.latestVersion = chunk->latestVersion;
    row.blasVersion = chunk->blasVersion;
    row.geometryCount = chunk->geometryCount;
    row.indexBufferAddresses = chunk->indexBufferAddresses;
    row.positionBufferAddresses = chunk->positionBufferAddresses;
    row.materialBufferAddresses = chunk->materialBufferAddresses;
    row.geometryGroupIds = chunk->geometryGroupIds;
    row.blas = chunk->blas;
    row.indexBuffer = chunk->indexBuffer;
    row.positionBuffer = chunk->positionBuffer;
    row.materialBuffer = chunk->materialBuffer;
    row.lightInfos = chunk->lightInfos;
    row.lightBuffer = chunk->lightBuffer;
    row.lightCount = chunk->lightCount;
    return row;
}

void WorldPrepareContext::uploadBuffer(std::vector<uint32_t> &blasOffsets,
                                       std::vector<uint64_t> &indexBufferAddrs,
                                       std::vector<uint64_t> &positionBufferAddrs,
                                       std::vector<uint64_t> &materialBufferAddrs,
                                       std::vector<uint64_t> &lastIndexBufferAddrs,
                                       std::vector<uint64_t> &lastPositionBufferAddrs,
                                       std::vector<glm::mat4> &lastObjToWorldMats) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto mainQueueIndex = physicalDevice->mainQueueIndex();
    auto cmdBuffer = context->worldCommandBuffer;

    blasOffsetsBuffer = vk::DeviceLocalBuffer::create(
        vma, device, blasOffsets.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    blasOffsetsBuffer->uploadToStagingBuffer(blasOffsets.data());

    indexBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, indexBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    indexBufferAddr->uploadToStagingBuffer(indexBufferAddrs.data());

    positionBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, positionBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    positionBufferAddr->uploadToStagingBuffer(positionBufferAddrs.data());

    materialBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, materialBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    materialBufferAddr->uploadToStagingBuffer(materialBufferAddrs.data());

    lastIndexBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, lastIndexBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    lastIndexBufferAddr->uploadToStagingBuffer(lastIndexBufferAddrs.data());

    lastPositionBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, lastPositionBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    lastPositionBufferAddr->uploadToStagingBuffer(lastPositionBufferAddrs.data());

    lastObjToWorldMat = vk::DeviceLocalBuffer::create(
        vma, device, lastObjToWorldMats.size() * sizeof(glm::mat4),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    lastObjToWorldMat->uploadToStagingBuffer(lastObjToWorldMats.data());

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> rayTracingMetaData{{
        blasOffsetsBuffer,
        indexBufferAddr,
        positionBufferAddr,
        materialBufferAddr,
        lastIndexBufferAddr,
        lastPositionBufferAddr,
        lastObjToWorldMat,
    }};

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> uploadPreBufferBarriers, uploadPostBufferBarriers;

    for (auto buffer : rayTracingMetaData) {
        if (buffer == nullptr) continue;
        uploadPreBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
        uploadPostBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
    }

    cmdBuffer->barriersBufferImage(uploadPreBufferBarriers, {});
    for (auto buffer : rayTracingMetaData) {
        if (buffer == nullptr) continue;
        buffer->uploadToBuffer(cmdBuffer);
    }
    cmdBuffer->barriersBufferImage(uploadPostBufferBarriers, {});
}

void WorldPrepareContext::render() {
    auto rayTracingContext = rayTracingModuleContext.lock();
    auto rayTracingModule = rayTracingContext != nullptr ? rayTracingContext->rayTracingModule.lock() : nullptr;
    auto worldPrepare1 = worldPrepare.lock();
    if (rayTracingModule == nullptr) { return; }
    if (worldPrepare1 == nullptr) { return; }
    RAD_PROFILE_SCOPE("world_prepare.render");

    std::shared_ptr<Framework> framework = Renderer::instance().framework();
    std::shared_ptr<FrameworkContext> context = frameworkContext.lock();
    std::shared_ptr<vk::VMA> vma = framework->vma();
    std::shared_ptr<vk::Device> device = framework->device();
    std::shared_ptr<vk::PhysicalDevice> physicalDevice = framework->physicalDevice();
    std::shared_ptr<vk::CommandBuffer> worldCommandBuffer = context->worldCommandBuffer;

    auto chunks = Renderer::instance().world()->chunks();
    auto entities = Renderer::instance().world()->entities();
    auto cameraPos = Renderer::instance().world()->getCameraPos();

    auto chunkBuildScheduler = chunks->chunkBuildScheduler();
    if (chunkBuildScheduler != nullptr) {
        RAD_PROFILE_SCOPE("world_prepare.chunk_scheduler");
        {
            RAD_PROFILE_SCOPE("world_prepare.try_check_batches_finish");
            chunkBuildScheduler->tryCheckBatchesFinish();
        }
        {
            RAD_PROFILE_SCOPE("world_prepare.try_schedule_batches");
            chunkBuildScheduler->tryScheduleBatches(chunkBuildScheduler->chunkBuildingBatchSize());
        }
    }

    std::unique_lock<std::recursive_mutex> lock(chunks->mutex());

    if (chunks->importantBLASBuilders().size() > 0) {
        vk::BLASBuilder::batchSubmit(chunks->importantBLASBuilders(), worldCommandBuffer);
    }

    if (entities->blasBatchBuilder() != nullptr) { entities->blasBatchBuilder()->submit(worldCommandBuffer); }

    worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask =
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    }});

    uint32_t blasAccu = 0, blasGroupAccu = 0;
    std::vector<uint32_t> blasOffset;
    hitGroupNameIds.clear();
    std::vector<uint64_t> indexBufferAddrs;
    std::vector<uint64_t> positionBufferAddrs, materialBufferAddrs;
    std::vector<uint64_t> lastIndexBufferAddrs;
    std::vector<uint64_t> lastPositionBufferAddrs;
    std::vector<glm::mat4> lastObjToWorldMats;

    tlasBuilder = vk::TLASBuilder::create();
    auto &instanceBuilder = tlasBuilder->beginInstanceBuilder();
    int blasIndex = 0;
    uint32_t entityInstanceCount = 0;
    uint32_t chunkInstanceCount = 0;
    uint32_t activeChunkSlots = 0;

    // Entity
    {
        RAD_PROFILE_SCOPE("world_prepare.entity_instance_loop");
        auto entityBatch = entities->entityBatch();

        if (entityBatch != nullptr) {
            std::unique_lock<std::recursive_mutex> entityHistoryLock(worldPrepare1->entityRenderDataBatchesMtx_);
            auto &previousEntityRenderDataBatches = worldPrepare1->previousEntityRenderDataBatches_;
            auto &emptyEntityRenderDataBatch = worldPrepare1->emptyEntityRenderDataBatch_;

            auto &previousEntityRenderDataBatch = previousEntityRenderDataBatches.empty() ?
                                                      emptyEntityRenderDataBatch :
                                                      previousEntityRenderDataBatches.back();
            if (previousEntityRenderDataBatches.size() > Renderer::instance().framework()->swapchain()->imageCount())
                previousEntityRenderDataBatches.pop();
            auto &currentEntityRenderDataBatch = previousEntityRenderDataBatches.emplace();

            auto worldUniformBuffer = Renderer::instance().buffers()->worldUniformBuffer();
            auto ubo = static_cast<vk::Data::WorldUBO *>(worldUniformBuffer->mappedPtr());

            auto &entities1 = entityBatch->entities;
            for (int i = 0; i < entities1.size(); i++) {
                VkGeometryInstanceFlagsKHR flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                // VkGeometryInstanceFlagsKHR flags = 0;
                VkTransformMatrixKHR transform;

                if (entities1[i]->prebuiltBLAS < 0) {
                    if (entities1[i]->coordinate == World::Coordinates::WORLD || !ubo) {
                        transform = {
                            1, 0, 0, static_cast<float>(entities1[i]->x - cameraPos.x), //
                            0, 1, 0, static_cast<float>(entities1[i]->y - cameraPos.y), //
                            0, 0, 1, static_cast<float>(entities1[i]->z - cameraPos.z), //
                        };
                    } else if (entities1[i]->coordinate == World::Coordinates::CAMERA) {
                        glm::mat4 viewMat = glm::transpose(ubo->cameraViewMatInv); // column major to row major

                        transform = {
                            viewMat[0][0], viewMat[0][1], viewMat[0][2], viewMat[0][3], //
                            viewMat[1][0], viewMat[1][1], viewMat[1][2], viewMat[1][3], //
                            viewMat[2][0], viewMat[2][1], viewMat[2][2], viewMat[2][3], //
                        };
                    } else if (entities1[i]->coordinate == World::Coordinates::CAMERA_SHIFT) {
                        glm::vec3 shift = glm::vec3(ubo->cameraViewMatInv[3]);
                        transform = {
                            1, 0, 0, shift.x, //
                            0, 1, 0, shift.y, //
                            0, 0, 1, shift.z, //
                        };
                    }

                    instanceBuilder.defineInstance(transform, blasIndex, entities1[i]->rayTracingFlag, blasGroupAccu,
                                                   flags, entities1[i]->blas);
                } else {
                    // auto &prebuiltBLAS =
                    //     Renderer::instance().framework()->prebuiltBLASs()[entityRenderData->prebuiltBLAS];
                    // transform = prebuiltBLAS.align(*entityRenderData->vertices, *entityRenderData->indices);

                    // instanceBuilder.defineInstance(transform, blasIndex, entityRenderData->rayTracingFlag,
                    // blasGroupAccu, flags,
                    //                                prebuiltBLAS.blas);
                    throw std::runtime_error("prebuilt blas not implemented yet!");
                }

                hitGroupNameIds.push_back(mcvr::HitGroupRegistry::shadowId());
                for (int j = 0; j < entities1[i]->geometryCount; j++) {
                    if (entities1[i]->geometryGroupIds != nullptr &&
                        j < static_cast<int>(entities1[i]->geometryGroupIds->size())) {
                        hitGroupNameIds.push_back((*entities1[i]->geometryGroupIds)[j]);
                    } else {
                        hitGroupNameIds.push_back(mcvr::HitGroupRegistry::defaultId());
                    }
                }

                for (int j = 0; j < entities1[i]->geometryCount; j++) {
                    indexBufferAddrs.push_back((*entities1[i]->indexBufferAddresses)[j]);
                    positionBufferAddrs.push_back((*entities1[i]->positionBufferAddresses)[j]);
                    materialBufferAddrs.push_back((*entities1[i]->materialBufferAddresses)[j]);
                }

                {
                    if (entities1[i]->hashCode) {
                        currentEntityRenderDataBatch[entities1[i]->hashCode].first = entities1[i];
                        currentEntityRenderDataBatch[entities1[i]->hashCode].second = transform;
                    }
                }

                {
                    glm::mat4 lastObjToWorldMat(1);
                    auto iter = previousEntityRenderDataBatch.find(entities1[i]->hashCode);
                    if (iter != previousEntityRenderDataBatch.end()) {
                        auto &previousEntityRenderData = (*iter).second.first;
                        if (previousEntityRenderData->geometryCount == entities1[i]->geometryCount) {
                            for (int j = 0; j < entities1[i]->geometryCount; j++) {
                                if ((*previousEntityRenderData->vertexCounts)[j] == (*entities1[i]->vertexCounts)[j] &&
                                    (*previousEntityRenderData->indexCounts)[j] == (*entities1[i]->indexCounts)[j]) {
                                    lastIndexBufferAddrs.push_back(
                                        (*previousEntityRenderData->indexBufferAddresses)[j]);
                                    lastPositionBufferAddrs.push_back(
                                        (*previousEntityRenderData->positionBufferAddresses)[j]);
                                } else {
                                    lastIndexBufferAddrs.push_back(0);
                                    lastPositionBufferAddrs.push_back(0);
                                }
                            }
                        } else {
                            for (int j = 0; j < entities1[i]->geometryCount; j++) {
                                lastIndexBufferAddrs.push_back(0);
                                lastPositionBufferAddrs.push_back(0);
                            }
                        }

                        VkTransformMatrixKHR lastObjToWorldVkMat = iter->second.second;
                        lastObjToWorldMat = glm::transpose(glm::mat4(glm::make_vec4(lastObjToWorldVkMat.matrix[0]), //
                                                                     glm::make_vec4(lastObjToWorldVkMat.matrix[1]), //
                                                                     glm::make_vec4(lastObjToWorldVkMat.matrix[2]), //
                                                                     glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)));
                    } else {
                        for (int j = 0; j < entities1[i]->geometryCount; j++) {
                            lastIndexBufferAddrs.push_back(0);
                            lastPositionBufferAddrs.push_back(0);
                        }
                    }
                    lastObjToWorldMats.push_back(lastObjToWorldMat);
                }

                blasOffset.push_back(blasAccu);
                blasAccu += entities1[i]->geometryCount;
                blasGroupAccu += entities1[i]->geometryCount + 1;

                blasIndex++;
                entityInstanceCount++;
            }
        }
    }

    // Chunk
    {
        RAD_PROFILE_SCOPE("world_prepare.chunk_instance_loop");
        auto &chunk1s = chunks->chunks();
        for (int i = 0; i < chunk1s.size(); i++) {
            auto &chunk1 = chunk1s[i];
            if (chunk1->blas == nullptr) {
                if (cachedChunkRows.size() > static_cast<size_t>(i)) { cachedChunkRows[i] = CachedChunkRow{}; }
                continue;
            }
            activeChunkSlots++;
            auto &chunkRow = refreshCachedChunkRow(i, chunk1);

            VkTransformMatrixKHR transform = {
                1, 0, 0, static_cast<float>(static_cast<double>(chunkRow.chunk->x) - cameraPos.x), //
                0, 1, 0, static_cast<float>(static_cast<double>(chunkRow.chunk->y) - cameraPos.y), //
                0, 0, 1, static_cast<float>(static_cast<double>(chunkRow.chunk->z) - cameraPos.z), //
            };

            instanceBuilder.defineInstance(transform, blasIndex, 0x01, blasGroupAccu, 0, chunkRow.blas);

            hitGroupNameIds.push_back(mcvr::HitGroupRegistry::shadowId());
            for (int j = 0; j < chunkRow.geometryCount; j++) {
                if (chunkRow.geometryGroupIds != nullptr && j < static_cast<int>(chunkRow.geometryGroupIds->size())) {
                    hitGroupNameIds.push_back((*chunkRow.geometryGroupIds)[j]);
                } else {
                    hitGroupNameIds.push_back(mcvr::HitGroupRegistry::defaultId());
                }
            }

            for (int j = 0; j < chunkRow.geometryCount; j++) {
                indexBufferAddrs.push_back((*chunkRow.indexBufferAddresses)[j]);
                positionBufferAddrs.push_back((*chunkRow.positionBufferAddresses)[j]);
                materialBufferAddrs.push_back((*chunkRow.materialBufferAddresses)[j]);
                lastIndexBufferAddrs.push_back(0);
                lastPositionBufferAddrs.push_back(0);
            }

            {
                glm::mat4 lastObjToWorldMat = glm::transpose(glm::mat4(
                    glm::vec4(1.0f, 0.0f, 0.0f,
                              static_cast<float>(static_cast<double>(chunkRow.chunk->x) - cameraPos.x)), //
                    glm::vec4(0.0f, 1.0f, 0.0f,
                              static_cast<float>(static_cast<double>(chunkRow.chunk->y) - cameraPos.y)), //
                    glm::vec4(0.0f, 0.0f, 1.0f,
                              static_cast<float>(static_cast<double>(chunkRow.chunk->z) - cameraPos.z)), //
                    glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)));
                lastObjToWorldMats.push_back(lastObjToWorldMat);
            }

            blasOffset.push_back(blasAccu);
            blasAccu += chunkRow.geometryCount;
            blasGroupAccu += chunkRow.geometryCount + 1;

            blasIndex++;
            chunkInstanceCount++;
        }
        RAD_PROFILE_COUNTER("world_prepare.chunk_slots", chunk1s.size());
        RAD_PROFILE_COUNTER("world_prepare.active_chunks", activeChunkSlots);
        RAD_PROFILE_COUNTER("world_prepare.chunk_instances", chunkInstanceCount);
    }
    RAD_PROFILE_COUNTER("world_prepare.entity_instances", entityInstanceCount);
    RAD_PROFILE_COUNTER("world_prepare.hit_group_names", hitGroupNameIds.size());
    RAD_PROFILE_COUNTER("world_prepare.tlas_instances", instanceBuilder.instances.size());

    if (instanceBuilder.instances.empty()) {
        tlas = nullptr;
        return;
    }

    {
        RAD_PROFILE_SCOPE("world_prepare.tlas_build");
        tlas = instanceBuilder.endInstanceBuilder(device, vma)
                   ->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                   ->querySizeInfo(device)
                   ->allocateBuffers(physicalDevice, device, vma)
                   ->buildAndSubmit(device, worldCommandBuffer);
    }

    {
        RAD_PROFILE_SCOPE("world_prepare.tlas_barrier");
        worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        }});
    }

    {
        RAD_PROFILE_SCOPE("world_prepare.upload_buffers");
        uploadBuffer(blasOffset, indexBufferAddrs, positionBufferAddrs, materialBufferAddrs, lastIndexBufferAddrs,
                     lastPositionBufferAddrs, lastObjToWorldMats);
    }
}

void WorldPrepareContext::setupHitGroupSbt(const std::unordered_map<std::string, uint32_t> &hitGroupNameToIndex,
                                           uint32_t fallbackHitGroupIndex,
                                           uint32_t shadowHitGroupIndex,
                                           std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                           std::shared_ptr<vk::SBT> updateSbt,
                                           std::shared_ptr<vk::SBT> querySbt) {
    RAD_PROFILE_SCOPE("world_prepare.setup_hit_group_sbt");
    std::vector<uint32_t> hitGroupIndices;
    hitGroupIndices.reserve(hitGroupNameIds.size());

    {
        RAD_PROFILE_SCOPE("world_prepare.resolve_hit_group_names");
        const auto registeredNames = mcvr::HitGroupRegistry::namesSnapshot();
        for (const uint32_t groupId : hitGroupNameIds) {
            if (groupId == mcvr::HitGroupRegistry::shadowId()) {
                hitGroupIndices.push_back(shadowHitGroupIndex);
                continue;
            }

            if (groupId >= registeredNames.size()) {
                hitGroupIndices.push_back(fallbackHitGroupIndex);
                continue;
            }

            const std::string &groupName = registeredNames[groupId];
            auto iter = hitGroupNameToIndex.find(groupName);
            hitGroupIndices.push_back(iter == hitGroupNameToIndex.end() ? fallbackHitGroupIndex : iter->second);
        }
    }

    if (updateSbt != nullptr) { updateSbt->setupHitSBT(hitGroupIndices, commandBuffer); }
    if (querySbt != nullptr) { querySbt->setupHitSBT(hitGroupIndices, commandBuffer); }
}
