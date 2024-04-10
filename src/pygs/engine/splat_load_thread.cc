#include "pygs/engine/splat_load_thread.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <algorithm>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

namespace pygs {
namespace {

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

}  // namespace

class SplatLoadThread::Impl {
 public:
  Impl() = delete;

  Impl(vk::Context context) : context_(context) {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = CHUNK_SIZE * 58 * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationInfo allocation_info;
    vmaCreateBuffer(context.allocator(), &buffer_info, &allocation_create_info,
                    &staging_, &allocation_, &allocation_info);
    staging_map_ = reinterpret_cast<uint8_t*>(allocation_info.pMappedData);
  }

  ~Impl() {
    if (thread_.joinable()) {
      terminate_ = true;
      thread_.join();
    }

    if (semaphore_) vkDestroySemaphore(context_.device(), semaphore_, NULL);
    vmaDestroyBuffer(context_.allocator(), staging_, allocation_);
  }

  void Start(const std::string& ply_filepath, vk::Buffer position,
             vk::Buffer cov3d, vk::Buffer sh, vk::Buffer opacity) {
    if (thread_.joinable()) {
      terminate_ = true;
      thread_.join();
    }

    // now that thread is completed, the following is thread safe
    terminate_ = false;
    total_point_count_ = 0;
    loaded_point_count_ = 0;
    timeline_ = 0;

    if (semaphore_) vkDestroySemaphore(context_.device(), semaphore_, NULL);

    VkSemaphoreTypeCreateInfo semaphore_type_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    semaphore_type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore_info.pNext = &semaphore_type_info;
    vkCreateSemaphore(context_.device(), &semaphore_info, NULL, &semaphore_);

    thread_ = std::thread([=] {
      std::ifstream in(ply_filepath, std::ios::binary);

      // parse header
      std::unordered_map<std::string, int> offsets;
      int offset = 0;
      size_t point_count = 0;
      std::string line;
      while (std::getline(in, line)) {
        if (line == "end_header") break;

        std::istringstream iss(line);
        std::string word;
        iss >> word;
        if (word == "property") {
          int size = 0;
          std::string type, property;
          iss >> type >> property;
          if (type == "float") {
            size = 4;
          }
          offsets[property] = offset;
          offset += size;
        } else if (word == "element") {
          std::string type;
          size_t count;
          iss >> type >> count;
          if (type == "vertex") {
            point_count = count;
          }
        }
      }

      // update total point count
      {
        std::unique_lock<std::mutex> guard{mutex_};
        total_point_count_ = point_count;
      }

      std::vector<char> buffer(offset * CHUNK_SIZE);
      std::vector<float> raw_position(3 * CHUNK_SIZE);
      std::vector<float> raw_cov3d(6 * CHUNK_SIZE);
      std::vector<float> raw_sh(48 * CHUNK_SIZE);
      std::vector<float> raw_opacity(CHUNK_SIZE);
      std::vector<float> rest(45);
      std::vector<VkBufferMemoryBarrier2> buffer_barriers;

      // thread-local command buffer
      VkCommandPoolCreateInfo command_pool_info = {
          VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      command_pool_info.queueFamilyIndex =
          context_.transfer_queue_family_index();
      VkCommandPool command_pool = VK_NULL_HANDLE;
      vkCreateCommandPool(context_.device(), &command_pool_info, NULL,
                          &command_pool);

      VkCommandBufferAllocateInfo command_buffer_info = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      command_buffer_info.commandPool = command_pool;
      command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      command_buffer_info.commandBufferCount = 1;
      VkCommandBuffer cb = VK_NULL_HANDLE;
      vkAllocateCommandBuffers(context_.device(), &command_buffer_info, &cb);

      for (uint32_t start = 0; start < point_count; start += CHUNK_SIZE) {
        if (terminate_) break;

        // read [start:end]
        uint32_t end = std::min<uint32_t>(start + CHUNK_SIZE, point_count);
        uint32_t chunk_point_count = end - start;

        // read chunk
        in.read(buffer.data(), offset * chunk_point_count);
        for (int i = 0; i < chunk_point_count; ++i) {
          float x = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                              offsets.at("x"));
          float y = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                              offsets.at("y"));
          float z = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                              offsets.at("z"));
          float f_dc_0 = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                                   offsets.at("f_dc_0"));
          float f_dc_1 = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                                   offsets.at("f_dc_1"));
          float f_dc_2 = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                                   offsets.at("f_dc_2"));
          float opacity = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                                    offsets.at("opacity"));
          float sx = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("scale_0"));
          float sy = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("scale_1"));
          float sz = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("scale_2"));
          float rw = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("rot_0"));
          float rx = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("rot_1"));
          float ry = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("rot_2"));
          float rz = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("rot_3"));

          raw_position[i * 3 + 0] = x;
          raw_position[i * 3 + 1] = y;
          raw_position[i * 3 + 2] = z;

          raw_sh[i * 48 + 0] = f_dc_0;
          raw_sh[i * 48 + 16] = f_dc_1;
          raw_sh[i * 48 + 32] = f_dc_2;

          for (int j = 0; j < 45; ++j) {
            rest[j] = *reinterpret_cast<float*>(
                buffer.data() + i * offset +
                offsets.at("f_rest_" + std::to_string(j)));
          }

          for (int j = 0; j < 15; ++j) {
            raw_sh[i * 48 + 1 + j] = rest[0 + j];
            raw_sh[i * 48 + 17 + j] = rest[15 + j];
            raw_sh[i * 48 + 33 + j] = rest[30 + j];
          }

          raw_opacity[i] = sigmoid(opacity);

          glm::mat3 s = glm::mat3(1.f);
          s[0][0] = std::exp(sx);
          s[1][1] = std::exp(sy);
          s[2][2] = std::exp(sz);
          float r = std::sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
          glm::quat q;
          q.w = rw / r;
          q.x = rx / r;
          q.y = ry / r;
          q.z = rz / r;
          glm::mat3 rot = glm::toMat3(q);

          glm::mat3 m = rot * s * s * glm::transpose(rot);  // cov = RSSR^T
          raw_cov3d[i * 6 + 0] = m[0][0];
          raw_cov3d[i * 6 + 1] = m[1][0];
          raw_cov3d[i * 6 + 2] = m[2][0];
          raw_cov3d[i * 6 + 3] = m[1][1];
          raw_cov3d[i * 6 + 4] = m[2][1];
          raw_cov3d[i * 6 + 5] = m[2][2];
        }

        // wait for previous transfer to complete
        VkSemaphoreWaitInfo wait_info = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &semaphore_;
        wait_info.pValues = &timeline_;
        vkWaitSemaphores(context_.device(), &wait_info, UINT64_MAX);

        // update loaded point count
        {
          std::unique_lock<std::mutex> guard{mutex_};
          loaded_point_count_ = timeline_;
          buffer_barriers_.insert(buffer_barriers_.end(),
                                  buffer_barriers.begin(),
                                  buffer_barriers.end());
          buffer_barriers.clear();
        }

        // copy to staging buffer
        std::memcpy(staging_map_, raw_position.data(),
                    chunk_point_count * 3 * sizeof(float));
        std::memcpy(staging_map_ + chunk_point_count * 3 * sizeof(float),
                    raw_cov3d.data(), chunk_point_count * 6 * sizeof(float));
        std::memcpy(staging_map_ + chunk_point_count * 9 * sizeof(float),
                    raw_sh.data(), chunk_point_count * 48 * sizeof(float));
        std::memcpy(staging_map_ + chunk_point_count * 57 * sizeof(float),
                    raw_opacity.data(), chunk_point_count * 1 * sizeof(float));

        // transfer command
        VkCommandBufferBeginInfo begin_info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &begin_info);

        VkBufferCopy position_region = {};
        position_region.srcOffset = 0;
        position_region.dstOffset = start * 3 * sizeof(float);
        position_region.size = chunk_point_count * 3 * sizeof(float);

        VkBufferCopy cov3d_region = {};
        cov3d_region.srcOffset = chunk_point_count * 3 * sizeof(float);
        cov3d_region.dstOffset = start * 6 * sizeof(float);
        cov3d_region.size = chunk_point_count * 6 * sizeof(float);

        VkBufferCopy sh_region = {};
        sh_region.srcOffset = chunk_point_count * 9 * sizeof(float);
        sh_region.dstOffset = start * 48 * sizeof(float);
        sh_region.size = chunk_point_count * 48 * sizeof(float);

        VkBufferCopy opacity_region = {};
        opacity_region.srcOffset = chunk_point_count * 57 * sizeof(float);
        opacity_region.dstOffset = start * 1 * sizeof(float);
        opacity_region.size = chunk_point_count * 1 * sizeof(float);

        vkCmdCopyBuffer(cb, staging_, position, 1, &position_region);
        vkCmdCopyBuffer(cb, staging_, cov3d, 1, &cov3d_region);
        vkCmdCopyBuffer(cb, staging_, sh, 1, &sh_region);
        vkCmdCopyBuffer(cb, staging_, opacity, 1, &opacity_region);

        // transfer ownership
        buffer_barriers.resize(4);
        buffer_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[0].srcQueueFamilyIndex =
            context_.transfer_queue_family_index();
        buffer_barriers[0].dstQueueFamilyIndex =
            context_.graphics_queue_family_index();
        buffer_barriers[0].buffer = position;
        buffer_barriers[0].offset = start * 3 * sizeof(float);
        buffer_barriers[0].size = chunk_point_count * 3 * sizeof(float);

        buffer_barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        buffer_barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[1].srcQueueFamilyIndex =
            context_.transfer_queue_family_index();
        buffer_barriers[1].dstQueueFamilyIndex =
            context_.graphics_queue_family_index();
        buffer_barriers[1].buffer = cov3d;
        buffer_barriers[1].offset = start * 6 * sizeof(float);
        buffer_barriers[1].size = chunk_point_count * 6 * sizeof(float);

        buffer_barriers[2].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        buffer_barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[2].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[2].srcQueueFamilyIndex =
            context_.transfer_queue_family_index();
        buffer_barriers[2].dstQueueFamilyIndex =
            context_.graphics_queue_family_index();
        buffer_barriers[2].buffer = sh;
        buffer_barriers[2].offset = start * 48 * sizeof(float);
        buffer_barriers[2].size = chunk_point_count * 48 * sizeof(float);

        buffer_barriers[3].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        buffer_barriers[3].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[3].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[3].srcQueueFamilyIndex =
            context_.transfer_queue_family_index();
        buffer_barriers[3].dstQueueFamilyIndex =
            context_.graphics_queue_family_index();
        buffer_barriers[3].buffer = opacity;
        buffer_barriers[3].offset = start * 1 * sizeof(float);
        buffer_barriers[3].size = chunk_point_count * 1 * sizeof(float);

        VkDependencyInfo dependency = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = buffer_barriers.size();
        dependency.pBufferMemoryBarriers = buffer_barriers.data();
        vkCmdPipelineBarrier2(cb, &dependency);

        vkEndCommandBuffer(cb);

        // submit
        {
          VkCommandBufferSubmitInfo command_buffer_info = {
              VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
          command_buffer_info.commandBuffer = cb;

          VkSemaphoreSubmitInfo signal_semaphore_info = {
              VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
          signal_semaphore_info.semaphore = semaphore_;
          signal_semaphore_info.value = timeline_ + chunk_point_count;
          signal_semaphore_info.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

          VkSubmitInfo2 submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
          submit_info.commandBufferInfoCount = 1;
          submit_info.pCommandBufferInfos = &command_buffer_info;
          submit_info.signalSemaphoreInfoCount = 1;
          submit_info.pSignalSemaphoreInfos = &signal_semaphore_info;
          vkQueueSubmit2(context_.transfer_queue(), 1, &submit_info, NULL);

          timeline_ += chunk_point_count;
        }
      }

      // wait for previous transfer to complete
      VkSemaphoreWaitInfo wait_info = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
      wait_info.semaphoreCount = 1;
      wait_info.pSemaphores = &semaphore_;
      wait_info.pValues = &timeline_;
      vkWaitSemaphores(context_.device(), &wait_info, UINT64_MAX);

      // update loaded point count
      {
        std::unique_lock<std::mutex> guard{mutex_};
        loaded_point_count_ = timeline_;
        buffer_barriers_.insert(buffer_barriers_.end(), buffer_barriers.begin(),
                                buffer_barriers.end());
        buffer_barriers.clear();
      }

      vkDestroyCommandPool(context_.device(), command_pool, NULL);
    });
  }

  Progress progress() {
    Progress result;
    std::unique_lock<std::mutex> guard{mutex_};
    result.total_point_count = total_point_count_;
    result.loaded_point_count = loaded_point_count_;
    result.buffer_barriers = std::move(buffer_barriers_);
    return result;
  }

  void cancel() { terminate_ = true; }

 private:
  vk::Context context_;

  std::thread thread_;
  std::atomic_bool terminate_ = false;
  std::mutex mutex_;

  uint32_t total_point_count_ = 0;
  uint32_t loaded_point_count_ = 0;
  std::vector<VkBufferMemoryBarrier2> buffer_barriers_;
  static constexpr uint32_t CHUNK_SIZE = 2048;

  // position: (N, 3), cov3d: (N, 6), sh: (N, 48), opacity: (N).
  // staging: (N, 58)
  VkBuffer staging_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  uint8_t* staging_map_ = nullptr;

  VkSemaphore semaphore_ = VK_NULL_HANDLE;
  uint64_t timeline_ = 0;
};

SplatLoadThread::SplatLoadThread() = default;

SplatLoadThread::SplatLoadThread(vk::Context context)
    : impl_(std::make_shared<Impl>(context)) {}

SplatLoadThread::~SplatLoadThread() = default;

void SplatLoadThread::Start(const std::string& ply_filepath,
                            vk::Buffer position, vk::Buffer cov3d,
                            vk::Buffer sh, vk::Buffer opacity) {
  impl_->Start(ply_filepath, position, cov3d, sh, opacity);
}

SplatLoadThread::Progress SplatLoadThread::progress() {
  return impl_->progress();
}

void SplatLoadThread::cancel() { impl_->cancel(); }

}  // namespace pygs
