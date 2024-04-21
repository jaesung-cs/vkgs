#include "pygs/engine/splat_load_thread.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cstring>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <pygs/util/timer.h>

namespace pygs {
namespace {

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

}  // namespace

class SplatLoadThread::Impl {
 public:
  Impl() = delete;

  Impl(vk::Context context) : context_(context) {
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(context_.device(), &fence_info, NULL, &fence_);
  }

  ~Impl() {
    if (thread_.joinable()) {
      terminate_ = true;
      thread_.join();
    }

    vkDestroyFence(context_.device(), fence_, NULL);
    if (staging_) vmaDestroyBuffer(context_.allocator(), staging_, allocation_);
  }

  void Start(const std::string& ply_filepath) {
    if (thread_.joinable()) {
      terminate_ = true;
      thread_.join();
    }

    // now that thread is completed, the following is thread safe
    terminate_ = false;
    total_point_count_ = 0;
    loaded_point_count_ = 0;

    thread_ = std::thread([this, ply_filepath] {
      Timer timer("splat load thread");
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

      // assuming all properties are float
      VkDeviceSize size = 60 * sizeof(uint32_t) +
                          static_cast<VkDeviceSize>(offset) * point_count;
      if (staging_size_ < size) {
        Timer timer("splat staging buffer allocation");
        if (staging_)
          vmaDestroyBuffer(context_.allocator(), staging_, allocation_);

        VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo allocation_create_info = {};
        allocation_create_info.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
        VmaAllocationInfo allocation_info;
        vmaCreateBuffer(context_.allocator(), &buffer_info,
                        &allocation_create_info, &staging_, &allocation_,
                        &allocation_info);
        staging_map_ = reinterpret_cast<uint8_t*>(allocation_info.pMappedData);
        staging_size_ = size;
      }

      vk::Buffer ply_buffer;
      {
        Timer timer("splat device buffer allocation");
        ply_buffer = vk::Buffer(context_, size,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      }

      // ply offsets
      std::vector<uint32_t> ply_offsets(60);
      ply_offsets[0] = offsets["x"] / 4;
      ply_offsets[1] = offsets["y"] / 4;
      ply_offsets[2] = offsets["z"] / 4;
      ply_offsets[3] = offsets["scale_0"] / 4;
      ply_offsets[4] = offsets["scale_1"] / 4;
      ply_offsets[5] = offsets["scale_2"] / 4;
      ply_offsets[6] = offsets["rot_1"] / 4;  // qx
      ply_offsets[7] = offsets["rot_2"] / 4;  // qy
      ply_offsets[8] = offsets["rot_3"] / 4;  // qz
      ply_offsets[9] = offsets["rot_0"] / 4;  // qw
      ply_offsets[10 + 0] = offsets["f_dc_0"] / 4;
      ply_offsets[10 + 16] = offsets["f_dc_1"] / 4;
      ply_offsets[10 + 32] = offsets["f_dc_2"] / 4;
      for (int i = 0; i < 15; ++i) {
        ply_offsets[10 + 1 + i] = offsets["f_rest_" + std::to_string(i)] / 4;
        ply_offsets[10 + 17 + i] =
            offsets["f_rest_" + std::to_string(15 + i)] / 4;
        ply_offsets[10 + 33 + i] =
            offsets["f_rest_" + std::to_string(30 + i)] / 4;
      }
      ply_offsets[58] = offsets["opacity"] / 4;
      ply_offsets[59] = offset / 4;

      // read all binary data
      std::vector<char> buffer;
      {
        Timer timer("splat binary read");
        buffer.resize(offset * point_count);

        constexpr uint32_t chunk_size = 131072;
        for (uint32_t start = 0; start < point_count; start += chunk_size) {
          if (terminate_) break;

          auto chunk_point_count =
              std::min<uint32_t>(chunk_size, point_count - start);
          in.read(buffer.data() + offset * start, offset * chunk_point_count);

          {
            std::unique_lock<std::mutex> guard{mutex_};
            loaded_point_count_ = start + chunk_point_count;
          }
        }
      }

      // copy to staging buffer
      {
        Timer timer("splat copy to staging");
        std::memcpy(staging_map_, ply_offsets.data(),
                    ply_offsets.size() * sizeof(uint32_t));
        std::memcpy(staging_map_ + 60 * sizeof(uint32_t), buffer.data(),
                    buffer.size() * sizeof(char));
      }

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

      // transfer command
      VkCommandBufferBeginInfo begin_info = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cb, &begin_info);

      VkBufferCopy region = {};
      region.srcOffset = 0;
      region.dstOffset = 0;
      region.size = size;
      vkCmdCopyBuffer(cb, staging_, ply_buffer, 1, &region);

      // transfer ownership
      std::vector<VkBufferMemoryBarrier2> buffer_barriers(1);
      buffer_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
      buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      buffer_barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      buffer_barriers[0].srcQueueFamilyIndex =
          context_.transfer_queue_family_index();
      buffer_barriers[0].dstQueueFamilyIndex =
          context_.graphics_queue_family_index();
      buffer_barriers[0].buffer = ply_buffer;
      buffer_barriers[0].offset = 0;
      buffer_barriers[0].size = size;
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

        VkSubmitInfo2 submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit_info.commandBufferInfoCount = 1;
        submit_info.pCommandBufferInfos = &command_buffer_info;
        vkQueueSubmit2(context_.transfer_queue(), 1, &submit_info, fence_);

        Timer timer("splat staging to device");
        vkWaitForFences(context_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
        vkResetFences(context_.device(), 1, &fence_);
      }

      // update loaded point count
      {
        std::unique_lock<std::mutex> guard{mutex_};
        buffer_barriers_.insert(buffer_barriers_.end(), buffer_barriers.begin(),
                                buffer_barriers.end());
        buffer_barriers.clear();
        ply_buffer_ = ply_buffer;
      }

      vkDestroyCommandPool(context_.device(), command_pool, NULL);
    });
  }

  Progress progress() {
    Progress result;
    std::unique_lock<std::mutex> guard{mutex_};
    result.total_point_count = total_point_count_;
    result.loaded_point_count = loaded_point_count_;
    result.ply_buffer = ply_buffer_;
    result.buffer_barriers = std::move(buffer_barriers_);

    ply_buffer_ = {};

    return result;
  }

  void cancel() { terminate_ = true; }

 private:
  vk::Context context_;

  std::thread thread_;
  std::mutex mutex_;
  std::atomic_bool terminate_{false};

  uint32_t total_point_count_ = 0;
  uint32_t loaded_point_count_ = 0;
  std::vector<VkBufferMemoryBarrier2> buffer_barriers_;

  VkFence fence_ = VK_NULL_HANDLE;

  // position: (N, 3), cov3d: (N, 6), sh: (N, 48), opacity: (N).
  // staging: (N, 58)
  VkDeviceSize staging_size_ = 0;
  VkBuffer staging_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  uint8_t* staging_map_ = nullptr;

  vk::Buffer ply_buffer_;
};

SplatLoadThread::SplatLoadThread() = default;

SplatLoadThread::SplatLoadThread(vk::Context context)
    : impl_(std::make_shared<Impl>(context)) {}

SplatLoadThread::~SplatLoadThread() = default;

void SplatLoadThread::Start(const std::string& ply_filepath) {
  impl_->Start(ply_filepath);
}

SplatLoadThread::Progress SplatLoadThread::progress() {
  return impl_->progress();
}

void SplatLoadThread::cancel() { impl_->cancel(); }

}  // namespace pygs
