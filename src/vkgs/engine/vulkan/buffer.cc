#include "vkgs/engine/vulkan/buffer.h"

#include <cstring>

#include "vk_mem_alloc.h"

namespace vkgs {
namespace vk {

class Buffer::Impl {
 public:
  Impl() = delete;

  Impl(Context context, VkDeviceSize size, VkBufferUsageFlags usage)
      : context_(context), size_(size) {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage;
    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(context.allocator(), &buffer_info, &allocation_create_info,
                    &buffer_, &allocation_, NULL);
  }

  ~Impl() {
    vmaDestroyBuffer(context_.allocator(), buffer_, allocation_);

    if (staging_buffer_) {
      vmaDestroyBuffer(context_.allocator(), staging_buffer_,
                       staging_allocation_);
    }
  }

  operator VkBuffer() const noexcept { return buffer_; }

  VkDeviceSize size() const { return size_; }

  void FromCpu(VkCommandBuffer command_buffer, const void* src,
               VkDeviceSize size) {
    if (!staging_buffer_) {
      VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      buffer_info.size = size_;
      buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      VmaAllocationCreateInfo allocation_create_info = {};
      allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
      allocation_create_info.flags =
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT;
      VmaAllocationInfo allocation_info;
      vmaCreateBuffer(context_.allocator(), &buffer_info,
                      &allocation_create_info, &staging_buffer_,
                      &staging_allocation_, &allocation_info);
      map_ = allocation_info.pMappedData;
    }

    std::memcpy(map_, src, size);

    VkBufferCopy region = {};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = size;
    vkCmdCopyBuffer(command_buffer, staging_buffer_, buffer_, 1, &region);
  }

 private:
  Context context_;
  VkDeviceSize size_ = 0;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;

  VkBuffer staging_buffer_ = VK_NULL_HANDLE;
  VmaAllocation staging_allocation_ = VK_NULL_HANDLE;
  void* map_ = nullptr;
};

Buffer::Buffer() = default;

Buffer::Buffer(Context context, VkDeviceSize size, VkBufferUsageFlags usage)
    : impl_(std::make_shared<Impl>(context, size, usage)) {}

Buffer::~Buffer() = default;

Buffer::operator VkBuffer() const { return *impl_; }

VkDeviceSize Buffer::size() const { return impl_->size(); }

void Buffer::FromCpu(VkCommandBuffer command_buffer, const void* src,
                     VkDeviceSize size) {
  impl_->FromCpu(command_buffer, src, size);
}

}  // namespace vk
}  // namespace vkgs
