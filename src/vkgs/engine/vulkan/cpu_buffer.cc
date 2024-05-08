#include "vkgs/engine/vulkan/cpu_buffer.h"

#include <cstring>

#include "vk_mem_alloc.h"

namespace vkgs {
namespace vk {

class CpuBuffer::Impl {
 public:
  Impl() = delete;

  Impl(Context context, VkDeviceSize size) : context_(context), size_(size) {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    allocation_create_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocation_info;
    vmaCreateBuffer(context.allocator(), &buffer_info, &allocation_create_info,
                    &buffer_, &allocation_, &allocation_info);
    map_ = allocation_info.pMappedData;
  }

  ~Impl() { vmaDestroyBuffer(context_.allocator(), buffer_, allocation_); }

  operator VkBuffer() const noexcept { return buffer_; }

  const void* data() const noexcept { return map_; }
  VkDeviceSize size() const noexcept { return size_; }

 private:
  Context context_;
  VkDeviceSize size_ = 0;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  void* map_ = nullptr;
};

CpuBuffer::CpuBuffer() = default;

CpuBuffer::CpuBuffer(Context context, VkDeviceSize size)
    : impl_(std::make_shared<Impl>(context, size)) {}

CpuBuffer::~CpuBuffer() = default;

CpuBuffer::operator VkBuffer() const { return *impl_; }

const void* CpuBuffer::data() const { return impl_->data(); }
VkDeviceSize CpuBuffer::size() const { return impl_->size(); }

}  // namespace vk
}  // namespace vkgs
