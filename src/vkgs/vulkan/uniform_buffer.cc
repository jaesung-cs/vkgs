#include "vkgs/vulkan/uniform_buffer.h"

#include "vk_mem_alloc.h"

namespace vkgs {
namespace vk {

class UniformBufferBase::Impl {
 public:
  Impl() = delete;

  Impl(Context context, VkDeviceSize size) : context_(context) {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

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

  void* ptr() { return map_; }
  const void* ptr() const { return map_; }

 private:
  Context context_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  void* map_ = nullptr;
};

UniformBufferBase::UniformBufferBase() = default;

UniformBufferBase::UniformBufferBase(Context context, VkDeviceSize size)
    : impl_(std::make_shared<Impl>(context, size)) {}

UniformBufferBase::~UniformBufferBase() = default;

UniformBufferBase::operator VkBuffer() const { return *impl_; }

void* UniformBufferBase::ptr() { return impl_->ptr(); }

const void* UniformBufferBase::ptr() const { return impl_->ptr(); }

}  // namespace vk
}  // namespace vkgs
