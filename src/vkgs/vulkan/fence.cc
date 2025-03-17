#include "vkgs/vulkan/fence.h"

namespace vkgs {
namespace vk {

class Fence::Impl {
 public:
  Impl() = delete;

  Impl(Context context) : context_(context) {
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(context_.device(), &fence_info, NULL, &fence_);
  }

  ~Impl() { vkDestroyFence(context_.device(), fence_, NULL); }

  operator VkFence() const noexcept { return fence_; }

  void wait() const { vkWaitForFences(context_.device(), 1, &fence_, VK_TRUE, UINT64_MAX); }

  void reset() { vkResetFences(context_.device(), 1, &fence_); }

 private:
  Context context_;
  VkFence fence_ = VK_NULL_HANDLE;
};

Fence::Fence() = default;

Fence::Fence(Context context) : impl_(std::make_shared<Impl>(context)) {}

Fence::~Fence() = default;

Fence::operator VkFence() const { return *impl_; }

void Fence::wait() const { impl_->wait(); }

void Fence::reset() { impl_->reset(); }

}  // namespace vk
}  // namespace vkgs
