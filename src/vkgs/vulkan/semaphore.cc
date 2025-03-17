#include "vkgs/vulkan/semaphore.h"

namespace vkgs {
namespace vk {

class Semaphore::Impl {
 public:
  Impl() = delete;

  Impl(Context context) : context_(context) {
    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(context_.device(), &semaphore_info, NULL, &semaphore_);
  }

  ~Impl() { vkDestroySemaphore(context_.device(), semaphore_, NULL); }

  operator VkSemaphore() const noexcept { return semaphore_; }

 private:
  Context context_;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

Semaphore::Semaphore() = default;

Semaphore::Semaphore(Context context) : impl_(std::make_shared<Impl>(context)) {}

Semaphore::~Semaphore() = default;

Semaphore::operator VkSemaphore() const { return *impl_; }

}  // namespace vk
}  // namespace vkgs
