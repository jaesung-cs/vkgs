#include "vkgs/vulkan/timeline_semaphore.h"

namespace vkgs {
namespace vk {

class TimelineSemaphore::Impl {
 public:
  Impl() = delete;

  Impl(Context context, uint64_t initial_value = 0) : context_(context), value_(initial_value) {
    VkSemaphoreTypeCreateInfo timeline_create_info = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_create_info.initialValue = initial_value;
    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore_info.pNext = &timeline_create_info;
    vkCreateSemaphore(context_.device(), &semaphore_info, NULL, &semaphore_);
  }

  ~Impl() { vkDestroySemaphore(context_.device(), semaphore_, NULL); }

  operator VkSemaphore() const noexcept { return semaphore_; }

  uint64_t value() const noexcept { return value_; }

  Impl& operator+=(uint64_t value) noexcept {
    value_ += value;
    return *this;
  }

  Impl& operator++() noexcept {
    value_++;
    return *this;
  }

 private:
  Context context_;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
  uint64_t value_ = 0;
};

TimelineSemaphore::TimelineSemaphore() = default;

TimelineSemaphore::TimelineSemaphore(Context context, uint64_t initial_value)
    : impl_(std::make_shared<Impl>(context, initial_value)) {}

TimelineSemaphore::~TimelineSemaphore() = default;

TimelineSemaphore::operator VkSemaphore() const { return *impl_; }

uint64_t TimelineSemaphore::value() const { return impl_->value(); }

TimelineSemaphore& TimelineSemaphore::operator+=(uint64_t value) {
  (*impl_) += value;
  return *this;
}

TimelineSemaphore& TimelineSemaphore::operator++() {
  ++(*impl_);
  return *this;
}

TimelineSemaphore TimelineSemaphore::operator++(int) {
  ++(*impl_);
  return *this;
}

}  // namespace vk
}  // namespace vkgs
