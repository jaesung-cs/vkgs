#ifndef VKGS_VULKAN_TIMELINE_SEMAPHORE_H
#define VKGS_VULKAN_TIMELINE_SEMAPHORE_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
namespace vk {

class TimelineSemaphore {
 public:
  TimelineSemaphore();

  explicit TimelineSemaphore(Context context, uint64_t initial_value = 0);

  ~TimelineSemaphore();

  operator VkSemaphore() const;

  uint64_t value() const;
  TimelineSemaphore& operator+=(uint64_t value);
  TimelineSemaphore& operator++();
  TimelineSemaphore operator++(int);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_TIMELINE_SEMAPHORE_H
