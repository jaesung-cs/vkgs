#ifndef VKGS_VULKAN_SEMAPHORE_H
#define VKGS_VULKAN_SEMAPHORE_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
namespace vk {

class Semaphore {
 public:
  Semaphore();

  explicit Semaphore(Context context);

  ~Semaphore();

  operator VkSemaphore() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_SEMAPHORE_H
