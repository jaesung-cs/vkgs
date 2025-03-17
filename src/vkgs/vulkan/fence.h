#ifndef VKGS_VULKAN_FENCE_H
#define VKGS_VULKAN_FENCE_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
namespace vk {

class Fence {
 public:
  Fence();

  explicit Fence(Context context);

  ~Fence();

  operator VkFence() const;

  void wait() const;
  void reset();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_FENCE_H
