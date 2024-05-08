#ifndef VKGS_ENGINE_VULKAN_SWAPCHAIN_H
#define VKGS_ENGINE_VULKAN_SWAPCHAIN_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/image_spec.h"

namespace vkgs {
namespace vk {

class Swapchain {
 public:
  Swapchain();

  Swapchain(Context context, VkSurfaceKHR surface, bool vsync = true);

  ~Swapchain();

  operator VkSwapchainKHR() const;
  uint32_t width() const;
  uint32_t height() const;
  VkImageUsageFlags usage() const;
  VkFormat format() const;
  ImageSpec image_spec() const;
  int image_count() const;
  VkImage image(int index) const;
  VkImageView image_view(int index) const;

  void SetVsync(bool flag = true);
  bool ShouldRecreate() const;
  void Recreate();

  bool AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_SWAPCHAIN_H
