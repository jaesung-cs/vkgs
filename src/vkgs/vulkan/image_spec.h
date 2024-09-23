#ifndef VKGS_VULKAN_IMAGE_SPEC_H
#define VKGS_VULKAN_IMAGE_SPEC_H

#include <vulkan/vulkan.h>

namespace vkgs {
namespace vk {

struct ImageSpec {
  uint32_t width = 0;
  uint32_t height = 0;
  VkImageUsageFlags usage = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_IMAGE_SPEC_H
