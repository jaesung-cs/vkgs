#ifndef PYGS_ENGINE_VULKAN_IMAGE_SPEC_H
#define PYGS_ENGINE_VULKAN_IMAGE_SPEC_H

#include <vulkan/vulkan.h>

namespace pygs {
namespace vk {

struct ImageSpec {
  uint32_t width = 0;
  uint32_t height = 0;
  VkImageUsageFlags usage = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_IMAGE_SPEC_H
