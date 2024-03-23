#ifndef PYGS_ENGINE_VULKAN_SHADER_MODULE_H
#define PYGS_ENGINE_VULKAN_SHADER_MODULE_H

#include <string>

#include <vulkan/vulkan.h>

namespace pygs {
namespace vk {

VkShaderModule CreateShaderModule(VkDevice device, VkShaderStageFlagBits stage,
                                  const std::string& source);

}
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_MODULE_H
