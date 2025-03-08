#ifndef VKGS_VULKAN_SHADER_UNIFORMS_H
#define VKGS_VULKAN_SHADER_UNIFORMS_H

#include <glm/glm.hpp>

namespace vkgs {
namespace vk {
namespace shader {

struct alignas(64) Camera {
  glm::mat4 projection;
  glm::mat4 view;
  glm::vec3 camera_position;
  alignas(16) glm::uvec2 screen_size;
};

struct GaussianComputePushConstant {
  glm::mat4 model;
  uint32_t point_count;
};

}  // namespace shader
}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_SHADER_UNIFORMS_H
