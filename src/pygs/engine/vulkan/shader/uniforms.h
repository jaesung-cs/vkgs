#ifndef PYGS_ENGINE_VULKAN_SHADER_UNIFORMS_H
#define PYGS_ENGINE_VULKAN_SHADER_UNIFORMS_H

#include <glm/glm.hpp>

namespace pygs {
namespace vk {
namespace shader {

struct Camera {
  glm::mat4 projection;
  glm::mat4 view;
};

struct SplatInfo {
  uint32_t point_count;
};

}  // namespace shader
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_UNIFORMS_H
