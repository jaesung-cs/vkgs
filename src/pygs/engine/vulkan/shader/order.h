#ifndef PYGS_ENGINE_VULKAN_SHADER_ORDER_H
#define PYGS_ENGINE_VULKAN_SHADER_ORDER_H

namespace pygs {
namespace vk {
namespace shader {

const char* order_comp = R"shader(
#version 460

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (set = 0, binding = 0) uniform Camera {
  mat4 projection;
  mat4 view;
  vec3 camera_position;
  float pad0;
  uvec2 screen_size;  // (width, height)
};

layout (push_constant, std430) uniform PushConstants {
  mat4 model;
};

layout (set = 1, binding = 0) uniform Info {
  uint point_count;
};

layout (std430, set = 1, binding = 1) readonly buffer GaussianPosition {
  float gaussian_position[];  // (N, 3)
};

layout (std430, set = 2, binding = 2) buffer NumElements {
  uint num_elements;
};

layout (std430, set = 2, binding = 3) writeonly buffer InstanceKey {
  uint key[];
};

layout (std430, set = 2, binding = 4) writeonly buffer InstanceIndex {
  uint index[];
};

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= point_count) return;

  vec4 pos = vec4(gaussian_position[id * 3 + 0], gaussian_position[id * 3 + 1], gaussian_position[id * 3 + 2], 1.f);
  pos = projection * view * model * pos;
  pos = pos / pos.w;
  float depth = pos.z;

  // valid only when center is inside NDC clip space.
  if (abs(pos.x) <= 1.f && abs(pos.y) <= 1.f && pos.z >= 0.f && pos.z <= 1.f) {
    uint instance_index = atomicAdd(num_elements, 1);
    key[instance_index] = floatBitsToUint(1.f - depth);
    index[instance_index] = id;
  }
}
)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_ORDER_H
