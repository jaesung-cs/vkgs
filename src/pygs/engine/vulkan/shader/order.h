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
};

layout (set = 1, binding = 0) uniform Info {
  uint point_count;
};

layout (std430, set = 1, binding = 1) readonly buffer Gaussian3d {
  float gaussian3d[];
};

layout (std430, set = 3, binding = 0) writeonly buffer RadixsortValue {
  uint depth[];
};

layout (std430, set = 3, binding = 1) writeonly buffer RadixsortIndex {
  uint index[];
};

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= point_count) return;

  vec4 pos = vec4(gaussian3d[id * 9 + 6], gaussian3d[id * 9 + 7], gaussian3d[id * 9 + 8], 1.f);
  pos = projection * view * pos;
  pos = pos / pos.w;

  // valid only when center is inside NDC clip space.
  index[id] = id;
  if (abs(pos.x) > 1.f || abs(pos.y) > 1.f || pos.z < 0.f || pos.z > 1.f) {
    depth[id] = -1;
  } else {
    depth[id] = floatBitsToUint(pos.z);
  }
}
)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_ORDER_H
