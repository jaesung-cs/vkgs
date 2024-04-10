#ifndef PYGS_ENGINE_VULKAN_SHADER_INVERSE_INDEX_H
#define PYGS_ENGINE_VULKAN_SHADER_INVERSE_INDEX_H

namespace pygs {
namespace vk {
namespace shader {

const char* inverse_index_comp = R"shader(
#version 460 core

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (std430, set = 2, binding = 2) buffer VisiblePointCount {
  uint visible_point_count;
};

layout (std430, set = 2, binding = 4) readonly buffer InstanceIndex {
  uint index[];
};

layout (std430, set = 2, binding = 5) writeonly buffer InverseMap {
  int inverse_index[];  // (N), inverse map from id to sorted index
};

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= visible_point_count) return;

  inverse_index[index[id]] = int(id);
}

)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_INVERSE_INDEX_H
