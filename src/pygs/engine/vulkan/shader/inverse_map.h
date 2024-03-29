#ifndef PYGS_ENGINE_VULKAN_SHADER_INVERSE_MAP_H
#define PYGS_ENGINE_VULKAN_SHADER_INVERSE_MAP_H

namespace pygs {
namespace vk {
namespace shader {

const char* inverse_map_comp = R"shader(
#version 460 core

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (std430, set = 2, binding = 0) readonly buffer DrawIndirect {
  uint indexCount;
  uint instanceCount;
  uint firstIndex;
  int vertexOffset;
  uint firstInstance;
};

layout (std430, set = 2, binding = 3) readonly buffer InstanceIndex {
  uint index[];
};

layout (std430, set = 2, binding = 4) writeonly buffer InverseMap {
  int inverse_map[];  // (N), inverse map from id to sorted index
};

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= instanceCount) return;

  inverse_map[index[id]] = int(id);
}

)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_INVERSE_MAP_H
