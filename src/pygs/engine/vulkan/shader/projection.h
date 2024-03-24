#ifndef PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H
#define PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H

namespace pygs {
namespace vk {
namespace shader {

const char* projection_comp = R"shader(
#version 460 core

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (set = 0, binding = 0) uniform Camera {
  mat4 projection;
  mat4 view;
};

layout (set = 1, binding = 0) uniform Info {
  uint point_count;
};

layout (std430, set = 1, binding = 1) readonly buffer GaussianPosition {
  float gaussian_position[];  // (N, 3)
};

layout (std430, set = 1, binding = 2) readonly buffer GaussianCov3d {
  float gaussian_cov3d[];  // (N, 6)
};

layout (std430, set = 1, binding = 3) readonly buffer GaussianColor {
  // TODO: spherical harmonics
  float gaussian_color[];  // (N, 4), rgba.
};

layout (std430, set = 2, binding = 0) writeonly buffer DrawIndirect {
  uint indexCount;
  uint instanceCount;
  uint firstIndex;
  int vertexOffset;
  uint firstInstance;
};

layout (std430, set = 2, binding = 1) writeonly buffer Instances {
  float instances[];  // (M, 10). 3 for ndc position, 3 for cov2d, 4 for color
};

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= point_count) return;

  vec3 v0 = vec3(gaussian_cov3d[id * 6 + 0], gaussian_cov3d[id * 6 + 1], gaussian_cov3d[id * 6 + 2]);
  vec3 v1 = vec3(gaussian_cov3d[id * 6 + 3], gaussian_cov3d[id * 6 + 4], gaussian_cov3d[id * 6 + 5]);
  vec4 pos = vec4(gaussian_position[id * 3 + 0], gaussian_position[id * 3 + 1], gaussian_position[id * 3 + 2], 1.f);
  // [v0.x v0.y v0.z]
  // [v0.y v1.x v1.y]
  // [v0.z v1.y v1.z]
  mat3 cov3d = mat3(v0, v0.y, v1.xy, v0.z, v1.yz);

  // view matrix
  mat3 view3d = mat3(view);
  cov3d = view3d * cov3d * transpose(view3d);
  pos = view * pos;

  // projection
  float r = length(vec3(pos));
  mat3 J = mat3(
    -1.f / pos.z, 0.f, -2.f * pos.x / r,
    0.f, -1.f / pos.z, -2.f * pos.y / r,
    pos.x / pos.z / pos.z, pos.y / pos.z / pos.z, -2.f * pos.z / r
  );
  cov3d = J * cov3d * transpose(J);

  // projection xy
  mat2 projection_scale = mat2(projection);
  mat2 cov2d = projection_scale * mat2(cov3d) * projection_scale;

  pos = projection * pos;
  pos = pos / pos.w;

  // valid only when center is inside NDC clip space.
  if (abs(pos.x) <= 1.f && abs(pos.y) <= 1.f && pos.z >= 0.f && pos.z <= 1.f) {
    uint instance_index = atomicAdd(instanceCount, 1);

    // TODO: calculate spherical harmonics
    vec4 color = vec4(gaussian_color[id * 4 + 0], gaussian_color[id * 4 + 1], gaussian_color[id * 4 + 2], gaussian_color[id * 4 + 3]);

    instances[instance_index * 10 + 0] = pos.x;
    instances[instance_index * 10 + 1] = pos.y;
    instances[instance_index * 10 + 2] = pos.z;
    instances[instance_index * 10 + 3] = cov2d[0][0];
    instances[instance_index * 10 + 4] = cov2d[1][0];
    instances[instance_index * 10 + 5] = cov2d[1][1];
    instances[instance_index * 10 + 6] = color.r;
    instances[instance_index * 10 + 7] = color.g;
    instances[instance_index * 10 + 8] = color.b;
    instances[instance_index * 10 + 9] = color.a;
  }
}

)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H
