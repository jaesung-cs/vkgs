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

layout (std430, set = 1, binding = 1) readonly buffer GaussianPosition {
  float gaussian_position[];  // (N, 3)
};

layout (std430, set = 1, binding = 2) readonly buffer GaussianCov3d {
  float gaussian_cov3d[];  // (N, 6)
};

layout (std430, set = 1, binding = 3) readonly buffer GaussianSh0 {
  float gaussian_opacity[];  // (N)
};

layout (std430, set = 1, binding = 4) readonly buffer GaussianOpacity {
  float gaussian_sh0[];  // (N, 3)
};

layout (std430, set = 2, binding = 0) readonly buffer DrawIndirect {
  uint indexCount;
  uint instanceCount;
  uint firstIndex;
  int vertexOffset;
  uint firstInstance;
};

layout (std430, set = 2, binding = 1) writeonly buffer Instances {
  float instances[];  // (M, 10). 3 for ndc position, 3 for cov2d, 4 for color
};

layout (std430, set = 2, binding = 3) readonly buffer InstanceIndex {
  uint index[];
};

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= instanceCount) return;

  uint gaussian_id = index[id];

  vec3 v0 = vec3(gaussian_cov3d[gaussian_id * 6 + 0], gaussian_cov3d[gaussian_id * 6 + 1], gaussian_cov3d[gaussian_id * 6 + 2]);
  vec3 v1 = vec3(gaussian_cov3d[gaussian_id * 6 + 3], gaussian_cov3d[gaussian_id * 6 + 4], gaussian_cov3d[gaussian_id * 6 + 5]);
  vec4 pos = vec4(gaussian_position[gaussian_id * 3 + 0], gaussian_position[gaussian_id * 3 + 1], gaussian_position[gaussian_id * 3 + 2], 1.f);
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

  // TODO: calculate spherical harmonics
  const float C0 = 0.28209479177387814f;
  vec3 sh0 = vec3(
    gaussian_sh0[gaussian_id * 3 + 0],
    gaussian_sh0[gaussian_id * 3 + 1],
    gaussian_sh0[gaussian_id * 3 + 2]
  );
  sh0 = max(sh0 * C0 + 0.5f, vec3(0.f));

  vec4 color = vec4(
    sh0,
    gaussian_opacity[gaussian_id]
  );

  instances[id * 10 + 0] = pos.x;
  instances[id * 10 + 1] = pos.y;
  instances[id * 10 + 2] = pos.z;
  instances[id * 10 + 3] = cov2d[0][0];
  instances[id * 10 + 4] = cov2d[1][0];
  instances[id * 10 + 5] = cov2d[1][1];
  instances[id * 10 + 6] = color.r;
  instances[id * 10 + 7] = color.g;
  instances[id * 10 + 8] = color.b;
  instances[id * 10 + 9] = color.a;
}

)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H
