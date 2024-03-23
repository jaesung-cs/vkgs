#ifndef PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H
#define PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H

namespace pygs {
namespace vk {
namespace shader {

const char* projection_comp = R"shader(
#version 450 core

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

layout (std430, set = 1, binding = 2) writeonly buffer Gaussian2d {
    float gaussian2d[];
};

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= point_count) return;

  vec3 v0 = vec3(gaussian3d[id * 9 + 0], gaussian3d[id * 9 + 1], gaussian3d[id * 9 + 2]);
  vec3 v1 = vec3(gaussian3d[id * 9 + 3], gaussian3d[id * 9 + 4], gaussian3d[id * 9 + 5]);
  vec4 pos = vec4(gaussian3d[id * 9 + 6], gaussian3d[id * 9 + 7], gaussian3d[id * 9 + 8], 1.f);
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
  if (abs(pos.x) > 1.f || abs(pos.y) > 1.f || pos.z < 0.f || pos.z > 1.f) {
    cov2d = mat2(0.f);
  }

  gaussian2d[id * 6 + 0] = cov2d[0][0];
  gaussian2d[id * 6 + 1] = cov2d[1][0];
  gaussian2d[id * 6 + 2] = cov2d[1][1];
  gaussian2d[id * 6 + 3] = pos.x;
  gaussian2d[id * 6 + 4] = pos.y;
  gaussian2d[id * 6 + 5] = pos.z;

  // TODO: calculate spherical harmonics
}

)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H
