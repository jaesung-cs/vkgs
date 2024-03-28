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
  vec3 camera_position;
  float pad0;
  uvec2 screen_size;  // (width, height)
};

layout (std430, set = 1, binding = 1) readonly buffer GaussianPosition {
  float gaussian_position[];  // (N, 3)
};

layout (std430, set = 1, binding = 2) readonly buffer GaussianCov3d {
  float gaussian_cov3d[];  // (N, 6)
};

layout (std430, set = 1, binding = 3) readonly buffer GaussianOpacity {
  float gaussian_opacity[];  // (N)
};

layout (std430, set = 1, binding = 4) readonly buffer GaussianSh {
  vec4 gaussian_sh[];  // (N, 3, 16), 16 packed with 4 vec4.
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

	vec3 dir = normalize(pos.xyz - camera_position);

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

  // low-pass filter
  cov2d[0][0] += 1.f / screen_size.x / screen_size.x;
  cov2d[1][1] += 1.f / screen_size.y / screen_size.y;

  pos = projection * pos;
  pos = pos / pos.w;

  // calculate spherical harmonics
  const float C0 = 0.28209479177387814f;
  const float C1 = 0.4886025119029199f;
  const float C20 = 1.0925484305920792f;
  const float C21 = 0.31539156525252005f;
  const float C22 = 0.5462742152960396f;
  const float C30 = 0.5900435899266435f;
  const float C31 = 2.890611442640554f;
  const float C32 = 0.4570457994644658f;
  const float C33 = 0.3731763325901154f;
  const float C34 = 1.445305721320277f;
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float xy = x * y;
  float yz = y * z;
  float xz = x * z;
  vec4 basis0 = vec4(C0, -C1 * y, C1 * z, -C1 * x);
  vec4 basis1 = vec4(C20 * xy, -C20 * yz, C21 * (2.f * zz - xx - yy), -C20 * xz);
  vec4 basis2 = vec4(C22 * (xx - yy), -C30 * y * (3.f * xx - yy), C31 * xy * z, -C32 * y * (4.f * zz - xx - yy));
  vec4 basis3 = vec4(C33 * z * (2.f * zz - 3.f * xx - 3.f * yy), -C32 * x * (4.f * zz - xx - yy), C34 * z * (xx - yy), -C30 * x * (xx - 3.f * yy));

  mat3x4 sh0 = mat3x4(gaussian_sh[gaussian_id * 12 + 0], gaussian_sh[gaussian_id * 12 + 4], gaussian_sh[gaussian_id * 12 +  8]);
  mat3x4 sh1 = mat3x4(gaussian_sh[gaussian_id * 12 + 1], gaussian_sh[gaussian_id * 12 + 5], gaussian_sh[gaussian_id * 12 +  9]);
  mat3x4 sh2 = mat3x4(gaussian_sh[gaussian_id * 12 + 2], gaussian_sh[gaussian_id * 12 + 6], gaussian_sh[gaussian_id * 12 + 10]);
  mat3x4 sh3 = mat3x4(gaussian_sh[gaussian_id * 12 + 3], gaussian_sh[gaussian_id * 12 + 7], gaussian_sh[gaussian_id * 12 + 11]);

  // row vector-matrix multiplication
  vec3 color = basis0 * sh0 + basis1 * sh1 + basis2 * sh2 + basis3 * sh3;

  // translation and clip
  color = max(color + 0.5f, 0.f);
  float opacity = gaussian_opacity[gaussian_id];

  instances[id * 10 + 0] = pos.x;
  instances[id * 10 + 1] = pos.y;
  instances[id * 10 + 2] = pos.z;
  instances[id * 10 + 3] = cov2d[0][0];
  instances[id * 10 + 4] = cov2d[1][0];
  instances[id * 10 + 5] = cov2d[1][1];
  instances[id * 10 + 6] = color.r;
  instances[id * 10 + 7] = color.g;
  instances[id * 10 + 8] = color.b;
  instances[id * 10 + 9] = opacity;
}

)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H
