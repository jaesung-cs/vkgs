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

layout (std430, set = 1, binding = 4) readonly buffer GaussianSh0 {
  float gaussian_sh0[];  // (N, 3), rgb
};

layout (std430, set = 1, binding = 5) readonly buffer GaussianSh1 {
  float gaussian_sh1[];  // (N, 9), rgbrgbrgb
};

layout (std430, set = 1, binding = 6) readonly buffer GaussianSh2 {
  float gaussian_sh2[];  // (N, 15), rgbrgbrgbrgbrgb
};

layout (std430, set = 1, binding = 7) readonly buffer GaussianSh3 {
  float gaussian_sh3[];  // (N, 21), rgbrgbrgbrgbrgbrgbrgb
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
  const float low_pass_filter = 0.0000001f;
  cov2d[0][0] += low_pass_filter;
  cov2d[1][1] += low_pass_filter;

  pos = projection * pos;
  pos = pos / pos.w;

  // TODO: calculate spherical harmonics
  const float C0 = 0.28209479177387814f;
  vec3 color = C0 * vec3(
    gaussian_sh0[gaussian_id * 3 + 0],
    gaussian_sh0[gaussian_id * 3 + 1],
    gaussian_sh0[gaussian_id * 3 + 2]
  );

  const float C1 = 0.4886025119029199f;
  vec3 sh10 = vec3(gaussian_sh1[gaussian_id * 9 + 0], gaussian_sh1[gaussian_id * 9 + 1], gaussian_sh1[gaussian_id * 9 + 2]);
  vec3 sh11 = vec3(gaussian_sh1[gaussian_id * 9 + 3], gaussian_sh1[gaussian_id * 9 + 4], gaussian_sh1[gaussian_id * 9 + 5]);
  vec3 sh12 = vec3(gaussian_sh1[gaussian_id * 9 + 6], gaussian_sh1[gaussian_id * 9 + 7], gaussian_sh1[gaussian_id * 9 + 8]);
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;
	color += C1 * (-y * sh10 + z * sh11 - x * sh12);

  vec3 sh20 = vec3(gaussian_sh2[gaussian_id * 15 +  0], gaussian_sh2[gaussian_id * 15 +  1], gaussian_sh2[gaussian_id * 15 +  2]);
  vec3 sh21 = vec3(gaussian_sh2[gaussian_id * 15 +  3], gaussian_sh2[gaussian_id * 15 +  4], gaussian_sh2[gaussian_id * 15 +  5]);
  vec3 sh22 = vec3(gaussian_sh2[gaussian_id * 15 +  6], gaussian_sh2[gaussian_id * 15 +  7], gaussian_sh2[gaussian_id * 15 +  8]);
  vec3 sh23 = vec3(gaussian_sh2[gaussian_id * 15 +  9], gaussian_sh2[gaussian_id * 15 + 10], gaussian_sh2[gaussian_id * 15 + 11]);
  vec3 sh24 = vec3(gaussian_sh2[gaussian_id * 15 + 12], gaussian_sh2[gaussian_id * 15 + 13], gaussian_sh2[gaussian_id * 15 + 14]);
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float xy = x * y;
  float yz = y * z;
  float xz = x * z;
  color +=
    1.0925484305920792f * xy * sh20 +
    -1.0925484305920792f * yz * sh21 +
    0.31539156525252005f * (2.f * zz - xx - yy) * sh22 +
    -1.0925484305920792f * xz * sh23 +
    0.5462742152960396f * (xx - yy) * sh24;

  vec3 sh30 = vec3(gaussian_sh3[gaussian_id * 21 +  0], gaussian_sh3[gaussian_id * 21 +  1], gaussian_sh3[gaussian_id * 21 +  2]);
  vec3 sh31 = vec3(gaussian_sh3[gaussian_id * 21 +  3], gaussian_sh3[gaussian_id * 21 +  4], gaussian_sh3[gaussian_id * 21 +  5]);
  vec3 sh32 = vec3(gaussian_sh3[gaussian_id * 21 +  6], gaussian_sh3[gaussian_id * 21 +  7], gaussian_sh3[gaussian_id * 21 +  8]);
  vec3 sh33 = vec3(gaussian_sh3[gaussian_id * 21 +  9], gaussian_sh3[gaussian_id * 21 + 10], gaussian_sh3[gaussian_id * 21 + 11]);
  vec3 sh34 = vec3(gaussian_sh3[gaussian_id * 21 + 12], gaussian_sh3[gaussian_id * 21 + 13], gaussian_sh3[gaussian_id * 21 + 14]);
  vec3 sh35 = vec3(gaussian_sh3[gaussian_id * 21 + 15], gaussian_sh3[gaussian_id * 21 + 16], gaussian_sh3[gaussian_id * 21 + 17]);
  vec3 sh36 = vec3(gaussian_sh3[gaussian_id * 21 + 18], gaussian_sh3[gaussian_id * 21 + 19], gaussian_sh3[gaussian_id * 21 + 20]);
  color +=
    -0.5900435899266435f * y * (3.f * xx - yy) * sh30 +
    2.890611442640554f * xy * z * sh31 +
    -0.4570457994644658f * y * (4.f * zz - xx - yy) * sh32 +
    0.3731763325901154f * z * (2.f * zz - 3.f * xx - 3.f * yy) * sh33 +
    -0.4570457994644658f * x * (4.f * zz - xx - yy) * sh34 +
    1.445305721320277f * z * (xx - yy) * sh35 +
    -0.5900435899266435f * x * (xx - 3.f * yy) * sh36;

  vec4 out_color = vec4(
    max(color + 0.5f, 0.f),
    gaussian_opacity[gaussian_id]
  );

  instances[id * 10 + 0] = pos.x;
  instances[id * 10 + 1] = pos.y;
  instances[id * 10 + 2] = pos.z;
  instances[id * 10 + 3] = cov2d[0][0];
  instances[id * 10 + 4] = cov2d[1][0];
  instances[id * 10 + 5] = cov2d[1][1];
  instances[id * 10 + 6] = out_color.r;
  instances[id * 10 + 7] = out_color.g;
  instances[id * 10 + 8] = out_color.b;
  instances[id * 10 + 9] = out_color.a;
}

)shader";

}
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_PROJECTION_H
