#version 460

layout(location = 0) in vec3 ndc_position;
layout(location = 1) in vec3 scale_rot;
layout(location = 2) in vec4 color;

layout(location = 0) out vec3 out_scale_rot;
layout(location = 1) out vec4 out_color;

void main() {
  gl_Position = vec4(ndc_position, 1.f);
  out_scale_rot = scale_rot;
  out_color = color;
}
