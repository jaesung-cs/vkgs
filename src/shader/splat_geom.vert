#version 460

layout(location = 0) in vec4 ndc_position;
layout(location = 1) in vec4 rot_scale;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 out_rot_scale;
layout(location = 1) out vec4 out_color;

void main() {
  gl_Position = vec4(ndc_position.xyz, 1.f);
  out_rot_scale = rot_scale;
  out_color = color;
}
