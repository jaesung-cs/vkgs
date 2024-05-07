#version 460

layout (location = 0) in vec4 color;

layout (location = 0) out vec4 out_color;

void main() {
  // premultiplied alpha
  out_color = vec4(color.rgb * color.a, color.a);
}
