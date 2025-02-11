#version 460

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in vec3 in_scale_rot[];
layout(location = 1) in vec4 in_color[];

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_position;

void main() {
  vec2 scale = in_scale_rot[0].xy;
  float theta = in_scale_rot[0].z;

  // quad positions (-1, -1), (-1, 1), (1, -1), (1, 1), ccw in screen space.
  mat4x2 positions = mat4x2(vec2(-1.f, -1.f), vec2(-1.f, 1.f), vec2(1.f, -1.f), vec2(1.f, 1.f));

  mat2 rot = mat2(cos(theta), sin(theta), -sin(theta), cos(theta));
  mat2 scale_mat = mat2(scale.x, 0.f, 0.f, scale.y);
  float confidence_radius = 3.f;
  mat4x2 screen_positions = confidence_radius * rot * scale_mat * positions;

  for (int i = 0; i < 4; ++i) {
    gl_Position = gl_in[0].gl_Position + vec4(screen_positions[i], 0.f, 0.f);
    out_color = in_color[0];
    out_position = confidence_radius * positions[i];
    EmitVertex();
  }
  EndPrimitive();
}
