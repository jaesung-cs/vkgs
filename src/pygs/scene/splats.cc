#include <pygs/scene/splats.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace pygs {
namespace {

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

}  // namespace

Splats Splats::load(const std::string& ply_filepath) {
  Splats splats;

  {
    std::unordered_map<std::string, int> offsets;
    int offset = 0;
    size_t vertex_count;

    std::ifstream in(ply_filepath, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
      if (line == "end_header") break;

      std::istringstream iss(line);
      std::string word;
      iss >> word;
      if (word == "property") {
        int size = 0;
        std::string type, property;
        iss >> type >> property;
        if (type == "float") {
          size = 4;
        }
        offsets[property] = offset;
        offset += size;
      } else if (word == "element") {
        std::string type;
        size_t count;
        iss >> type >> count;
        if (type == "vertex") {
          vertex_count = count;
        }
      }
    }

    std::vector<char> buffer(offset * vertex_count);
    in.read(buffer.data(), offset * vertex_count);

    splats.size_ = vertex_count;
    splats.position_.resize(vertex_count * 3);
    splats.sh0_.resize(vertex_count * 3);
    splats.sh1_.resize(vertex_count * 9);
    splats.sh2_.resize(vertex_count * 15);
    splats.sh3_.resize(vertex_count * 21);
    splats.opacity_.resize(vertex_count);
    splats.scale_.resize(vertex_count * 3);
    splats.rot_.resize(vertex_count * 4);

    for (int i = 0; i < vertex_count; i++) {
      float x = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                          offsets.at("x"));
      float y = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                          offsets.at("y"));
      float z = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                          offsets.at("z"));
      float f_dc_0 = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("f_dc_0"));
      float f_dc_1 = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("f_dc_1"));
      float f_dc_2 = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                               offsets.at("f_dc_2"));
      float opacity = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                                offsets.at("opacity"));
      float sx = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                           offsets.at("scale_0"));
      float sy = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                           offsets.at("scale_1"));
      float sz = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                           offsets.at("scale_2"));
      float rw = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                           offsets.at("rot_0"));
      float rx = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                           offsets.at("rot_1"));
      float ry = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                           offsets.at("rot_2"));
      float rz = *reinterpret_cast<float*>(buffer.data() + i * offset +
                                           offsets.at("rot_3"));

      splats.position_[i * 3 + 0] = x;
      splats.position_[i * 3 + 1] = y;
      splats.position_[i * 3 + 2] = z;

      splats.sh0_[i * 3 + 0] = f_dc_0;
      splats.sh0_[i * 3 + 1] = f_dc_1;
      splats.sh0_[i * 3 + 2] = f_dc_2;

      for (int j = 0; j < 9; j++) {
        splats.sh1_[i * 9 + j] = *reinterpret_cast<float*>(
            buffer.data() + i * offset +
            offsets.at("f_rest_" + std::to_string(j)));
      }

      for (int j = 0; j < 15; j++) {
        splats.sh2_[i * 15 + j] = *reinterpret_cast<float*>(
            buffer.data() + i * offset +
            offsets.at("f_rest_" + std::to_string(9 + j)));
      }

      for (int j = 0; j < 21; j++) {
        splats.sh3_[i * 21 + j] = *reinterpret_cast<float*>(
            buffer.data() + i * offset +
            offsets.at("f_rest_" + std::to_string(24 + j)));
      }

      splats.opacity_[i] = sigmoid(opacity);
      splats.scale_[i * 3 + 0] = std::exp(sx);
      splats.scale_[i * 3 + 1] = std::exp(sy);
      splats.scale_[i * 3 + 2] = std::exp(sz);
      float r = std::sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
      splats.rot_[i * 4 + 0] = rw / r;
      splats.rot_[i * 4 + 1] = rx / r;
      splats.rot_[i * 4 + 2] = ry / r;
      splats.rot_[i * 4 + 3] = rz / r;
    }
  }

  return splats;
}

Splats::Splats() = default;

Splats::~Splats() = default;

}  // namespace pygs
