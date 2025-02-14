#include <vector>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cstring>

namespace vkgs {

void parse_ply(std::string ply_filepath) {
  std::ifstream in(ply_filepath, std::ios::binary);

  // parse header
  std::unordered_map<std::string, int> offsets;
  int offset = 0;
  size_t point_count = 0;
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
        point_count = count;
      }
    }
  }

  // ply offsets
  std::vector<uint32_t> ply_offsets(60);
  ply_offsets[0] = offsets["x"] / 4;
  ply_offsets[1] = offsets["y"] / 4;
  ply_offsets[2] = offsets["z"] / 4;
  ply_offsets[3] = offsets["scale_0"] / 4;
  ply_offsets[4] = offsets["scale_1"] / 4;
  ply_offsets[5] = offsets["scale_2"] / 4;
  ply_offsets[6] = offsets["rot_1"] / 4;  // qx
  ply_offsets[7] = offsets["rot_2"] / 4;  // qy
  ply_offsets[8] = offsets["rot_3"] / 4;  // qz
  ply_offsets[9] = offsets["rot_0"] / 4;  // qw
  ply_offsets[10 + 0] = offsets["f_dc_0"] / 4;
  ply_offsets[10 + 16] = offsets["f_dc_1"] / 4;
  ply_offsets[10 + 32] = offsets["f_dc_2"] / 4;
  for (int i = 0; i < 15; ++i) {
    ply_offsets[10 + 1 + i] = offsets["f_rest_" + std::to_string(i)] / 4;
    ply_offsets[10 + 17 + i] = offsets["f_rest_" + std::to_string(15 + i)] / 4;
    ply_offsets[10 + 33 + i] = offsets["f_rest_" + std::to_string(30 + i)] / 4;
  }
  ply_offsets[58] = offsets["opacity"] / 4;
  ply_offsets[59] = offset / 4;

  // read all binary data
  std::vector<uint8_t> buffer(offset * point_count);

  constexpr uint32_t chunk_size = 65536;
  uint32_t loaded_point_count = 0;
  for (uint32_t start = 0; start < point_count; start += chunk_size) {
    auto chunk_point_count = std::min<uint32_t>(chunk_size, point_count - start);
    in.read(reinterpret_cast<char*>(buffer.data()) + offset * start, offset * chunk_point_count);
    loaded_point_count = start + chunk_point_count;
  }

  // TODO
}

}  // namespace vkgs
