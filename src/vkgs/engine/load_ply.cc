#include "vkgs/engine/load_ply.h"

#include <sstream>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cstring>

namespace vkgs {

PlyBuffer LoadPly(std::string plyFilepath) {
  std::ifstream in(plyFilepath, std::ios::binary);

  // parse header
  std::unordered_map<std::string, int> offsets;
  int offset = 0;
  size_t pointCount = 0;
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
        pointCount = count;
      }
    }
  }

  // ply offsets
  PlyBuffer result;
  result.pointCount = pointCount;
  result.plyOffsets.resize(60);
  result.plyOffsets[0] = offsets["x"] / 4;
  result.plyOffsets[1] = offsets["y"] / 4;
  result.plyOffsets[2] = offsets["z"] / 4;
  result.plyOffsets[3] = offsets["scale_0"] / 4;
  result.plyOffsets[4] = offsets["scale_1"] / 4;
  result.plyOffsets[5] = offsets["scale_2"] / 4;
  result.plyOffsets[6] = offsets["rot_1"] / 4;  // qx
  result.plyOffsets[7] = offsets["rot_2"] / 4;  // qy
  result.plyOffsets[8] = offsets["rot_3"] / 4;  // qz
  result.plyOffsets[9] = offsets["rot_0"] / 4;  // qw
  result.plyOffsets[10 + 0] = offsets["f_dc_0"] / 4;
  result.plyOffsets[10 + 16] = offsets["f_dc_1"] / 4;
  result.plyOffsets[10 + 32] = offsets["f_dc_2"] / 4;
  for (int i = 0; i < 15; ++i) {
    result.plyOffsets[10 + 1 + i] = offsets["f_rest_" + std::to_string(i)] / 4;
    result.plyOffsets[10 + 17 + i] = offsets["f_rest_" + std::to_string(15 + i)] / 4;
    result.plyOffsets[10 + 33 + i] = offsets["f_rest_" + std::to_string(30 + i)] / 4;
  }
  result.plyOffsets[58] = offsets["opacity"] / 4;
  result.plyOffsets[59] = offset / 4;

  // read all binary data
  result.buffer.resize(offset * pointCount);

  in.read(reinterpret_cast<char*>(result.buffer.data()), offset * pointCount);
  /*
  constexpr uint32_t chunkSize = 65536;
  uint32_t loadedPointCount = 0;
  for (uint32_t start = 0; start < pointCount; start += chunkSize) {
    auto chunkPointCount = std::min<uint32_t>(chunkSize, pointCount - start);
    in.read(reinterpret_cast<char*>(result.buffer.data()) + offset * start, offset * chunkPointCount);
    loadedPointCount = start + chunkPointCount;
  }
  */

  return result;
}

}  // namespace vkgs
