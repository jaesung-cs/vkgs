#ifndef VKGS_ENGINE_LOAD_PLY_H
#define VKGS_ENGINE_LOAD_PLY_H

#include <string>
#include <vector>

namespace vkgs {

struct PlyBuffer {
  uint32_t pointCount;
  std::vector<uint32_t> plyOffsets;
  std::vector<uint8_t> buffer;
};

PlyBuffer LoadPly(std::string plyFilepath);

}  // namespace vkgs

#endif  // VKGS_ENGINE_LOAD_PLY_H
