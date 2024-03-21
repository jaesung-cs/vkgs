#ifndef PYGS_SCENE_SPLATS_H
#define PYGS_SCENE_SPLATS_H

#include <string>
#include <vector>

namespace pygs {

class Splats {
 public:
  static Splats load(const std::string& ply_filepath);

 public:
  Splats();
  ~Splats();

  auto size() const noexcept { return size_; }
  auto positions() const noexcept { return position_; }
  auto colors() const noexcept { return color_; }
  auto scales() const noexcept { return scale_; }
  auto rots() const noexcept { return rot_; }

 private:
  size_t size_ = 0;
  std::vector<float> position_;  // (N, 3), x, y, z
  std::vector<float> color_;     // (N, 4), f_dc_0,1,2, opacity
  std::vector<float> scale_;     // (N, 3), scale_0,1,2
  std::vector<float> rot_;       // (N, 4), rot_0,1,2,3, in wxyz order
};

}  // namespace pygs

#endif  // PYGS_SCENE_SPLATS_H
