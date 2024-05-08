#ifndef VKGS_UTIL_CLOCK_H
#define VKGS_UTIL_CLOCK_H

#include <chrono>

namespace vkgs {

class Clock {
 public:
  static uint64_t timestamp() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }
};

}  // namespace vkgs

#endif  // VKGS_UTIL_CLOCK_H
