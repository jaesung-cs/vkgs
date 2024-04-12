#ifndef PYGS_UTIL_CLOCK_H
#define PYGS_UTIL_CLOCK_H

#include <chrono>

namespace pygs {

class Clock {
 public:
  static uint64_t timestamp() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }
};

}  // namespace pygs

#endif  // PYGS_UTIL_CLOCK_H
