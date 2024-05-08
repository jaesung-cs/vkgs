#ifndef VKGS_UTIL_TIMER_H
#define VKGS_UTIL_TIMER_H

#include <chrono>
#include <iostream>

#include <vkgs/util/clock.h>

namespace vkgs {

class Timer {
 public:
  Timer() = delete;

  Timer(const std::string& tag) : tag_(tag) { start_ = Clock::timestamp(); }

  ~Timer() {
    auto now = Clock::timestamp();
    auto ns = now - start_;
    auto ms = static_cast<double>(ns) / 1e6;
    std::cout << tag_ << " " << ms << "ms" << std::endl;
  }

 private:
  std::string tag_;
  int64_t start_ = 0;
};

}  // namespace vkgs

#endif  // VKGS_UTIL_TIMER_H
