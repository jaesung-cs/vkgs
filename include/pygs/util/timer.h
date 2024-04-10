#ifndef PYGS_UTIL_TIMER_H
#define PYGS_UTIL_TIMER_H

#include <chrono>
#include <iostream>

namespace pygs {

class Timer {
 public:
  Timer() = delete;

  Timer(const std::string& tag) : tag_(tag) {
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~Timer() {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = (now - start_).count();
    auto s = static_cast<double>(ns) / 1e9;
    std::cout << tag_ << " " << s << "s" << std::endl;
  }

 private:
  std::string tag_;
  std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace pygs

#endif  // PYGS_UTIL_TIMER_H