#include <pybind11/pybind11.h>

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>

#include <pygs/engine/engine.h>

namespace {

std::thread thread;
std::atomic_bool terminated = false;

std::mutex mutex;
std::unique_ptr<pygs::Engine> engine;

void Show() {
  if (thread.joinable()) {
    if (terminated) {
      thread.join();
      terminated = false;
    } else {
      std::cout << "[pygs] viewer is already running" << std::endl;
      return;
    }
  }

  thread = std::thread([] {
    if (engine == nullptr) {
      std::unique_lock<std::mutex> guard{mutex};
      engine = std::make_unique<pygs::Engine>();
    }
    engine->Run();
    std::cout << "[pygs] bye" << std::endl;
    terminated = true;
  });
}

void Close() {
  {
    std::unique_lock<std::mutex> guard{mutex};
    if (engine) {
      engine->Close();
    }
  }
}

void CleanupCallback() {
  std::cout << "[pygs] cleanup" << std::endl;

  if (engine) {
    engine->Close();
  }

  if (thread.joinable()) {
    thread.join();
    engine = nullptr;
  }
}

}  // namespace

namespace py = pybind11;

PYBIND11_MODULE(_pygs_cpp, m) {
  m.def("show", &Show);
  m.def("close", &Close);

  m.add_object("_cleanup", py::capsule(CleanupCallback));
}
