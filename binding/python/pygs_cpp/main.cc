#include <pybind11/pybind11.h>

#include <iostream>
#include <thread>
#include <atomic>

#include <vkgs/engine/engine.h>

namespace {

std::thread thread;
std::atomic_bool terminated = false;

std::unique_ptr<vkgs::Engine> engine;

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

  if (engine == nullptr) {
    engine = std::make_unique<vkgs::Engine>();
  }

  thread = std::thread([] {
    engine->Run();
    std::cout << "[pygs] bye" << std::endl;
    terminated = true;
  });
}

void Load(const std::string& ply_filepath) {
  if (engine) {
    engine->LoadSplatsAsync(ply_filepath);
  }
}

void Close() {
  if (engine) {
    engine->Close();
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
  m.def("load", &Load);
  m.def("close", &Close);

  m.add_object("_cleanup", py::capsule(CleanupCallback));
}
