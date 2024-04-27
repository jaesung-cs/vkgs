#include <pybind11/pybind11.h>

#include <iostream>
#include <thread>
#include <atomic>

#include <pygs/engine/engine.h>

namespace {

std::thread thread;
std::atomic_bool terminated = false;

void start() {
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
    pygs::Engine engine;
    engine.Run();
    std::cout << "[pygs] bye" << std::endl;
    terminated = true;
  });
}

void cleanup_callback() {
  if (thread.joinable()) {
    // TODO: send termination to thread
    std::cout << "[pygs] cleanup" << std::endl;
    thread.join();
  }
}

}  // namespace

namespace py = pybind11;

PYBIND11_MODULE(_pygs_cpp, m) {
  m.def("start", &start, py::call_guard<py::gil_scoped_release>());

  m.add_object("_cleanup", py::capsule(cleanup_callback));
}
