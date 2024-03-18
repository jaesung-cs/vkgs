#include <iostream>

#include <pygs/window/window.h>
#include <pygs/engine/engine.h>

int main() {
  try {
    pygs::Window window;
    pygs::Engine engine;

    while (!window.ShouldClose()) {
      const auto& events = window.PollEvents();
      for (const auto& event : events) {
        /*
        if (event.type == pygs::EventType::MOUSE_MOVE) {
          std::cout << "mouse move (" << event.mouse_move.x << ", "
                    << event.mouse_move.y << ")" << std::endl;
        }
        */
      }

      engine.Draw(window);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
