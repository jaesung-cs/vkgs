#include <iostream>

#include <pygs/window/window.h>
#include <pygs/engine/engine.h>
#include <pygs/scene/camera.h>

int main() {
  try {
    pygs::Window window;
    pygs::Engine engine;
    pygs::Camera camera;

    float cursor_x = 0.f;
    float cursor_y = 0.f;
    bool clicked[2] = {false, false};  // left, right

    while (!window.ShouldClose()) {
      const auto& events = window.PollEvents();
      for (const auto& event : events) {
        if (event.type == pygs::EventType::MOUSE_CLICK) {
          switch (event.mouse_click.button) {
            case pygs::MouseButton::LEFT:
              clicked[0] = event.mouse_click.pressed;
              break;

            case pygs::MouseButton::RIGHT:
              clicked[1] = event.mouse_click.pressed;
              break;
          }
        }

        else if (event.type == pygs::EventType::MOUSE_MOVE) {
          float dx = event.mouse_move.x - cursor_x;
          float dy = event.mouse_move.y - cursor_y;

          if (clicked[0] && !clicked[1]) {
            // Left drag to rotate
            camera.Rotate(dx, dy);
          } else if (!clicked[0] && clicked[1]) {
            // Right drag to translate
            camera.Translate(dx, dy);
          } else if (clicked[0] && clicked[1]) {
            // LR drag to zoom
            camera.Zoom(dy);
          }

          cursor_x = event.mouse_move.x;
          cursor_y = event.mouse_move.y;
        }
      }

      auto window_size = window.size();
      camera.SetWindowSize(window_size.width, window_size.height);

      engine.Draw(window, camera);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
