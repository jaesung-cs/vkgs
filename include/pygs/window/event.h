#ifndef PYGS_WINDOW_EVENT_H
#define PYGS_WINDOW_EVENT_H

namespace pygs {

enum class EventType {
  MOUSE_MOVE,
  MOUSE_CLICK,
};

enum class MouseButton {
  LEFT,
  RIGHT,
};

struct MouseMoveEvent {
  double x;
  double y;
};

struct MouseClickEvent {
  MouseButton button;
  bool pressed;
};

struct Event {
  EventType type;
  union {
    MouseMoveEvent mouse_move;
    MouseClickEvent mouse_click;
  };
};

}  // namespace pygs

#endif  // PYGS_WINDOW_EVENT_H
