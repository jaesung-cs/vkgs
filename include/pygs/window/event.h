#ifndef PYGS_WINDOW_EVENT_H
#define PYGS_WINDOW_EVENT_H

namespace pygs {

enum class EventType {
  MOUSE_MOVE,
};

struct MouseMoveEvent {
  double x;
  double y;
};

struct Event {
  EventType type;
  union {
    MouseMoveEvent mouse_move;
  };
};

}  // namespace pygs

#endif  // PYGS_WINDOW_EVENT_H
