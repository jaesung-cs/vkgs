#include <vkgs/engine/engine.h>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <mutex>
#include <vector>
#include <map>

#include <vulkan/vulkan.h>

#include "imgui.h"

#include "vkgs/viewer/viewer.h"

#include <vkgs/scene/camera.h>
#include <vkgs/util/clock.h>

namespace vkgs {
namespace {

struct Resolution {
  int width;
  int height;
  const char* tag;
};

std::vector<Resolution> preset_resolutions = {
    // clang-format off
    {640, 480, "640 x 480 (480p)"},
    {800, 600, "800 x 600"},
    {1280, 720, "1280 x 720 (720p, HD)"},
    {1600, 900, "1600 x 900"},
    {1920, 1080, "1920 x 1080 (1080p, FHD)"},
    {2560, 1440, "2560 x 1440 (1440p, QHD)"},
    // clang-format on
};

}  // namespace

class Engine::Impl {
 private:
  enum class DisplayMode {
    Windowed,
    WindowedFullscreen,
  };

 public:
  Impl() {
    // TODO
  }

  ~Impl() {
    // TODO
  }

  void Run() {
    // TODO: main loop
  }

 private:
  void Draw() {
    int32_t frame_index = frame_counter_ % 2;

    // draw ui
    viewer_.BeginUi();
    const auto& io = ImGui::GetIO();

    // handle events
    if (!io.WantCaptureMouse) {
      bool left = io.MouseDown[ImGuiMouseButton_Left];
      bool right = io.MouseDown[ImGuiMouseButton_Right];
      float dx = io.MouseDelta.x;
      float dy = io.MouseDelta.y;

      if (left && !right) {
        camera_.Rotate(dx, dy);
      } else if (!left && right) {
        camera_.Translate(dx, dy);
      } else if (left && right) {
        camera_.Zoom(dy);
      }

      if (io.MouseWheel != 0.f) {
        if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) {
          camera_.DollyZoom(io.MouseWheel);
        } else {
          camera_.Zoom(io.MouseWheel * 10.f);
        }
      }
    }

    if (!io.WantCaptureKeyboard) {
      constexpr float speed = 1000.f;
      float dt = io.DeltaTime;
      if (ImGui::IsKeyDown(ImGuiKey_W)) {
        camera_.Translate(0.f, 0.f, speed * dt);
      }
      if (ImGui::IsKeyDown(ImGuiKey_S)) {
        camera_.Translate(0.f, 0.f, -speed * dt);
      }
      if (ImGui::IsKeyDown(ImGuiKey_A)) {
        camera_.Translate(speed * dt, 0.f);
      }
      if (ImGui::IsKeyDown(ImGuiKey_D)) {
        camera_.Translate(-speed * dt, 0.f);
      }
      if (ImGui::IsKeyDown(ImGuiKey_Space)) {
        camera_.Translate(0.f, speed * dt);
      }

      if (ImGui::IsKeyDown(ImGuiKey::ImGuiMod_Alt) && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        ToggleDisplayMode();
      }
    }
  }

  void ToggleDisplayMode() {
    switch (display_mode_) {
      case DisplayMode::Windowed:
        SetWindowedFullscreen();
        break;
      case DisplayMode::WindowedFullscreen:
        SetWindowed();
        break;
    }
  }

  void SetWindowed() {
    if (display_mode_ == DisplayMode::WindowedFullscreen) {
      display_mode_ = DisplayMode::Windowed;
      viewer_.SetWindowed();
    }
  }

  void SetWindowedFullscreen() {
    if (display_mode_ == DisplayMode::Windowed) {
      display_mode_ = DisplayMode::WindowedFullscreen;
      viewer_.SetWindowedFullscreen();
    }
  }

  DisplayMode display_mode_ = DisplayMode::Windowed;

  viewer::Viewer viewer_;

  Camera camera_;

  std::vector<VkCommandBuffer> draw_command_buffers_;
  std::vector<VkSemaphore> image_acquired_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> render_finished_fences_;

  uint64_t frame_counter_ = 0;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() = default;

void Engine::Run() { impl_->Run(); }

}  // namespace vkgs
