#include "vkgs/viewer/viewer.h"

#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

namespace {

void check_vk_result(VkResult err) {
  if (err == 0) return;
  std::cerr << "[imgui vulkan] Error: VkResult = " << err << std::endl;
  if (err < 0) abort();
}

}  // namespace

namespace vkgs {
namespace viewer {

class Viewer::Impl {
 private:
  enum class DisplayMode {
    Windowed,
    WindowedFullscreen,
  };

 public:
  static void DropCallback(GLFWwindow* window, int count, const char** paths) {
    std::vector<std::string> filepaths(paths, paths + count);
    auto* impl = reinterpret_cast<Impl*>(glfwGetWindowUserPointer(window));
    impl->SetDroppedFilepaths(filepaths);
  }

 public:
  Impl() {
    if (glfwInit() == GLFW_FALSE) throw std::runtime_error("Failed to initialize glfw.");

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
  }

  ~Impl() {
    ImGui::DestroyContext();

    glfwTerminate();
  }

  VkSurfaceKHR surface() const noexcept { return surface_; }

  WindowSize window_size() const {
    WindowSize result;
    glfwGetFramebufferSize(window_, &result.width, &result.height);
    return result;
  }

  void CreateWindow(const WindowCreateInfo& create_info) {
    // create window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window_ = glfwCreateWindow(1600, 900, "vkgs", NULL, NULL);

    // Vulkan surface
    glfwCreateWindowSurface(create_info.instance, window_, NULL, &surface_);

    // file drop callback
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, DropCallback);

    ImGui_ImplGlfw_InitForVulkan(window_, true);
    auto init_info = GetImguiInitInfo(create_info);
    ImGui_ImplVulkan_Init(&init_info);
  }

  void DestroyWindow() {
    glfwDestroyWindow(window_);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
  }

  void RecreateUi(const WindowCreateInfo& create_info) {
    ImGui_ImplVulkan_Shutdown();
    auto init_info = GetImguiInitInfo(create_info);
    ImGui_ImplVulkan_Init(&init_info);
  }

  void Show() { glfwShowWindow(window_); }

  void SetWindowed() {
    if (display_mode == DisplayMode::WindowedFullscreen) {
      glfwSetWindowMonitor(window_, NULL, xpos_, ypos_, width_, height_, 0);
      display_mode = DisplayMode::Windowed;
    }
  }

  void SetWindowedFullscreen() {
    if (display_mode == DisplayMode::Windowed) {
      glfwGetWindowPos(window_, &xpos_, &ypos_);
      glfwGetWindowSize(window_, &width_, &height_);

      // TODO: multi monitor
      GLFWmonitor* primary = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(primary);
      glfwSetWindowMonitor(window_, primary, 0, 0, mode->width, mode->height, mode->refreshRate);
      display_mode = DisplayMode::WindowedFullscreen;
    }
  }

  void SetWindowSize(int width, int height) {
    if (glfwGetWindowAttrib(window_, GLFW_MAXIMIZED)) {
      glfwRestoreWindow(window_);
    }

    glfwSetWindowSize(window_, width, height);
  }

  void PollEvents() { glfwPollEvents(); }

  bool ShouldClose() const { return glfwWindowShouldClose(window_); }

  void BeginUi() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
  }

  void EndUi() { ImGui::Render(); }

  void DrawUi(VkCommandBuffer command_buffer) {
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
  }

  std::vector<std::string> ConsumeDroppedFilepaths() { return std::move(dropped_filepaths_); }

 private:
  ImGui_ImplVulkan_InitInfo GetImguiInitInfo(const WindowCreateInfo& window_info) {
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = window_info.instance;
    init_info.PhysicalDevice = window_info.physical_device;
    init_info.Device = window_info.device;
    init_info.QueueFamily = window_info.queue_family;
    init_info.Queue = window_info.queue;
    init_info.PipelineCache = window_info.pipeline_cache;
    init_info.DescriptorPool = window_info.descriptor_pool;
    init_info.Subpass = 0;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.RenderPass = window_info.render_pass;
    init_info.MSAASamples = window_info.samples;
    init_info.Allocator = VK_NULL_HANDLE;
    init_info.CheckVkResultFn = check_vk_result;
    return init_info;
  }

  void SetDroppedFilepaths(const std::vector<std::string>& filepaths) { dropped_filepaths_ = filepaths; }

  std::vector<std::string> dropped_filepaths_;

  GLFWwindow* window_ = nullptr;
  int xpos_ = 0;
  int ypos_ = 0;
  int width_ = 0;
  int height_ = 0;
  DisplayMode display_mode = DisplayMode::Windowed;

  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
};

Viewer::Viewer() : impl_(std::make_shared<Impl>()) {}

Viewer::~Viewer() = default;

void Viewer::CreateWindow(const WindowCreateInfo& create_info) { impl_->CreateWindow(create_info); }

void Viewer::DestroyWindow() { impl_->DestroyWindow(); }

void Viewer::RecreateUi(const WindowCreateInfo& create_info) { impl_->RecreateUi(create_info); }

void Viewer::Show() { impl_->Show(); }

void Viewer::SetWindowed() { impl_->SetWindowed(); }

void Viewer::SetWindowedFullscreen() { impl_->SetWindowedFullscreen(); }

void Viewer::SetWindowSize(int width, int height) { impl_->SetWindowSize(width, height); }

void Viewer::PollEvents() { impl_->PollEvents(); }

std::vector<std::string> Viewer::ConsumeDroppedFilepaths() { return impl_->ConsumeDroppedFilepaths(); }

bool Viewer::ShouldClose() const { return impl_->ShouldClose(); }

void Viewer::BeginUi() { impl_->BeginUi(); }

void Viewer::EndUi() { impl_->EndUi(); }

void Viewer::DrawUi(VkCommandBuffer command_buffer) { impl_->DrawUi(command_buffer); }

VkSurfaceKHR Viewer::surface() const { return impl_->surface(); }

WindowSize Viewer::window_size() const { return impl_->window_size(); }

}  // namespace viewer
}  // namespace vkgs
