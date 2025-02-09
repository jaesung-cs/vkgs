#ifndef VKGS_VIEWER_VIEWER_H
#define VKGS_VIEWER_VIEWER_H

#include <memory>
#include <vector>
#include <string>

#include <vulkan/vulkan.h>

namespace vkgs {
namespace viewer {

struct WindowCreateInfo {
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  uint32_t queue_family;
  VkQueue queue;
  VkPipelineCache pipeline_cache;
  VkDescriptorPool descriptor_pool;
  VkRenderPass render_pass;
  VkSampleCountFlagBits samples;
};

struct WindowSize {
  int width;
  int height;
};

class Viewer {
 public:
  Viewer();
  ~Viewer();

  void CreateWindow(const WindowCreateInfo& create_info);
  void DestroyWindow();
  void RecreateUi(const WindowCreateInfo& create_info);
  void Show();
  void PollEvents();
  std::vector<std::string> ConsumeDroppedFilepaths();
  bool ShouldClose() const;
  void NewUiFrame();
  void DrawUi(VkCommandBuffer command_buffer);

  VkSurfaceKHR surface() const;
  WindowSize window_size() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_VIEWER_H
