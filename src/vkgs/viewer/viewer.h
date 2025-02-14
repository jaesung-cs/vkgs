#ifndef VKGS_VIEWER_VIEWER_H
#define VKGS_VIEWER_VIEWER_H

#include <memory>
#include <vector>
#include <string>

#include <vulkan/vulkan.h>

namespace vkgs {
namespace viewer {

struct UiCreateInfo {
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  uint32_t queueFamily;
  VkQueue queue;
  VkPipelineCache pipelineCache;
  VkDescriptorPool descriptorPool;
  VkRenderPass renderPass;
  uint32_t subpass;
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

  void PrepareWindow(VkInstance instance);
  void DestroyWindow();
  void PrepareUi(const UiCreateInfo& createInfo);
  void SetWindowed();
  void SetWindowedFullscreen();
  void SetWindowSize(int width, int height);
  void PollEvents();
  std::vector<std::string> ConsumeDroppedFilepaths();
  bool ShouldClose() const;
  void BeginUi();
  void EndUi();
  void DrawUi(VkCommandBuffer commandBuffer);

  VkSurfaceKHR surface() const;
  WindowSize windowSize() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_VIEWER_H
