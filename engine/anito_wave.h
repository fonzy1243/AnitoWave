#ifndef ANITO_WAVE_H
#define ANITO_WAVE_H

#pragma once

#include "types/vk_types.h"

#include <deque>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vma/vk_mem_alloc.hpp>

#include "camera/camera.h"
#include "descriptors/vk_descriptors.h"
#include "loader/vk_loader.h"
#include "pipelines/vk_pipelines.h"

struct MeshAsset;
namespace fastgltf {
    struct Mesh;
}

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function) {
        deletors.push_back(function);
    }

    void flush() {
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)();
        }

        deletors.clear();
    }
};

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect {
    const char* name;

    vk::Pipeline pipeline;
    vk::PipelineLayout layout;

    ComputePushConstants data;
};

struct RenderObject {
    uint32_t indexCount;
    uint32_t firstIndex;
    vk::Buffer indexBuffer;

    MaterialInstance* material;
    Bounds bounds;
    glm::mat4 transform;
    vk::DeviceAddress vertexBufferAddress;
};

struct FrameData {
    vk::Semaphore _swapchainSemaphore, _renderSemaphore;
    vk::Fence _renderFence;

    DescriptorAllocatorGrowable _frameDescriptors;
    DeletionQueue _deletionQueue;

    vk::CommandPool _commandPool;
    vk::CommandBuffer _mainCommandBuffer;
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct DrawContext {
    std::vector<RenderObject> OpaqueSurfaces;
    std::vector<RenderObject> TransparentSurfaces;
};

struct EngineStats {
    float frametime;
    int triangle_count;
    int drawcall_count;
    float mesh_draw_time;
};

struct GLTFMetallic_Roughness {
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    vk::DescriptorSetLayout materialLayout;

    struct MaterialConstants {
        glm::vec4 colorFactors;
        glm::vec4 metal_rough_factors;
        glm::vec4 extra[14];
    };

    struct MaterialResources {
        AllocatedImage colorImage;
        vk::Sampler colorSampler;
        AllocatedImage metalRoughImage;
        vk::Sampler metalRoughSampler;
        vk::Buffer dataBuffer;
        uint32_t dataBufferOffset;
    };

    DescriptorWriter writer;

    void build_pipelines(AnitoWave* engine);
    void clear_resources(vk::Device device);

    MaterialInstance write_material(vk::Device device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};

struct MeshNode : public Node {
    std::shared_ptr<MeshAsset> mesh;

    virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

class AnitoWave {
public:
    bool _isInitialized{false};
    int _frameNumber{0};

    vk::Extent2D _windowExtent {1600, 900};

    struct SDL_Window* _window {nullptr};

    vk::Instance _instance;
    vk::DebugUtilsMessengerEXT _debug_messenger;
    vk::PhysicalDevice _chosenGPU;
    vk::Device _device;

    vk::Queue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    AllocatedBuffer _defaultGLTFMaterialData;

    FrameData _frames[FRAME_OVERLAP];

    vk::SurfaceKHR _surface;
    vk::SwapchainKHR _swapchain;
    vk::Format _swapchainImageFormat;
    vk::Extent2D _swapchainExtent;
    vk::Extent2D _drawExtent;

    vk::DescriptorPool _descriptorPool;

    DescriptorAllocator globalDescriptorAllocator;

    vk::Pipeline _gradientPipeline;
    vk::PipelineLayout _gradientPipelineLayout;

    std::vector<vk::Image> _swapchainImages;
    std::vector<vk::ImageView> _swapchainImageViews;

    vk::DescriptorSet _drawImageDescriptors;
    vk::DescriptorSetLayout _drawImageDescriptorLayout;

    DeletionQueue _mainDeletionQueue;

    vma::Allocator _allocator;

    vk::DescriptorSetLayout _gpuSceneDataDescriptorLayout;

    GLTFMetallic_Roughness metalRoughMaterial;

    // draw resources
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;

    // immediate submit structures
    vk::Fence _immFence;
    vk::CommandBuffer _immCommandBuffer;
    vk::CommandPool _immCommandPool;

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    vk::Sampler _defaultSamplerLinear;
    vk::Sampler _defaultSamplerNearest;

    GPUMeshBuffers rectangle;
    DrawContext drawCommands;

    GPUSceneData sceneData;

    Camera mainCamera;

    EngineStats stats;

    std::vector<ComputeEffect> backgroundEffects;
    int currentBackgroundEffect{0};

    // Singleton getter
    static AnitoWave& Get();

    // Initialize engine
    void init();

    // Shutdown engine
    void cleanup();

    // Main draw loop
    void draw();
    void draw_main(vk::CommandBuffer cmd);
    void draw_imgui(vk::CommandBuffer cmd, vk::ImageView targetImageView);

    void render_nodes();

    void draw_geometry(vk::CommandBuffer cmd);

    // Run main loop
    void run();

    void update_scene();

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    FrameData& get_current_frame();
    FrameData& get_last_frame();

    AllocatedBuffer create_buffer(size_t allocSize, vk::BufferUsageFlags usage, vma::MemoryUsage memoryUsage);

    AllocatedImage create_image(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped = false);

    void immediate_submit(std::function<void(vk::CommandBuffer cmd)>&& function);

    std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;
    std::vector<std::shared_ptr<LoadedGLTF>> brickadiaScene;

    void destroy_image(const AllocatedImage& img);
    void destroy_buffer(const AllocatedBuffer& buffer);

    float renderScale = 1;

    bool resize_requested{false};
    bool freeze_rendering{false};
private:
    void init_vulkan();

    void init_swapchain();

    void create_swapchain(uint32_t width, uint32_t height);

    void resize_swapchain();

    void destroy_swapchain();

    void init_commands();

    void init_pipelines();
    void init_background_pipelines();

    void init_descriptors();

    void init_sync_structures();

    void init_renderables();

    void init_imgui();

    void init_default_data();
};

#endif //ANITO_WAVE_H
