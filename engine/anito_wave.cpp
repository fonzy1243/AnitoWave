#include "anito_wave.h"

#include "images/vk_images.h"
#include "loader/vk_loader.h"
#include "descriptors/vk_descriptors.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <glm/gtx/transform.hpp>

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>
#include <vma/vk_mem_alloc.hpp>

#include "VkBootstrap.h"
#include "initializers/vk_initializers.h"

constexpr bool bUseValidationLayers = true;

AnitoWave* loadedEngine = nullptr;

AnitoWave &AnitoWave::Get() {
    return *loadedEngine;
}

void AnitoWave::init() {
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow("AnitoWave", _windowExtent.width, _windowExtent.height, window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    init_default_data();

    init_renderables();

    init_imgui();

    _isInitialized = true;

    mainCamera.velocity = glm::vec3(0.0f);
    mainCamera.position = glm::vec3(0.0f);

    mainCamera.pitch = 0;
    mainCamera.yaw = 0;
}

void AnitoWave::init_default_data() {
    std::array<Vertex, 4> rect_vertices;

    rect_vertices[0].position = { 0.5,-0.5, 0 };
    rect_vertices[1].position = { 0.5,0.5, 0 };
    rect_vertices[2].position = { -0.5,-0.5, 0 };
    rect_vertices[3].position = { -0.5,0.5, 0 };

    rect_vertices[0].color = { 0,0, 0,1 };
    rect_vertices[1].color = { 0.5,0.5,0.5 ,1 };
    rect_vertices[2].color = { 1,0, 0,1 };
    rect_vertices[3].color = { 0,1, 0,1 };

    rect_vertices[0].uv_x = 1;
    rect_vertices[0].uv_y = 0;
    rect_vertices[1].uv_x = 0;
    rect_vertices[1].uv_y = 0;
    rect_vertices[2].uv_x = 1;
    rect_vertices[2].uv_y = 1;
    rect_vertices[3].uv_x = 0;
    rect_vertices[3].uv_y = 1;

    std::array<uint32_t, 6> rect_indices;

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    rectangle = uploadMesh(rect_indices, rect_vertices);

    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _whiteImage = create_image((void*)&white, vk::Extent3D{1, 1, 1}, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eSampled);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _greyImage = create_image((void*)&grey, vk::Extent3D{1, 1, 1}, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eSampled);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    _blackImage = create_image((void*)&black, vk::Extent3D{1, 1, 1}, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eSampled);

    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }

    _errorCheckerboardImage = create_image(pixels.data(), vk::Extent3D{16, 16, 1}, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eSampled);

    vk::SamplerCreateInfo sampl = vk::SamplerCreateInfo()
    .setMagFilter(vk::Filter::eNearest)
    .setMinFilter(vk::Filter::eNearest);

    vk::Result createRes = _device.createSampler(&sampl, nullptr, &_defaultSamplerNearest);
    if (createRes != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create sampler");
    }

    sampl.setMagFilter(vk::Filter::eLinear)
    .setMinFilter(vk::Filter::eLinear);

    createRes = _device.createSampler(&sampl, nullptr, &_defaultSamplerLinear);
    if (createRes != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create sampler");
    }
}

void AnitoWave::cleanup() {
    if (_isInitialized) {
        _device.waitIdle();

        loadedScenes.clear();

        for (auto& frame : _frames) {
            frame._deletionQueue.flush();
        }

        _mainDeletionQueue.flush();

        destroy_swapchain();

        _instance.destroySurfaceKHR(_surface);

        _allocator.destroy();

        _device.destroy();
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        _instance.destroy();

        SDL_DestroyWindow(_window);
    }
}

void AnitoWave::init_background_pipelines() {
    vk::PushConstantRange pushConstant = vk::PushConstantRange()
    .setOffset(0)
    .setSize(sizeof(ComputePushConstants))
    .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    vk::PipelineLayoutCreateInfo computeLayout = vk::PipelineLayoutCreateInfo()
    .setPNext(nullptr)
    .setPSetLayouts(&_drawImageDescriptorLayout)
    .setSetLayoutCount(1)
    .setPPushConstantRanges(&pushConstant)
    .setPushConstantRangeCount(1);

    vk::Result createRes = _device.createPipelineLayout(&computeLayout, nullptr, &_gradientPipelineLayout);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Failed to create pipeline layout.\n");
    }

    vk::ShaderModule gradientShader;
    if (!vkutil::load_shader_module("../shaders/gradient_color.comp.spv", _device, &gradientShader)) {
        fmt::print("Error when building the compute shader.\n");
    }

    vk::ShaderModule skyShader;
    if (!vkutil::load_shader_module("../shaders/sky.comp.spv", _device, &skyShader)) {
        fmt::print("Error when building the compute shader.\n");
    }

    vk::PipelineShaderStageCreateInfo stageinfo = vk::PipelineShaderStageCreateInfo()
    .setPNext(nullptr)
    .setStage(vk::ShaderStageFlagBits::eCompute)
    .setModule(gradientShader)
    .setPName("main");

    vk::ComputePipelineCreateInfo computePipelineCreateInfo = vk::ComputePipelineCreateInfo()
    .setPNext(nullptr)
    .setLayout(_gradientPipelineLayout)
    .setStage(stageinfo);

    ComputeEffect gradient;
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};

    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    createRes = _device.createComputePipelines(VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Error when building the compute pipeline.\n");
    }

    computePipelineCreateInfo.stage.setModule(skyShader);

    ComputeEffect sky;
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    createRes = _device.createComputePipelines(VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Error when building the compute pipeline.\n");
    }

    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    _device.destroyShaderModule(gradientShader);
    _device.destroyShaderModule(skyShader);

    _mainDeletionQueue.push_function([&]() {
        _device.destroyPipelineLayout(_gradientPipelineLayout);
        _device.destroyPipeline(sky.pipeline);
        _device.destroyPipeline(gradient.pipeline);
    });
}

void AnitoWave::draw_main(vk::CommandBuffer cmd) {
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    // bind the background compute pipeline
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, effect.pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors, 0, nullptr);

    cmd.pushConstants(_gradientPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ComputePushConstants), &effect.data);
    // Execute the compute pipeline dispatch. Divide due to the 16x16 workgroup size.
    cmd.dispatch(std::ceil(_windowExtent.width / 16.0), std::ceil(_windowExtent.height / 16.0), 1);

    // Draw the triangle
    vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eColorAttachmentOptimal);

    vk::RenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr, vk::ImageLayout::eColorAttachmentOptimal);
    vk::RenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, vk::ImageLayout::eDepthAttachmentOptimal);

    vk::RenderingInfo renderInfo = vkinit::rendering_info(_windowExtent, &colorAttachment, &depthAttachment);

    cmd.beginRendering(&renderInfo);
    auto start = std::chrono::system_clock::now();

    draw_geometry(cmd);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    stats.mesh_draw_time = elapsed.count() / 1000.f;

    cmd.endRendering();
}

void AnitoWave::draw_imgui(vk::CommandBuffer cmd, vk::ImageView targetImageView) {
    vk::RenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, vk::ImageLayout::eColorAttachmentOptimal);
    vk::RenderingInfo renderInfo = vkinit::rendering_info(_windowExtent, &colorAttachment, nullptr);

    cmd.beginRendering(&renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    cmd.endRendering();
}

void AnitoWave::draw() {
    vk::Result waitRes = _device.waitForFences(1, &get_current_frame()._renderFence, true, 1000000000);
    if (waitRes != vk::Result::eSuccess) {
        throw std::runtime_error("Error waiting for fences.");
    }

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);
    // Request image from the swapchain
    uint32_t swapchainImageIndex;

    vk::Result acquireRes = _device.acquireNextImageKHR(_swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (acquireRes == vk::Result::eErrorOutOfDateKHR) {
        resize_requested = true;
        return;
    }

    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;
    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;

    vk::Result resetRes = _device.resetFences(1, &get_current_frame()._renderFence);
    if (resetRes != vk::Result::eSuccess) {
        throw std::runtime_error("Error resetting fence.");
    }

    // Can safely reset command buffer as the commands are done executing.
    get_current_frame()._mainCommandBuffer.reset();

    vk::CommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // Begin recording command buffer.
    vk::CommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cmd.begin(cmdBeginInfo);

    // Transition main draw image to general layout for writing
    vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    vkutil::transition_image(cmd, _depthImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal);

    draw_main(cmd);

    // Transition the draw image and swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

    vk::Extent2D extent;
    extent.height = _windowExtent.height;
    extent.width = _windowExtent.width;

    // Copy draw image into swapchain
    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    // Set swapchain image layout to attachment optimal for drawing.
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal);

    // Draw imgui onto the swapchain image
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    // Set swapchain image layout to present for drawing.
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);

    // Finalize the command buffer
    cmd.end();

    // Prepare submission to the queue.
    // Wait on present semaphore to signal that swapchain is ready.
    // Signal render semaphore that rendering has finished.

    vk::CommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);

    vk::SemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(vk::PipelineStageFlagBits2::eColorAttachmentOutput, get_current_frame()._swapchainSemaphore);
    vk::SemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(vk::PipelineStageFlagBits2::eAllGraphics, get_current_frame()._renderSemaphore);

    vk::SubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);

    // Submit command buffer to queue and execute it.
    // _renderFence will block until the graphics commands finish executing.
    vk::Result submitRes = _graphicsQueue.submit2(1, &submit, get_current_frame()._renderFence);
    if (submitRes != vk::Result::eSuccess) {
        throw std::runtime_error("Error submitting command buffer.");
    }

    // Prepare to present.
    // Put the image that was just rendered onto the window.
    // Wait for the _renderSemaphore so that drawing commands have finished before displaying image.
    vk::PresentInfoKHR presentInfo = vkinit::present_info();

    presentInfo.setPSwapchains(&_swapchain);
    presentInfo.setSwapchainCount(1);

    presentInfo.setPWaitSemaphores(&get_current_frame()._renderSemaphore);
    presentInfo.setWaitSemaphoreCount(1);

    presentInfo.setPImageIndices(&swapchainImageIndex);

    vk::Result presentRes = _graphicsQueue.presentKHR(&presentInfo);
    if (presentRes == vk::Result::eErrorOutOfDateKHR) {
        resize_requested = true;
        return;
    }

    // Increase the number of frames drawn
    _frameNumber++;
}

bool is_visible(const RenderObject& obj, const glm::mat4& viewproj) {
    std::array corners {
        glm::vec3 { 1, 1, 1 },
        glm::vec3 { 1, 1, -1 },
        glm::vec3 { 1, -1, 1 },
        glm::vec3 { 1, -1, -1 },
        glm::vec3 { -1, 1, 1 },
        glm::vec3 { -1, 1, -1 },
        glm::vec3 { -1, -1, 1 },
        glm::vec3 { -1, -1, -1 },
    };

    glm::mat4 matrix = viewproj * obj.transform;

    glm::vec3 min = { 1.5, 1.5, 1.5 };
    glm::vec3 max = { -1.5, -1.5, -1.5 };

    for (int c = 0; c < 8; c++) {
        // project each corner into clip space
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.f);

        // perspective correction
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        min = glm::min(glm::vec3 { v.x, v.y, v.z }, min);
        max = glm::max(glm::vec3 { v.x, v.y, v.z }, max);
    }

    // check the clip space box is within the view
    if (min.z > 1.f || max.z < 0.f || min.x > 1.f || max.x < -1.f || min.y > 1.f || max.y < -1.f) {
        return false;
    } else {
        return true;
    }
}

void AnitoWave::draw_geometry(vk::CommandBuffer cmd) {
    std::vector<uint32_t> opaque_draws;
    opaque_draws.reserve(drawCommands.OpaqueSurfaces.size());

    for (int i = 0; i < drawCommands.OpaqueSurfaces.size(); i++) {
        if (is_visible(drawCommands.OpaqueSurfaces[i], sceneData.viewproj)) {
            opaque_draws.push_back(i);
        }
    }

    // Sort the opaque surfaces by mesh and material
    std::sort(opaque_draws.begin(), opaque_draws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = drawCommands.OpaqueSurfaces[iA];
        const RenderObject& B = drawCommands.OpaqueSurfaces[iB];
        if (A.material == B.material) {
            return A.indexBuffer < B.indexBuffer;
        } else {
            return A.material < B.material;
        }
    });

    // Allocate a new uniform buffer for the scene data
    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), vk::BufferUsageFlagBits::eUniformBuffer, vma::MemoryUsage::eCpuToGpu);

    // Add buffer to deletion queue
    get_current_frame()._deletionQueue.push_function([=, this]() {
        destroy_buffer(gpuSceneDataBuffer);
    });

    // Write the buffer
    GPUSceneData* sceneUniformData = static_cast<GPUSceneData *>(static_cast<VmaAllocation>(gpuSceneDataBuffer.allocation)->GetMappedData());
    *sceneUniformData = sceneData;

    // Create descriptor sest to bind and update buffer
    vk::DescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);

    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, vk::DescriptorType::eUniformBuffer);
    writer.update_set(_device, globalDescriptor);

    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;

    vk::Buffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const RenderObject& r) {
        if (r.material != lastMaterial) {
            lastMaterial = r.material;
            if (r.material->pipeline != lastPipeline) {
                lastPipeline = r.material->pipeline;
                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, r.material->pipeline->pipeline);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, r.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);

                vk::Viewport viewport = {};
                viewport.x = 0;
                viewport.y = 0;
                viewport.width = static_cast<float>(_windowExtent.width);
                viewport.height = static_cast<float>(_windowExtent.height);
                viewport.minDepth = 0.f;
                viewport.maxDepth = 1.f;

                cmd.setViewport(0, 1, &viewport);

                vk::Rect2D scissor = {};
                scissor.offset.x = 0;
                scissor.offset.y = 0;
                scissor.extent.width = _windowExtent.width;
                scissor.extent.height = _windowExtent.height;

                cmd.setScissor(0, 1, &scissor);
            }

            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, r.material->pipeline->layout, 1, 1, &r.material->materialSet, 0, nullptr);
        }

        if (r.indexBuffer != lastIndexBuffer) {
            lastIndexBuffer = r.indexBuffer;
            cmd.bindIndexBuffer(r.indexBuffer, 0, vk::IndexType::eUint32);
        }

        GPUDrawPushConstants push_constants;
        push_constants.worldMatrix = r.transform;
        push_constants.vertexBuffer = r.vertexBufferAddress;

        cmd.pushConstants(r.material->pipeline->layout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(GPUDrawPushConstants), &push_constants);

        stats.drawcall_count++;
        stats.triangle_count += r.indexCount / 3;
        cmd.drawIndexed(r.indexCount, 1, r.firstIndex, 0, 0);
    };

    stats.drawcall_count = 0;
    stats.triangle_count = 0;

    for (auto& r : opaque_draws) {
        draw(drawCommands.OpaqueSurfaces[r]);
    }

    for (auto& r : drawCommands.TransparentSurfaces) {
        draw(r);
    }

    drawCommands.OpaqueSurfaces.clear();
    drawCommands.TransparentSurfaces.clear();
}

void AnitoWave::run() {
    SDL_Event e;
    bool bQuit = false;

    // Main engine loop
    while (!bQuit) {
        auto start = std::chrono::system_clock::now();

        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_EVENT_QUIT)
                bQuit = true;

            if (e.window.type == SDL_EVENT_WINDOW_RESIZED) {
                resize_requested = true;
            }
            if (e.window.type == SDL_EVENT_WINDOW_MINIMIZED) {
                freeze_rendering = true;
            }
            if (e.window.type == SDL_EVENT_WINDOW_RESTORED) {
                freeze_rendering = false;
            }

            mainCamera.processSDLEvent(e);
            ImGui_ImplSDL3_ProcessEvent(&e);
        }

        if (freeze_rendering) continue;

        if (resize_requested) {
            resize_swapchain();
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();


        ImGui::NewFrame();

        ImGui::Begin("Stats");

        ImGui::Text("frametime %f ms", stats.frametime);
        ImGui::Text("drawtime %f ms", stats.mesh_draw_time);
        ImGui::Text("triangles %i", stats.triangle_count);
        ImGui::Text("draws %i", stats.drawcall_count);
        ImGui::End();

        if (ImGui::Begin("background")) {

            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];

            ImGui::Text("Selected effect: ", selected.name);

            ImGui::SliderInt("Effect Index", &currentBackgroundEffect, 0, backgroundEffects.size() - 1);

            ImGui::InputFloat4("data1", (float*)&selected.data.data1);
            ImGui::InputFloat4("data2", (float*)&selected.data.data2);
            ImGui::InputFloat4("data3", (float*)&selected.data.data3);
            ImGui::InputFloat4("data4", (float*)&selected.data.data4);

            ImGui::End();
        }

        ImGui::Render();

        update_scene();

        draw();

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        stats.frametime = elapsed.count() / 1000.f;
    }
}

void AnitoWave::update_scene() {
    mainCamera.update();

    glm::mat4 view = mainCamera.getViewMatrix();

    // Camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), (float)_drawExtent.width / (float)_drawExtent.height, 10000.f, 0.1f);

    // Invert the Y direction
    projection[1][1] *= -1;

    sceneData.view = view;
    sceneData.proj = projection;
    sceneData.viewproj = projection * view;

    loadedScenes["structure"]->Draw(glm::mat4{ 1.f }, drawCommands);
}

AllocatedBuffer AnitoWave::create_buffer(size_t allocSize, vk::BufferUsageFlags usage, vma::MemoryUsage memoryUsage) {
    // Allocate buffer
    vk::BufferCreateInfo bufferInfo = vk::BufferCreateInfo()
    .setPNext(nullptr)
    .setSize(allocSize)
    .setUsage(usage);

    vma::AllocationCreateInfo vmaallocinfo = vma::AllocationCreateInfo()
    .setUsage(memoryUsage)
    .setFlags(vma::AllocationCreateFlagBits::eMapped);

    AllocatedBuffer newBuffer;

    vk::Result allocRes = _allocator.createBuffer(&bufferInfo, &vmaallocinfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info);
    if (allocRes != vk::Result::eSuccess) {
        fmt::print("Failed to allocate buffer\n");
    }

    return newBuffer;
}

AllocatedImage AnitoWave::create_image(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped) {
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    vk::ImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // Always allocate on dedicated GPU memory
    vma::AllocationCreateInfo allocinfo = vma::AllocationCreateInfo()
    .setUsage(vma::MemoryUsage::eGpuOnly)
    .setRequiredFlags(vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Allocate and create image
    vk::Result allocRes = _allocator.createImage(&img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr);
    if (allocRes != vk::Result::eSuccess) {
        fmt::print("Failed to allocate image\n");
    }

    // If the format is a depth format, use the correct aspect flag
    vk::ImageAspectFlags aspectFlag = vk::ImageAspectFlagBits::eColor;
    if (format == vk::Format::eD32Sfloat) {
        aspectFlag = vk::ImageAspectFlagBits::eDepth;
    }

    // Build an image view for the image
    vk::ImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.setLevelCount(img_info.mipLevels);

    vk::Result createRes = _device.createImageView(&view_info, nullptr, &newImage.imageView);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Failed to create image view\n");
    }

    return newImage;
}

AllocatedImage AnitoWave::create_image(void *data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped) {
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadBuffer = create_buffer(data_size, vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eCpuToGpu);

    memcpy(uploadBuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(size, format, usage | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc, mipmapped);

    immediate_submit([&](vk::CommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy copyRegion = {};
        copyRegion.setBufferOffset(0);
        copyRegion.setBufferRowLength(0);
        copyRegion.setBufferImageHeight(0);

        copyRegion.imageSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor);
        copyRegion.imageSubresource.setMipLevel(0);
        copyRegion.imageSubresource.setBaseArrayLayer(0);
        copyRegion.imageSubresource.setLayerCount(1);
        copyRegion.setImageExtent(size);

        // Copy the buffer into the image
        cmd.copyBufferToImage(uploadBuffer.buffer, new_image.image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

        if (mipmapped) {
            vkutil::generate_mipmaps(cmd, new_image.image, vk::Extent2D{new_image.imageExtent.width, new_image.imageExtent.height});
        } else {
            vkutil::transition_image(cmd, new_image.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    });

    destroy_buffer(uploadBuffer);
    return new_image;
}

GPUMeshBuffers AnitoWave::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexBuffer = create_buffer(vertexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eGpuOnly);

    vk::BufferDeviceAddressInfo deviceAddressInfo = vk::BufferDeviceAddressInfo()
    .setBuffer(newSurface.vertexBuffer.buffer);
    newSurface.vertexBufferAddress = _device.getBufferAddress(&deviceAddressInfo);

    newSurface.indexBuffer = create_buffer(indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eGpuOnly);

    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eCpuOnly);

    void* data = static_cast<VmaAllocation>(staging.allocation)->GetMappedData();

    // Copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // Copy index buffer
    memcpy((char*) data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy { 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        cmd.copyBuffer(staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        vk::BufferCopy indexCopy { 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        cmd.copyBuffer(staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(staging);

    return newSurface;
}

FrameData &AnitoWave::get_current_frame() {
    return _frames[_frameNumber % FRAME_OVERLAP];
}

FrameData &AnitoWave::get_last_frame() {
    return _frames[(_frameNumber - 1) % FRAME_OVERLAP];
}

void AnitoWave::immediate_submit(std::function<void(vk::CommandBuffer cmd)> &&function) {
    vk::Result resetRes = _device.resetFences(1, &_immFence);
    if (resetRes != vk::Result::eSuccess) {
        fmt::print("Failed to reset fences.\n");
    }
    _immCommandBuffer.reset();

    vk::CommandBuffer cmd = _immCommandBuffer;
    // Begin recording the command buffer.
    vk::CommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    vk::Result beginRes = cmd.begin(&cmdBeginInfo);
    if (beginRes != vk::Result::eSuccess) {
        fmt::print("Failed to begin command buffer.\n");
    }

    function(cmd);

    cmd.end();

    vk::CommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    vk::SubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // Submit command buffer to the queue and execute it.
    // _renderFence blocks until commands finish executing.
    vk::Result submitRes = _graphicsQueue.submit2(1, &submit, _immFence);
    if (submitRes != vk::Result::eSuccess) {
        fmt::print("Failed to submit command buffer.\n");
    }

    vk::Result waitRes = _device.waitForFences(1, &_immFence, true, 9999999999);
    if (waitRes != vk::Result::eSuccess) {
        fmt::print("Failed to wait for immediate fences.\n");
    }
}

void AnitoWave::destroy_image(const AllocatedImage &img) {
    _device.destroyImageView(img.imageView);
    _allocator.destroyImage(img.image, img.allocation);
}

void AnitoWave::destroy_buffer(const AllocatedBuffer &buffer) {
    _allocator.destroyBuffer(buffer.buffer, buffer.allocation);
}

void AnitoWave::init_vulkan() {
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    vkb::InstanceBuilder builder;

    auto inst_ret = builder.set_app_name("AnitoWave")
    .request_validation_layers(bUseValidationLayers)
    .use_default_debug_messenger()
    .require_api_version(1, 4, 0)
    .build();

    if (!inst_ret) {
        throw std::runtime_error("failed to create instance: " + inst_ret.error().message());
    }

    vkb::Instance vkb_inst = inst_ret.value();

    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    VULKAN_HPP_DEFAULT_DISPATCHER.init(_instance);

    VkSurfaceKHR surface;
    SDL_Vulkan_CreateSurface(_window, _instance, nullptr, &surface);
    _surface = surface, surface = nullptr;

    vk::PhysicalDeviceVulkan13Features features = vk::PhysicalDeviceVulkan13Features().setDynamicRendering(true).setSynchronization2(true);

    vk::PhysicalDeviceVulkan12Features features12 = vk::PhysicalDeviceVulkan12Features().setBufferDeviceAddress(true).setDescriptorIndexing(true);

    vkb::PhysicalDeviceSelector selector{vkb_inst};
    auto phys_ret = selector.set_minimum_version(1, 4)
    .set_required_features_13(features)
    .set_required_features_12(features12)
    .set_surface(_surface)
    .add_required_extension(vk::KHRPushDescriptorExtensionName)
    .select();

    if (!phys_ret) {
        throw std::runtime_error("Failed to create physical device: " + phys_ret.error().message());
    }

    vkb::PhysicalDevice physicalDevice = phys_ret.value();

    vkb::DeviceBuilder deviceBuilder{physicalDevice};

    vkb::Device vkbDevice = deviceBuilder.build().value();

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    VULKAN_HPP_DEFAULT_DISPATCHER.init(_device);

    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    vma::AllocatorCreateInfo allocatorInfo = vma::AllocatorCreateInfo()
    .setPhysicalDevice(_chosenGPU)
    .setDevice(_device)
    .setInstance(_instance)
    .setFlags(vma::AllocatorCreateFlagBits::eBufferDeviceAddress);

    _allocator = vma::createAllocator(allocatorInfo);

    _mainDeletionQueue.push_function([&]() { _allocator.destroy(); });
}

void AnitoWave::init_swapchain() {
    create_swapchain(_windowExtent.width, _windowExtent.height);

    // Depth image matches the window
    vk::Extent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    // Hard coded 32-bit float
    _drawImage.imageFormat = vk::Format::eR16G16B16A16Sfloat;
    _drawImage.imageExtent = drawImageExtent;

    vk::ImageUsageFlags drawImageUsages = vk::ImageUsageFlagBits::eColorAttachment;
    drawImageUsages |= vk::ImageUsageFlagBits::eTransferSrc;
    drawImageUsages |= vk::ImageUsageFlagBits::eStorage;
    drawImageUsages |= vk::ImageUsageFlagBits::eColorAttachment;

    vk::ImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    // Allocate draw image from GPU local memory
    vma::AllocationCreateInfo rimg_allocinfo = vma::AllocationCreateInfo().setUsage(vma::MemoryUsage::eGpuOnly).setRequiredFlags(vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Allocate and create the image
    vk::Result createRes = _allocator.createImage(&rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Failed to create image!");
    }

    // Build an image-view for the draw image
    vk::ImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, vk::ImageAspectFlagBits::eColor);

    createRes = _device.createImageView(&rview_info, nullptr, &_drawImage.imageView);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Failed to create image view!");
    }

    // Create a depth image
    _depthImage.imageFormat = vk::Format::eD32Sfloat;
    _depthImage.imageExtent = drawImageExtent;

    vk::ImageUsageFlags depthImageUsages{};
    depthImageUsages |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    vk::ImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    // Allocate and create depth image
    createRes = _allocator.createImage(&dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Failed to create depth image!");
    }

    // Build an image-view for the depth image
    vk::ImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, vk::ImageAspectFlagBits::eDepth);

    createRes = _device.createImageView(&dview_info, nullptr, &_depthImage.imageView);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Failed to create depth image view!");
    }

    _mainDeletionQueue.push_function([=]() {
        _device.destroyImageView(_drawImage.imageView);
        _allocator.destroyImage(_drawImage.image, _drawImage.allocation);

        _device.destroyImageView(_depthImage.imageView);
        _allocator.destroyImage(_depthImage.image, _depthImage.allocation);
    });
}

void AnitoWave::create_swapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device, _surface };

    _swapchainImageFormat = vk::Format::eB8G8R8A8Unorm;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
    .set_desired_format(vk::SurfaceFormatKHR(_swapchainImageFormat, vk::ColorSpaceKHR::eSrgbNonlinear))
    .set_desired_present_mode(static_cast<VkPresentModeKHR>(vk::PresentModeKHR::eFifo))
    .set_desired_extent(width, height)
    .add_image_usage_flags(static_cast<VkImageUsageFlags>(vk::ImageUsageFlagBits::eTransferDst))
    .build().value();

    _swapchainExtent = vkbSwapchain.extent;

    // Store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;

    auto vkb_images = vkbSwapchain.get_images().value();
    _swapchainImages = {vkb_images.begin(), vkb_images.end()};

    auto vkb_image_views = vkbSwapchain.get_image_views().value();
    _swapchainImageViews = {vkb_image_views.begin(), vkb_image_views.end()};
}

void AnitoWave::destroy_swapchain() {
    _device.destroySwapchainKHR(_swapchain);

    // Destroy swapchain resources
    for (int i = 0; i < _swapchainImages.size(); i++) {
        _device.destroyImageView(_swapchainImageViews[i]);
    }
}

void AnitoWave::resize_swapchain() {
    _device.waitIdle();

    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resize_requested = false;
}

void AnitoWave::init_commands() {
    // Create a command pool for commands submitted to the graphics queue
    // Allow for resetting of individual command buffers
    vk::CommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        vk::Result createRes = _device.createCommandPool(&commandPoolInfo, nullptr, &_frames[i]._commandPool);
        if (createRes != vk::Result::eSuccess) {
            fmt::print("Failed to create command pool!");
        }

        // Allocate the default command buffer for rendering
        vk::CommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        vk::Result allocRes = _device.allocateCommandBuffers(&cmdAllocInfo, &_frames[i]._mainCommandBuffer);
        if (allocRes != vk::Result::eSuccess) {
            fmt::print("Failed to allocate command buffers!");
        }

        _mainDeletionQueue.push_function([=]() { _device.destroyCommandPool(_frames[i]._commandPool); });
    }

    vk::Result createRes = _device.createCommandPool(&commandPoolInfo, nullptr, &_immCommandPool);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Failed to create immediate command pool!");
    }

    // Allocate command buffer for immediate rendering
    vk::CommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    vk::Result allocRes = _device.allocateCommandBuffers(&cmdAllocInfo, &_immCommandBuffer);
    if (allocRes != vk::Result::eSuccess) {
        fmt::print("Failed to allocate immediate command buffers!");
    }

    _mainDeletionQueue.push_function([=]() { _device.destroyCommandPool(_immCommandPool); });
}

void AnitoWave::init_sync_structures() {
    // Create sync structures
    // One fence to control when the GPU has finished rendering a frame,
    // and 2 semaphores to synchronize rendering with swapchain.
    // The fence starts as signaled in order to wait for it on the first frame.
    vk::FenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(vk::FenceCreateFlagBits::eSignaled);
    vk::Result createRes = _device.createFence(&fenceCreateInfo, nullptr, &_immFence);
    if (createRes != vk::Result::eSuccess) {
        fmt::print("Failed to create immediate fence!");
    }

    _mainDeletionQueue.push_function([=]() { _device.destroyFence(_immFence); });

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        createRes = _device.createFence(&fenceCreateInfo, nullptr, &_frames[i]._renderFence);
        if (createRes != vk::Result::eSuccess) {
            fmt::print("Failed to create render fence!");
        }

        vk::SemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

        createRes = _device.createSemaphore(&semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore);
        if (createRes != vk::Result::eSuccess) {
            fmt::print("Failed to create swapchain semaphore!");
        }

        createRes = _device.createSemaphore(&semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore);
        if (createRes != vk::Result::eSuccess) {
            fmt::print("Failed to create render semaphore!");
        }

        _mainDeletionQueue.push_function([=]() {
            _device.destroyFence(_frames[i]._renderFence);
            _device.destroySemaphore(_frames[i]._swapchainSemaphore);
            _device.destroySemaphore(_frames[i]._renderSemaphore);
        });
    }
}

void AnitoWave::init_renderables() {
    std::string structurePath = { "..//assets//structure.glb" };
    auto structureFile = loadGltf(this, structurePath);

    assert(structureFile.has_value());

    loadedScenes["structure"] = *structureFile;
}

void AnitoWave::init_imgui() {
    // Create descriptor pool for imgui
    vk::DescriptorPoolSize poolSizes[] = {
            {vk::DescriptorType::eSampler, 1000},
            {vk::DescriptorType::eCombinedImageSampler, 1000},
            {vk::DescriptorType::eSampledImage, 1000},
            {vk::DescriptorType::eStorageImage, 1000},
            {vk::DescriptorType::eUniformTexelBuffer, 1000},
            {vk::DescriptorType::eStorageTexelBuffer, 1000},
            {vk::DescriptorType::eUniformBuffer, 1000},
            {vk::DescriptorType::eStorageBuffer, 1000},
            {vk::DescriptorType::eUniformBufferDynamic, 1000},
            {vk::DescriptorType::eStorageBufferDynamic, 1000},
            {vk::DescriptorType::eInputAttachment, 1000},
    };

    vk::DescriptorPoolCreateInfo pool_info = vk::DescriptorPoolCreateInfo()
                                                     .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
                                                     .setMaxSets(1000)
                                                     .setPoolSizeCount((uint32_t) std::size(poolSizes))
                                                     .setPPoolSizes(poolSizes);

    vk::DescriptorPool imguiPool = _device.createDescriptorPool(pool_info);

    // Init imgui library

    ImGui::CreateContext();

    ImGui_ImplSDL3_InitForVulkan(_window);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    // Dynamic rendering parameters for imgui
    init_info.PipelineRenderingCreateInfo =
            vk::PipelineRenderingCreateInfo().setColorAttachmentCount(1).setPColorAttachmentFormats(
                    &_swapchainImageFormat);

    init_info.MSAASamples = static_cast<VkSampleCountFlagBits>(vk::SampleCountFlagBits::e1);

    ImGui_ImplVulkan_Init(&init_info);

    // ImGui_ImplVulkan_CreateFontsTexture();

    _mainDeletionQueue.push_function([=, this]() {
        ImGui_ImplVulkan_Shutdown();
        _device.destroyDescriptorPool(imguiPool);
    });
}

void AnitoWave::init_pipelines() {
    // COMPUTE PIPELINES
    init_background_pipelines();

    metalRoughMaterial.build_pipelines(this);
}

void AnitoWave::init_descriptors() {
    // Create a descriptor pool
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        { vk::DescriptorType::eStorageImage, 3},
        { vk::DescriptorType::eStorageBuffer, 3},
        { vk::DescriptorType::eCombinedImageSampler, 3},
    };

    globalDescriptorAllocator.init_pool(_device, 10, sizes);
    _mainDeletionQueue.push_function([&]() { _device.destroyDescriptorPool(globalDescriptorAllocator.pool); });

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, vk::DescriptorType::eSampledImage);
        _drawImageDescriptorLayout = builder.build(_device, vk::ShaderStageFlagBits::eCompute);
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, vk::DescriptorType::eUniformBuffer);
        _gpuSceneDataDescriptorLayout = builder.build(_device, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    }

    _mainDeletionQueue.push_function([&]() {
        _device.destroyDescriptorSetLayout(_drawImageDescriptorLayout);
        _device.destroyDescriptorSetLayout(_gpuSceneDataDescriptorLayout);
    });

    _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, vk::ImageLayout::eGeneral, vk::DescriptorType::eStorageImage);
        writer.update_set(_device, _drawImageDescriptors);
    }

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        // Create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            { vk::DescriptorType::eStorageImage, 3 },
            { vk::DescriptorType::eStorageBuffer, 3 },
            { vk::DescriptorType::eUniformBuffer, 3 },
            { vk::DescriptorType::eCombinedImageSampler, 3 },
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);
        _mainDeletionQueue.push_function([&, i]() {
            _frames[i]._frameDescriptors.destroy_pools(_device);
        });
    }
}

void GLTFMetallic_Roughness::build_pipelines(AnitoWave *engine) {
    vk::ShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../shaders/mesh.frag.spv", engine->_device, &meshFragShader)) {
        fmt::println("Error when building the triangle fragment shader module.");
    }

    vk::ShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../shaders/mesh.vert.spv", engine->_device, &meshVertexShader)) {
        fmt::println("Error when building the triangle vertex shader module.");
    }

    vk::PushConstantRange matrixRange = vk::PushConstantRange()
    .setOffset(0)
    .setSize(sizeof(GPUDrawPushConstants))
    .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, vk::DescriptorType::eUniformBuffer);
    layoutBuilder.add_binding(1, vk::DescriptorType::eCombinedImageSampler);
    layoutBuilder.add_binding(2, vk::DescriptorType::eCombinedImageSampler);

    materialLayout = layoutBuilder.build(engine->_device, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);

    vk::DescriptorSetLayout layouts[] = { engine->_gpuSceneDataDescriptorLayout, materialLayout };

    vk::PipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount(2);
    mesh_layout_info.setPSetLayouts(layouts);
    mesh_layout_info.setPPushConstantRanges(&matrixRange);
    mesh_layout_info.setPushConstantRangeCount(1);

    vk::PipelineLayout newLayout;
    vk::Result createRes = engine->_device.createPipelineLayout(&mesh_layout_info, nullptr, &newLayout);
    if (createRes != vk::Result::eSuccess) {
        fmt::println("Error creating the pipeline layout.");
    }

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    // Build the stage-create-info for both vertex and fragment stages.
    PipelineBuilder pipelineBuilder;

    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);

    pipelineBuilder.set_input_topology(vk::PrimitiveTopology::eTriangleList);

    pipelineBuilder.set_polygon_mode(vk::PolygonMode::eFill);

    pipelineBuilder.set_cull_mode(vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise);

    pipelineBuilder.set_multisampling_none();

    pipelineBuilder.disable_blending();

    pipelineBuilder.enable_depthtest(true, vk::CompareOp::eGreaterOrEqual);

    // Render format
    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    // Use triangle layout
    pipelineBuilder._pipelineLayout = newLayout;

    // Build the pipeline
    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    pipelineBuilder.enable_depthtest(false, vk::CompareOp::eGreaterOrEqual);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    engine->_device.destroyShaderModule(meshFragShader);
    engine->_device.destroyShaderModule(meshVertexShader);
}

void GLTFMetallic_Roughness::clear_resources(vk::Device device) {

}

MaterialInstance GLTFMetallic_Roughness::write_material(vk::Device device, MaterialPass pass, const MaterialResources &resources, DescriptorAllocatorGrowable &descriptorAllocator) {
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent) {
        matData.pipeline = &transparentPipeline;
    } else {
        matData.pipeline = &opaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, vk::DescriptorType::eUniformBuffer);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal, vk::DescriptorType::eCombinedImageSampler);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler, vk::ImageLayout::eShaderReadOnlyOptimal, vk::DescriptorType::eCombinedImageSampler);

    writer.update_set(device, matData.materialSet);

    return matData;
}

void MeshNode::Draw(const glm::mat4 &topMatrix, DrawContext &ctx) {
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : mesh->surfaces) {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;
        def.bounds = s.bounds;
        def.transform = nodeMatrix;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;

        if (s.material->data.passType == MaterialPass::Transparent) {
            ctx.TransparentSurfaces.push_back(def);
        } else {
            ctx.OpaqueSurfaces.push_back(def);
        }
    }

    Node::Draw(topMatrix, ctx);
}
