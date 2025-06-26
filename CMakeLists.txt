cmake_minimum_required(VERSION 3.31)
project(AnitoWave)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
add_compile_definitions(GLM_FORCE_DEPTH_ZERO_TO_ONE)
add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

if (MSVC)
    add_compile_options(/Zc:_cplusplus)
    add_compile_options(-Wno-nullability-completeness)
    add_compile_options(/EHsc)
endif ()

add_subdirectory(SDL EXCLUDE_FROM_ALL)
find_package(fmt)
find_package(glm)
find_package(Vulkan REQUIRED)
find_package(fastgltf)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/vma)
include_directories(${Vulkan_INCLUDE_DIRS})

# --------------------
# imgui
# --------------------

set(IMGUI_DIR imgui)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

# --------------------
# Shader Compilation
# --------------------

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

file(GLOB_RECURSE SHADER_FILES
        ${SHADER_SOURCE_DIR}/*.vert
        ${SHADER_SOURCE_DIR}/*.frag
        ${SHADER_SOURCE_DIR}/*.comp
)

set(COMPILED_SHADERS)

foreach (SHADER ${SHADER_FILES})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(OUTPUT_FILE ${SHADER_BINARY_DIR}/${FILE_NAME}.spv)

    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND glslangValidator -V ${SHADER} -o ${OUTPUT_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader  ${FILE_NAME}"
            VERBATIM
    )

    list(APPEND COMPILED_SHADERS ${OUTPUT_FILE})
endforeach ()

add_custom_target(compile_shaders ALL DEPENDS ${COMPILED_SHADERS})

# -----------------------------
# Model and texture build files
# -----------------------------

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# ---------------------------
# vk-bootstrap
# ---------------------------
include(FetchContent)
FetchContent_Declare(
        fetch_vk_bootstrap
        GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
        GIT_TAG main
)
FetchContent_MakeAvailable(fetch_vk_bootstrap)

add_executable(AnitoWave main.cpp
        include/vk_include.h
        engine/anito_wave.cpp
        engine/anito_wave.h
        ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
        ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        include/vk_disp_loader.cpp
        include/tiny_obj_loader.cpp
        engine/types/vk_types.h
        engine/pipelines/vk_pipelines.cpp
        engine/pipelines/vk_pipelines.h
        engine/initializers/vk_initializers.cpp
        engine/initializers/vk_initializers.h
        engine/images/vk_images.cpp
        engine/images/vk_images.h
        engine/descriptors/vk_descriptors.cpp
        engine/descriptors/vk_descriptors.h
        engine/loader/vk_loader.cpp
        engine/loader/vk_loader.h
        engine/camera/camera.cpp
        engine/camera/camera.h)

add_dependencies(AnitoWave compile_shaders)

target_link_libraries(AnitoWave SDL3::SDL3)
target_link_libraries(AnitoWave glm::glm)
target_link_libraries(AnitoWave Vulkan::Vulkan)
target_link_libraries(AnitoWave vk-bootstrap::vk-bootstrap)
target_link_libraries(AnitoWave fmt::fmt)
target_link_libraries(AnitoWave fastgltf::fastgltf)