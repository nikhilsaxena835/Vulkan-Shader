cmake_minimum_required(VERSION 3.20)
project(NPlayer LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project directories
set(PROJECT_ROOT /home/nikhil-saxena/Documents/GitHub/NPlayer)
set(INCLUDE_DIR ${PROJECT_ROOT}/include)
set(LIB_DIR ${PROJECT_ROOT}/lib)
set(SOURCE_DIR ${PROJECT_ROOT}/src)
set(SHADER_DIR ${PROJECT_ROOT}/shaders)
set(ASSETS_DIR ${PROJECT_ROOT}/assets)

# Find Vulkan
set(VULKAN_SDK ${PROJECT_ROOT})
find_library(VULKAN_LIBRARY NAMES vulkan vulkan-1 PATHS ${LIB_DIR}/vulkan)
find_path(VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h PATHS ${INCLUDE_DIR}/vulkan)
if(NOT VULKAN_LIBRARY OR NOT VULKAN_INCLUDE_DIR)
    message(FATAL_ERROR "Vulkan SDK not found in ${VULKAN_SDK}")
endif()

# Find OpenCV
find_library(OPENCV_CORE_LIBRARY NAMES opencv_core opencv_core480 PATHS ${LIB_DIR}/opencv4)
find_library(OPENCV_DNN_LIBRARY NAMES opencv_dnn opencv_dnn480 PATHS ${LIB_DIR}/opencv4)
find_library(OPENCV_VIDEOIO_LIBRARY NAMES opencv_videoio opencv_videoio480 PATHS ${LIB_DIR}/opencv4)
find_path(OPENCV_INCLUDE_DIR NAMES opencv2/opencv.hpp PATHS ${INCLUDE_DIR}/opencv4)
if(NOT OPENCV_CORE_LIBRARY OR NOT OPENCV_DNN_LIBRARY OR NOT OPENCV_VIDEOIO_LIBRARY OR NOT OPENCV_INCLUDE_DIR)
    message(FATAL_ERROR "OpenCV not found in ${LIB_DIR}/opencv4 or ${INCLUDE_DIR}/opencv4")
endif()

# Find SDL2 (optional)
find_library(SDL2_LIBRARY NAMES SDL2 PATHS ${LIB_DIR}/SDL2)
find_path(SDL2_INCLUDE_DIR NAMES SDL.h PATHS ${INCLUDE_DIR}/SDL2)
if(SDL2_LIBRARY AND SDL2_INCLUDE_DIR)
    set(USE_SDL2 TRUE)
else()
    set(USE_SDL2 FALSE)
    message(WARNING "SDL2 not found; using ImGui for display")
endif()

# Include directories
include_directories(
    ${VULKAN_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${INCLUDE_DIR}/imgui
    ${SDL2_INCLUDE_DIR}
    ${SOURCE_DIR}
)

# Source files
set(SOURCES
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/core/vulkan_engine.cpp
    ${SOURCE_DIR}/core/compute_pipeline.cpp
    ${SOURCE_DIR}/core/buffer_manager.cpp
    ${SOURCE_DIR}/core/shader_manager.cpp
    ${SOURCE_DIR}/processing/frame_processor.cpp
    ${SOURCE_DIR}/processing/object_detector.cpp
    ${SOURCE_DIR}/processing/mask_generator.cpp
    ${SOURCE_DIR}/io/video_io.cpp
    ${SOURCE_DIR}/io/ppm_handler.cpp
    ${SOURCE_DIR}/ui/ui_manager.cpp
    ${SOURCE_DIR}/ui/shader_controls.cpp
    ${INCLUDE_DIR}/imgui/imgui.cpp
    ${INCLUDE_DIR}/imgui/imgui_widgets.cpp
    ${INCLUDE_DIR}/imgui/imgui_draw.cpp
    ${INCLUDE_DIR}/imgui/imgui_tables.cpp
    ${INCLUDE_DIR}/imgui/imgui_impl_vulkan.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${VULKAN_LIBRARY}
    ${OPENCV_CORE_LIBRARY}
    ${OPENCV_DNN_LIBRARY}
    ${OPENCV_VIDEOIO_LIBRARY}
)
if(USE_SDL2)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${SDL2_LIBRARY})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_SDL2)
endif()

# Define assets and shader paths
target_compile_definitions(${PROJECT_NAME} PRIVATE
    SHADER_DIR="${SHADER_DIR}"
    ASSET_DIR="${ASSETS_DIR}"
)

# Compiler flags
if(UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
elseif(WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
endif()

# Custom target for shader compilation
add_custom_target(CompileShaders
    COMMAND ${PROJECT_ROOT}/scripts/compile_shaders.sh
    WORKING_DIRECTORY ${PROJECT_ROOT}
    COMMENT "Compiling GLSL shaders to SPIR-V"
)
add_dependencies(${PROJECT_NAME} CompileShaders)

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY ${SHADER_DIR}/ DESTINATION shaders FILES_MATCHING PATTERN "*.spv")
install(DIRECTORY ${ASSETS_DIR}/ DESTINATION assets)
