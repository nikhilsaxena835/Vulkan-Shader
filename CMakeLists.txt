cmake_minimum_required(VERSION 3.20)
project(NPlayer LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project directories early
set(PROJECT_ROOT /home/nikhil-saxena/Documents/GitHub/NPlayer)
set(INCLUDE_DIR ${PROJECT_ROOT}/include)
set(LIB_DIR ${PROJECT_ROOT}/lib)
set(SOURCE_DIR ${PROJECT_ROOT}/src)
#set(SHADER_DIR ${PROJECT_ROOT}/shaders)
#set(ASSETS_DIR ${PROJECT_ROOT}/assets)

# Source files
set(SOURCES
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/core/vulkan_engine.cpp
    ${SOURCE_DIR}/core/pipeline.cpp
    ${SOURCE_DIR}/core/buffer_manager.cpp
    ${SOURCE_DIR}/core/shader_manager.cpp
    ${SOURCE_DIR}/processing/frame_processor.cpp
    ${SOURCE_DIR}/processing/object_detector.cpp
    ${SOURCE_DIR}/processing/mask_generator.cpp
    ${SOURCE_DIR}/io/video_io.cpp
    ${SOURCE_DIR}/io/ppm_handler.cpp
#    ${SOURCE_DIR}/ui/ui_manager.cpp
#    ${SOURCE_DIR}/ui/shader_controls.cpp
#    ${INCLUDE_DIR}/imgui/imgui.cpp
#    ${INCLUDE_DIR}/imgui/imgui_widgets.cpp
#    ${INCLUDE_DIR}/imgui/imgui_draw.cpp
#    ${INCLUDE_DIR}/imgui/imgui_tables.cpp
#    ${INCLUDE_DIR}/imgui/imgui_impl_vulkan.cpp
)

# Define executable target
add_executable(${PROJECT_NAME} ${SOURCES})

# Find Vulkan
set(VULKAN_SDK ${PROJECT_ROOT})
find_library(VULKAN_LIBRARY NAMES vulkan vulkan-1 PATHS ${LIB_DIR}/vulkan)
find_path(VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h PATHS ${INCLUDE_DIR}/vulkan)

if(NOT VULKAN_LIBRARY OR NOT VULKAN_INCLUDE_DIR)
    message(FATAL_ERROR "Vulkan SDK not found in ${VULKAN_SDK}")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(
    ${VULKAN_INCLUDE_DIR}
    ${INCLUDE_DIR}/imgui
    ${SOURCE_DIR}
)

# Link all required libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${VULKAN_LIBRARY}
    ${OpenCV_LIBS}
)


# Define asset and shader paths as compile definitions
#target_compile_definitions(${PROJECT_NAME} PRIVATE
#    SHADER_DIR="${SHADER_DIR}"
#    ASSET_DIR="${ASSETS_DIR}"
#)

# Compiler flags
if(UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
elseif(WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
endif()

# Custom target to compile shaders
#add_custom_target(CompileShaders
#    COMMAND ${PROJECT_ROOT}/scripts/compile_shaders.sh
#    WORKING_DIRECTORY ${PROJECT_ROOT}
#    COMMENT "Compiling GLSL shaders to SPIR-V"
#)
#add_dependencies(${PROJECT_NAME} CompileShaders)

# Installation targets
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY ${SHADER_DIR}/ DESTINATION shaders FILES_MATCHING PATTERN "*.spv")
install(DIRECTORY ${ASSETS_DIR}/ DESTINATION assets)
