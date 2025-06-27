// g++ -o main main.cpp -lvulkan

/*
This program works as follows : 
    1) Using ffmpeg decompose video into images/frames.
    2) On each of these images, run the ML model and make shades.
    3) For each new masked image, run the Vulkan Compute code.
    
    Future Goals : Have a GUI using IMGUI for this system, rendering on screen is not a goal.
    The syntax is ./main <path_to_video_file> <compiled_shader_path> <flag_object_detection> 
    Eg : ./main test/video.mp4 ghibli.spv false
*/

#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include "io/video_io.hpp"




int main(int argc, char* argv[])
{
    try{

        if (!checkFFMPEG()) 
                throw std::runtime_error("ffmpeg is not installed. Please install ffmpeg to continue.");
            
        std::string videoPath;
        std::string shaderPath;
        bool objectDetection = false;
        std::cout << argc << std::endl;
        if (argc == 4) 
        {   
            videoPath = argv[1];
            shaderPath = argv[2];
            std::string temp = argv[3];
            if(temp == "true")
            objectDetection = true;
        }
        else 
        {
            std::cout << "Incorrect syntax : ./main <path_to_video_file> <compiled_shader_path> <flag_object_detection> ";
            return EXIT_SUCCESS;
        }
    
        if (!std::filesystem::exists(videoPath)) 
            throw std::runtime_error("Input video file does not exist: " + videoPath);

        std::filesystem::path inputPath(videoPath);
        std::string baseDir = inputPath.parent_path().string();
        std::string tempFramesDir = baseDir + "/temp_frames";
        std::string processedFramesDir = baseDir + "/processed_frames";
        std::string outputVideo = baseDir + "/output_" + inputPath.filename().string();    

        std::cout << "Extracting frames from video ..." << std::endl;
        extractFrames(videoPath, tempFramesDir);

        if(objectDetection){
            std::cout << "Masking frames ..." << std::endl;

        }
    }
    catch (const std::exception& e) 
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    
    return EXIT_SUCCESS;
}