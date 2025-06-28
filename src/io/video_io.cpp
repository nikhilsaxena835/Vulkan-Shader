#include "video_io.hpp"
#include <iostream>


/*
Though system() is not safe, we will still do it this way. The command has two parts. 
Part one checks for the ffmpeg version. And part two has two subparts. First we redirect 
any output (from the stdout) to /dev/null. Dev contains device files that are connected. 
dev/null is a pseudo device file. So we are redirecting the output to nowhere. 
Next there are three streams : input file stream (0), output file stream (1) and error stream (2). 

We also want to redirect the error stream to the output stream which is already redirected 
to the nowhere place. Therefore doing this allows this command to execute and not produce 
any output. The return value of system() is the exit code of the process. 
Zero means success and anything else means failure.
*/
bool checkFFMPEG() {
    return system("ffmpeg -version > /dev/null 2>&1") == 0;
}

void extractFrames(const std::string& videoPath, const std::string& outputDir)
{
    std::string command = "ffmpeg -i \"" + videoPath + "\" -vf \"fps=30,format=rgb24\" \"" + 
                      outputDir + "/frame_%d.ppm\" -start_number 1 2>/dev/null";

    if (system(command.c_str()) != 0) 
        throw std::runtime_error("Failed to extract frames from video");
    
}

void createVideo(const std::string& inputFramesDir, const std::string& outputVideo, const std::string& inputVideo, int framerate = 30) 
{
    std::string command = "ffmpeg -framerate " + std::to_string(framerate) + 
                    " -i \"" + inputFramesDir + "/processed_frame_%d.ppm\" " +
                    " -i \"" + inputVideo + "\" " +  // Add input video as second input
                    "-c:v libx264 -pix_fmt yuv420p " +
                    "-c:a copy " +  // Copy audio stream without re-encoding
                    "-map 0:v:0 " +  // Map video from first input (processed frames)
                    "-map 1:a:0 " +  // Map audio from second input (original video)
                    "\"" + outputVideo + "\" 2>/dev/null";
                    
    if (system(command.c_str()) != 0) {
        throw std::runtime_error("Failed to create output video");
    }
}