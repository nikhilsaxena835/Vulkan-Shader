#include <string>
#include <stdexcept>

void extractFrames(const std::string& videoPath, const std::string& outputDir);

bool checkFFMPEG();

void createVideo(const std::string& inputFramesDir, const std::string& outputVideo, const std::string& inputVideo, int framerate);
