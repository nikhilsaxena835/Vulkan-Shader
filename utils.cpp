
#include "utils.h"

/*
Though system() is not safe, we know what we are doing. The command has two parts. Part one checks for the ffmpeg
version. And part two has two subparts. First we redirect any output (from the stdout) to /dev/null. Dev contains 
device files that are connected. dev/null is a pseudo device file. So we are redirecting the output to nowhere. 
Next there are three streams : input file stream (0), output file stream (1) and error stream (2). 

We also want to redirect the error stream to the output stream which is already redirected to the nowhere place.
Therefore doing this allows this command to execute and not produce any output. The return value of system() is the 
exit code of the process. Zero means success and anything else means failure.
*/
bool checkFFMPEG() {
    return system("ffmpeg -version > /dev/null 2>&1") == 0;
}