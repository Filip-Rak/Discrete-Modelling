#include "Controller.h"

int main() 
{
    const int window_width = 1600, window_height = 900;
    // const int window_width = 1400, window_height = 900;
    // const int grid_width = 110, grid_height = 90;   // Def
    const int grid_width = 600, grid_height = 450;

    // Create and run the app
    Controller controller(window_width, window_height, grid_width, grid_height);
    controller.run();

    return 0;
}
