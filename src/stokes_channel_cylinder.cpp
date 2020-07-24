#include <iostream>
#include "nitsche_stokes.h"

int main() {
    std::cout << "StokesNitsche" << std::endl;
    {
        using namespace Stokes;
        StokesNitsche<2> stokesNitsche(1);
        stokesNitsche.run();
    }
}
