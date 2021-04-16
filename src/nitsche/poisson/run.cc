#include "poisson.h"


int main() {
    std::cout << "PoissonNitsche" << std::endl;
    {
        PoissonNitsche<2> poisson(1, 6);
        poisson.run();
    }
}
