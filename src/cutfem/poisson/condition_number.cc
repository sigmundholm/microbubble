#include "poisson.h"

void write_cond_num(std::ofstream &file, int step, double cond_num) {
    file << step << "," << cond_num << std::endl;
}


template<int dim>
void condition_number_sensitivity() {

    int element_order = 1;
    int n_refines = 5;
    bool write_output = true;

    double radius = 1.1;
    double half_length = 1.2;
    double sphere_rad = 1.0;

    std::ofstream file("condnums-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order)
                       + "r" + std::to_string(n_refines) + ".csv");

    double h, center;
    unsigned int n = 500;
    const bool stabilized = true;

    for (unsigned int k = 0; k <= n; ++k) {
        std::cout << std::endl << "k = " << k << std::endl;
        center = k / (pow(2, 0.5) * n) * h;

        RightHandSide<dim> rhs(center, center);
        BoundaryValues<dim> bdd(center, center);
        AnalyticalSolution<dim> soln(center, center);

        Poisson<dim> poisson(radius, half_length, n_refines, element_order,
                             write_output, rhs, bdd, soln, sphere_rad, center,
                             center, stabilized);

        Error error = poisson.run(true, "-k" + std::to_string(k));

        file << k << "," << error.cond_num << std::endl;
        h = error.mesh_size;
    }
}


int main() {
    const int dim = 2;
    condition_number_sensitivity<dim>();

    return 0;
}