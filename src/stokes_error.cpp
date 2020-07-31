#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "nitsche_stokes.h"

using namespace dealii;

namespace Error {
    using namespace Stokes;

    template<int dim>
    class ErrorRightHandSide : public RightHandSide<dim> {
    public:
        double point_value(const Point<dim> &p, const unsigned int component = 0) const;
    };

    template<int dim>
    double ErrorRightHandSide<dim>::point_value(const Point<dim> &p, const unsigned int component) const {
        (void) p;
        std::cout << "RightHandSide used" << std::endl;
        double pressure_drop = 15.5;
        double length = 2;
        if (component == 1) {
            return pressure_drop / length;
        }
        return 0;
    }

    template<int dim>
    class ErrorBoundaryValues : public BoundaryValues<dim> {
    public:
        ErrorBoundaryValues(double left_boundary, double radius);

        virtual double point_value(const Point<dim> &p, const unsigned int component) const;

        double left_boundary;  // x-coordinate of left side of the domain.
        double radius; // radius of the cylinder
    };


    template<int dim>
    ErrorBoundaryValues<dim>::ErrorBoundaryValues(double left_boundary, double radius)
            : left_boundary(left_boundary), radius(radius) {}

    template<int dim>
    double ErrorBoundaryValues<dim>::point_value(const Point<dim> &p, const unsigned int component) const {
        (void) p;
        // std::cout << "ErrorBoundaryValues.point_value" << std::endl;
        if (component == 0 && p[0] <= left_boundary + 0.001) {
            if (dim == 2) {
                return -2.5 * (p[1] - radius) * (p[1] + radius);
            }
            throw std::exception(); // TODO fix 3D
        }
        return 0;
    }


    template<int dim>
    class ExactSolution : public Function<dim> {
    public:
        ExactSolution(const double radius, const double length,
                      const double pressure_drop);

        virtual void vector_value(const Point<dim> &p, Vector<double> &value) const override;

        double radius;
        double length;
        double pressure_drop;
    };

    template<int dim>
    ExactSolution<dim>::ExactSolution(const double radius, const double length,
                                      const double pressure_drop)
            : Function<dim>(dim + 1), radius(radius), length(length), pressure_drop(pressure_drop) {}

    template<int dim>
    void ExactSolution<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const {
        Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));
        // value = u_1, u_2, p
        values(0) = pressure_drop / (4 * length) * (radius * radius - p[1] * p[1]);
        values(1) = 0;
        values(2) = 0;
    }


    template<int dim>
    class StokesError : public StokesNitsche<dim> {
    public:
        StokesError(const unsigned int degree, ErrorRightHandSide<dim> &rhs, ErrorBoundaryValues<dim> &bdd_val);

        void make_grid();

        void run();

    private:
        double get_pressure_difference() const;

        void compute_errors() const;

        double left_boundary;
        double radius;
    };

    template<int dim>
    StokesError<dim>::StokesError(const unsigned int degree, ErrorRightHandSide<dim> &rhs,
                                  ErrorBoundaryValues<dim> &bdd_val)
            : StokesNitsche<dim>(degree, rhs, bdd_val) {
        // TODO ta inn hvilke boundary_ids som skal ignoreres for Do Nothing bdd conditions. Gjør dette i base klassen.
        // TODO finn ut hvordan man endrer type på objektene right_hand_side og boundary_values til de som er
        //  spesifisert lenger opp i fila. Se c++ boka.
        // TODO sett eget navn på fila som skrives her.
        left_boundary = bdd_val.left_boundary;
        radius = bdd_val.radius;
    }

    template<int dim>
    void StokesError<dim>::make_grid() {
        std::cout << "StokesError::makegrid" << std::endl;

        GridGenerator::cylinder(StokesNitsche<dim>::triangulation, 0.205, -left_boundary);
        GridTools::remove_anisotropy(StokesNitsche<dim>::triangulation, 1.618, 5);
        StokesNitsche<dim>::triangulation.refine_global(dim == 2 ? 4 : 0);
    }

    template<int dim>
    double StokesError<dim>::get_pressure_difference() const {
        int count = 0;
        double max = StokesNitsche<dim>::solution[0];
        double min = StokesNitsche<dim>::solution[0];
        // TODO fix, denne går gjennom både trykk og hastighet
        for (double value : StokesNitsche<dim>::solution) {
            ++count;
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }
        std::cout << "min: " << min << std::endl;
        std::cout << "diff: " << max - min << std::endl;

        return max - min;
    }

    template<int dim>
    void StokesError<dim>::compute_errors() const {

        const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                         dim + 1);
        double pressure_drop = this->get_pressure_difference();  // TODO Calc from pressure component?

        double length = -2 * left_boundary;
        std::cout << "length: " << length << std::endl;
        std::cout << "radius: " << radius << std::endl;
        std::cout << "presure drop: " << pressure_drop << std::endl;
        ExactSolution<dim> exact_solution(radius, length, pressure_drop);

        Vector<double> cellwise_errors(StokesNitsche<dim>::triangulation.n_active_cells());
        QTrapez<1> q_trapez;
        QIterated<dim> quadrature(q_trapez, StokesNitsche<dim>::degree + 2);

        VectorTools::integrate_difference(StokesNitsche<dim>::dof_handler,
                                          StokesNitsche<dim>::solution,
                                          ZeroFunction<dim>(dim + 1),
                                          cellwise_errors,
                                          quadrature,
                                          VectorTools::L2_norm,
                                          &velocity_mask);
        const double u_l2 = VectorTools::compute_global_error(StokesNitsche<dim>::triangulation,
                                                              cellwise_errors,
                                                              VectorTools::L2_norm);
        VectorTools::integrate_difference(StokesNitsche<dim>::dof_handler,
                                          StokesNitsche<dim>::solution,
                                          exact_solution,
                                          cellwise_errors,
                                          quadrature,
                                          VectorTools::L2_norm,
                                          &velocity_mask);
        const double u_l2_error = VectorTools::compute_global_error(StokesNitsche<dim>::triangulation,
                                                                    cellwise_errors,
                                                                    VectorTools::L2_norm);

        std::cout << "  Integral: ||e_p||_L2 = " << u_l2 << std::endl;
        std::cout << "  Errors: ||e_p||_L2 = " << u_l2_error << std::endl;
    }

    template<int dim>
    void StokesError<dim>::run() {
        StokesNitsche<dim>::run();
        compute_errors();
    }

}


int main() {
    using namespace Error;

    std::cout << "StokesNitsche" << std::endl;

    const int dim = 2;
    double left_boundary = -1;
    double radius = 0.21;
    ErrorRightHandSide<dim> error_rhs;
    ErrorBoundaryValues<dim> error_bdd(left_boundary, radius);

    StokesError<dim> stokes_error(1, error_rhs, error_bdd);
    stokes_error.run();

    std::cout << "main error" << std::endl;

    return 0;
}