#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>


#include "nitsche_stokes.h"

using namespace dealii;

namespace Error {
    using namespace Stokes;

    template<int dim>
    class ErrorRightHandSide : public RightHandSide<dim> {
    public:
        ErrorRightHandSide(double length, double pressure_drop);

        double point_value(const Point<dim> &p, const unsigned int component = 0) const;

        double length;
        double pressure_drop;
    };

    template<int dim>
    ErrorRightHandSide<dim>::ErrorRightHandSide(double length, double pressure_drop)
            : length(length), pressure_drop(pressure_drop) {}


    template<int dim>
    double ErrorRightHandSide<dim>::point_value(const Point<dim> &p, const unsigned int component) const {
        (void) p;
        if (component == 0) {
            return -0.5 * pressure_drop / length;
        }
        return 0;
    }

    template<int dim>
    class ErrorBoundaryValues : public BoundaryValues<dim> {
    public:
        ErrorBoundaryValues(double left_boundary, double radius, double pressure_drop);

        virtual double point_value(const Point<dim> &p, const unsigned int component) const;

        double left_boundary;  // x-coordinate of left side of the domain.
        double radius;         // radius of the cylinder
        double pressure_drop;
    };


    template<int dim>
    ErrorBoundaryValues<dim>::ErrorBoundaryValues(double left_boundary, double radius, double pressure_drop)
            : left_boundary(left_boundary), radius(radius), pressure_drop(pressure_drop) {}

    template<int dim>
    double ErrorBoundaryValues<dim>::point_value(const Point<dim> &p, const unsigned int component) const {
        (void) p;
        double length = -2 * left_boundary;
        // TODO check boundary id instead of coordinate for boundary
        if (component == 0 && p[0] == left_boundary) {
            if (dim == 2) {
                return pressure_drop / (4 * length) * (radius * radius - p[1] * p[1]);
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
        StokesError(const unsigned int degree, ErrorRightHandSide<dim> &rhs, ErrorBoundaryValues<dim> &bdd_val,
                    unsigned int do_nothing_bdd_id);

        void make_grid();

        void run();

    private:
        void compute_errors() const;

        double left_boundary;
        double radius;
        double pressure_drop;
    };

    template<int dim>
    StokesError<dim>::StokesError(const unsigned int degree, ErrorRightHandSide<dim> &rhs,
                                  ErrorBoundaryValues<dim> &bdd_val, const unsigned int do_nothing_bdd_id)
            : StokesNitsche<dim>(degree, rhs, bdd_val, do_nothing_bdd_id) {
        // TODO ta inn hvilke boundary_ids som skal ignoreres for Do Nothing bdd conditions. Gjør dette i base klassen.
        // TODO sett eget navn på fila som skrives her.
        left_boundary = bdd_val.left_boundary;
        radius = bdd_val.radius;
        pressure_drop = bdd_val.pressure_drop;
    }

    template<int dim>
    void StokesError<dim>::make_grid() {
        GridGenerator::cylinder(this->triangulation, radius, -left_boundary);
        GridTools::remove_anisotropy(this->triangulation, 1.618, 5);
        this->triangulation.refine_global(dim == 2 ? 4 : 0);
    }

    template<int dim>
    void StokesError<dim>::compute_errors() const {
        const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                         dim + 1);
        double length = -2 * left_boundary;

        ExactSolution<dim> exact_solution(radius, length, pressure_drop);

        Vector<double> cellwise_errors(this->triangulation.n_active_cells());
        QTrapez<1> q_trapez;
        QIterated<dim> quadrature(q_trapez, this->degree + 2);

        VectorTools::integrate_difference(this->dof_handler,
                                          this->solution,
                                          exact_solution,
                                          cellwise_errors,
                                          quadrature,
                                          VectorTools::L2_norm,
                                          &velocity_mask);
        const double u_l2_error = VectorTools::compute_global_error(this->triangulation,
                                                                    cellwise_errors,
                                                                    VectorTools::L2_norm);

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

    std::cout << "StokesError" << std::endl;

    const int dim = 2;
    double left_boundary = -1;
    double radius = 0.21;
    double pressure_drop = 20;
    double length = -2 * left_boundary;
    ErrorRightHandSide<dim> error_rhs(length, pressure_drop);
    ErrorBoundaryValues<dim> error_bdd(left_boundary, radius, pressure_drop);

    StokesError<dim> stokes_error(1, error_rhs, error_bdd, 2);
    stokes_error.run();

    // TODO create loop for convergence plot
    return 0;
}