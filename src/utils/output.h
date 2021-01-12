#ifndef MICROBUBBLE_OUTPUT_H
#define MICROBUBBLE_OUTPUT_H


#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/hp/fe_collection.h>


using namespace dealii;

namespace Utils {

    /**
     * This is a wrapper class for easier interpolation of the analytical
     * solution from a TensorFunction for the velocity, and a Function for the
     * pressure. These are combined in this class as a Function of dim + 1
     * components.
     */
    template<int dim>
    class AnalyticalSolutionWrapper : public Function<dim> {

    public:
        AnalyticalSolutionWrapper(TensorFunction<1, dim> &analytical_velocity,
                                  Function<dim> &analytical_pressure);

        double
        value(const Point<dim> &p, const unsigned int component) const override;

    private:
        TensorFunction<1, dim> *velocity;
        Function<dim> *pressure;
    };


    template<int dim>
    AnalyticalSolutionWrapper<dim>::
    AnalyticalSolutionWrapper(
            TensorFunction<1, dim> &analytical_velocity,
            Function<dim> &analytical_pressure)
            : Function<dim>(dim + 1) {
        velocity = &analytical_velocity;
        pressure = &analytical_pressure;
    }


    template<int dim>
    double AnalyticalSolutionWrapper<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        if (component < dim) {
            Tensor<1, dim> value = velocity->value(p);
            return value[component];
        } else {
            return pressure->value(p);
        }
    }


    /**
     * Write the numerical solution to the supplied file object.
     */
    template<int dim>
    void writeNumericalSolution(const DoFHandler<dim> &dof,
                                const Vector<double> &solution,
                                std::ofstream &file) {
        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(
                dim, DataComponentInterpretation::component_is_part_of_vector);
        dci.push_back(DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof);
        data_out.add_data_vector(solution,
                                 solution_names,
                                 DataOut<dim>::type_dof_data,
                                 dci);
        data_out.build_patches();
        data_out.write_vtk(file);
    }


    /**
     * Write the analytical solution, and the difference between the analytical
     * and numerical solution to vtk file.
     */
    template<int dim>
    void writeAnalyticalSolutionAndDiff(
            const DoFHandler<dim> &dof,
            const hp::FECollection<dim> &fe_collection,
            const Vector<double> &solution,
            TensorFunction<1, dim> &analytical_velocity,
            Function<dim> &analytical_pressure,
            std::ofstream &file_analytical,
            std::ofstream &file_diff) {
        // A few things needed for naming the output as vectors and
        // scalars. See step-22.
        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(
                dim, DataComponentInterpretation::component_is_part_of_vector);
        dci.push_back(DataComponentInterpretation::component_is_scalar);

        // A wrapper function to combine the analytical solution for the
        // velocity and the pressure in one Function object with dim + 1
        // components.
        AnalyticalSolutionWrapper<dim> wrapper(analytical_velocity,
                                               analytical_pressure);

        // TODO note that the pressure is interpolated as is. "zero-mean" is
        //  not asserted .
        // Vector for interpolating the analytical solution into
        Vector<double> vector(solution.size());
        VectorTools::interpolate(dof, wrapper, vector);

        // Write the analytical solution to vtk.
        DataOut<dim> out;
        out.attach_dof_handler(dof);
        out.add_data_vector(vector,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            dci);
        out.build_patches();
        out.write_vtk(file_analytical);

        // Calculate the difference between analytical and numerical solution,
        // and write this to file too.
        vector -= solution;
        out.clear_data_vectors();
        out.add_data_vector(vector,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            dci);
        out.build_patches();
        out.write_vtk(file_diff);
    }

} // namespace Utils

#endif //MICROBUBBLE_OUTPUT_H
