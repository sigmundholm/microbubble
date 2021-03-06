#include "StokesCylinder.h"

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/non_matching/cut_mesh_classifier.h>
#include <deal.II/non_matching/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/optional.hpp>

#include <cmath>
#include <fstream>

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"


using namespace cutfem;


template<int dim>
StokesCylinder<dim>::StokesCylinder(const double radius,
                                    const double half_length,
                                    const unsigned int n_refines,
                                    const int element_order,
                                    const bool write_output,
                                    StokesRhs<dim> &rhs,
                                    BoundaryValues<dim> &bdd_values,
                                    const double sphere_radius,
                                    const double sphere_x_coord)
        : radius(radius), half_length(half_length), n_refines(n_refines),
          write_output(write_output), sphere_radius(sphere_radius),
          sphere_x_coord(sphere_x_coord),
          element_order(element_order),
          stokes_fe(FESystem<dim>(FE_Q<dim>(element_order + 1), dim),
                    1,
                    FE_Q<dim>(element_order),
                    1), fe_levelset(element_order),
          levelset_dof_handler(triangulation), dof_handler(triangulation),
          cut_mesh_classifier(triangulation, levelset_dof_handler, levelset) {
    h = 0;
    // Use no constraints when projecting.
    constraints.close();

    rhs_function = &rhs;
    boundary_values = &bdd_values;

    if (dim == 2) {
        this->center = Point<dim>(sphere_x_coord, 0);
    } else if (dim == 3) {
        this->center = Point<dim>(sphere_x_coord, 0, 0);
    }
}

template<int dim>
void
StokesCylinder<dim>::setup_quadrature() {
    const unsigned int quadOrder = 2 * element_order + 1;
    q_collection.push_back(QGauss<dim>(quadOrder));
    q_collection1D.push_back(QGauss<1>(quadOrder));
}

template<int dim>
void
StokesCylinder<dim>::run() {
    make_grid();
    setup_quadrature();
    setup_level_set();
    cut_mesh_classifier.reclassify();
    distribute_dofs();
    initialize_matrices();
    assemble_system();
    solve();
    if (write_output) {
        output_results();
    }
}

template<int dim>
void
StokesCylinder<dim>::make_grid() {
    std::cout << "Creating triangulation" << std::endl;

    GridGenerator::cylinder(triangulation, radius, half_length);
    GridTools::remove_anisotropy(triangulation, 1.618, 5);
    triangulation.refine_global(n_refines);

    mapping_collection.push_back(MappingCartesian<dim>());

    // Save the cell-size, we need it in the Nitsche term.
    typename Triangulation<dim>::active_cell_iterator cell =
            triangulation.begin_active();
    h = std::pow(cell->measure(), 1.0 / dim);
}

template<int dim>
void
StokesCylinder<dim>::setup_level_set() {
    std::cout << "Setting up level set" << std::endl;

    // The level set function lives on the whole background mesh.
    levelset_dof_handler.distribute_dofs(fe_levelset);
    printf("leveset dofs: %d\n", levelset_dof_handler.n_dofs());
    levelset.reinit(levelset_dof_handler.n_dofs());

    // Project the geometry onto the mesh.
    cutfem::geometry::SignedDistanceSphere<dim> signed_distance_sphere(
            sphere_radius, center, -1);
    VectorTools::project(levelset_dof_handler,
                         constraints,
                         QGauss<dim>(2 * element_order + 1),
                         signed_distance_sphere,
                         levelset);
}

template<int dim>
void
StokesCylinder<dim>::distribute_dofs() {
    std::cout << "Distributing dofs" << std::endl;

    // We want to types of elements on the mesh
    // Lagrange elements and elements that are constant zero..
    fe_collection.push_back(stokes_fe);
    fe_collection.push_back(FESystem<dim>(
            FESystem<dim>(FE_Nothing<dim>(), dim), 1, FE_Nothing<dim>(), 1));

    // Set outside finite elements to stokes_fe, and inside to FE_nothing
    for (const auto &cell : dof_handler.active_cell_iterators()) {
        if (LocationToLevelSet::OUTSIDE ==
            cut_mesh_classifier.location_to_level_set(cell)) {
            // 1 is FE_nothing
            cell->set_active_fe_index(1);
        } else {
            // 0 is stokes_fe
            cell->set_active_fe_index(0);
        }
    }
    dof_handler.distribute_dofs(fe_collection);
}

template<int dim>
void
StokesCylinder<dim>::initialize_matrices() {
    std::cout << "Initialize marices" << std::endl;
    solution.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());

    cutfem::nla::make_sparsity_pattern_for_stabilized(dof_handler,
                                                      sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
}

template<int dim>
void
StokesCylinder<dim>::assemble_system() {
    std::cout << "Assembling" << std::endl;

    stiffness_matrix = 0;
    rhs = 0;

    // Use a helper object to compute the stabilisation for both the velocity
    // and the pressure component.
    // TODO ta ut stabiliseringen i en egen funksjon?
    const FEValuesExtractors::Vector velocities(0);
    stabilization::JumpStabilization<dim, FEValuesExtractors::Vector>
            velocity_stabilization(dof_handler,
                                   mapping_collection,
                                   cut_mesh_classifier,
                                   constraints);
    velocity_stabilization.set_function_describing_faces_to_stabilize(
            stabilization::inside_stabilization);
    velocity_stabilization.set_weight_function(stabilization::taylor_weights);
    velocity_stabilization.set_extractor(velocities);

    const FEValuesExtractors::Scalar pressure(dim);
    stabilization::JumpStabilization<dim, FEValuesExtractors::Scalar>
            pressure_stabilization(dof_handler,
                                   mapping_collection,
                                   cut_mesh_classifier,
                                   constraints);
    pressure_stabilization.set_function_describing_faces_to_stabilize(
            stabilization::inside_stabilization);
    pressure_stabilization.set_weight_function(stabilization::taylor_weights);
    pressure_stabilization.set_extractor(pressure);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_JxW_values |
                                 update_gradients | update_quadrature_points;
    region_update_flags.surface = update_values | update_JxW_values |
                                  update_gradients | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> cut_fe_values(mapping_collection,
                                             fe_collection,
                                             q_collection,
                                             q_collection1D,
                                             region_update_flags,
                                             cut_mesh_classifier,
                                             levelset_dof_handler,
                                             levelset);

    // Quadrature for the faces of the cells on the outer boundary
    QGauss<dim - 1> face_quadrature_formula(stokes_fe.degree + 1);
    FEFaceValues<dim> fe_face_values(stokes_fe,
                                     face_quadrature_formula,
                                     update_values | update_gradients |
                                     update_quadrature_points |
                                     update_normal_vectors | update_JxW_values);

    double beta_0 = 0.1;
    double gamma_A = beta_0 * pow(element_order, 2);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
        std::vector<types::global_dof_index> loc2glb(n_dofs);
        cell->get_dof_indices(loc2glb);

        // This call will compute quadrature rules relevant for this cell
        // in the background.
        cut_fe_values.reinit(cell);

        // Retrieve an FEValues object with quadrature points
        // over the full cell.
        const boost::optional<const FEValues<dim> &> fe_values_bulk =
                cut_fe_values.get_inside_fe_values();

        if (fe_values_bulk) {
            assemble_local_over_bulk(*fe_values_bulk, loc2glb);
        }

        // Loop through all faces that constitutes the outer boundary of the
        // domain.
        for (const auto &face : cell->face_iterators()) {
            if (face->at_boundary() && face->boundary_id() != do_nothing_id) {
                fe_face_values.reinit(cell, face);
                assemble_local_over_surface(fe_face_values, loc2glb);
            }
        }

        // Retrieve an FEValues object with quadrature points
        // on the immersed surface.
        const boost::optional<const FEImmersedSurfaceValues<dim> &>
                fe_values_surface = cut_fe_values.get_surface_fe_values();

        if (fe_values_surface)
            assemble_local_over_surface(*fe_values_surface, loc2glb);

        // Compute and add the velocity stabilization.
        velocity_stabilization.compute_stabilization(cell);
        velocity_stabilization.add_stabilization_to_matrix(gamma_A / (h * h),
                                                           stiffness_matrix);
        // Compute and add the pressure stabilisation.
        pressure_stabilization.compute_stabilization(cell);
        pressure_stabilization.add_stabilization_to_matrix(-gamma_A,
                                                           stiffness_matrix);
    }
}

template<int dim>
void
StokesCylinder<dim>::assemble_local_over_bulk(
        const FEValues<dim> &fe_values,
        const std::vector<types::global_dof_index> &loc2glb) {
    // TODO generelt: er det for mange hjelpeobjekter som opprettes her i cella?
    //  bør det heller gjøres i funksjonen før og sendes som argumenter? hvis
    //  det er mulig mtp cellene som blir cut da

    // Matrix and vector for the contribution of each cell
    const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    // Vector for values of the RightHandSide for all quadrature points on a cell.
    std::vector<Tensor<1, dim>> rhs_values(fe_values.n_quadrature_points,
                                           Tensor<1, dim>());
    rhs_function->value_list(fe_values.get_quadrature_points(), rhs_values);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    // Calculate often used terms in the beginning of each cell-loop
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());
    std::vector<double> phi_p(dofs_per_cell);

    for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
        for (const unsigned int k : fe_values.dof_indices()) {
            grad_phi_u[k] = fe_values[velocities].gradient(k, q);
            div_phi_u[k] = fe_values[velocities].divergence(k, q);
            phi_u[k] = fe_values[velocities].value(k, q);
            phi_p[k] = fe_values[pressure].value(k, q);
        }

        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                local_matrix(i, j) +=
                        (scalar_product(grad_phi_u[j],
                                        grad_phi_u[i]) // (grad u, grad v)
                         - (div_phi_u[i] * phi_p[j])   // -(div v, p)
                         - (div_phi_u[j] * phi_p[i])   // -(div u, q)
                        ) *
                        fe_values.JxW(q); // dx
            }
            // RHS
            local_rhs(i) += rhs_values[q] * phi_u[i] // (f, v)
                            * fe_values.JxW(q);      // dx
        }
    }
    stiffness_matrix.add(loc2glb, local_matrix);
    rhs.add(loc2glb, local_rhs);
}


template<int dim>
void
StokesCylinder<dim>::assemble_local_over_surface(
        const FEValuesBase<dim> &fe_values,
        const std::vector<types::global_dof_index> &loc2glb) {
    // Matrix and vector for the contribution of each cell
    const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    // Evaluate the boundary function for all quadrature points on this face.
    std::vector<Tensor<1, dim>> bdd_values(fe_values.n_quadrature_points,
                                           Tensor<1, dim>());
    boundary_values->value_list(fe_values.get_quadrature_points(), bdd_values);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    // Calculate often used terms in the beginning of each cell-loop
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());
    std::vector<double> phi_p(dofs_per_cell);

    double mu = 50 / h; // Penalty parameter
    Tensor<1, dim> normal;

    for (unsigned int q : fe_values.quadrature_point_indices()) {
        normal = fe_values.normal_vector(q);

        for (const unsigned int k : fe_values.dof_indices()) {
            grad_phi_u[k] = fe_values[velocities].gradient(k, q);
            div_phi_u[k] = fe_values[velocities].divergence(k, q);
            phi_u[k] = fe_values[velocities].value(k, q);
            phi_p[k] = fe_values[pressure].value(k, q);
        }

        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                local_matrix(i, j) +=
                        (-(grad_phi_u[j] * normal) *
                         phi_u[i]  // -(n * grad u, v)
                         -
                         (grad_phi_u[i] * normal) * phi_u[j] // -(n * grad v, u)
                         + mu * (phi_u[j] * phi_u[i])          // mu (u, v)
                         + (normal * phi_u[i]) * phi_p[j]      // (n * v, p)
                         + (normal * phi_u[j]) * phi_p[i]      // (n * u, q)
                        ) *
                        fe_values.JxW(q); // ds
            }

            Tensor<1, dim> prod_r =
                    mu * phi_u[i] - grad_phi_u[i] * normal + phi_p[i] * normal;
            local_rhs(i) +=
                    prod_r * bdd_values[q] // (g, mu v - n grad v + q * n)
                    * fe_values.JxW(q);    // ds
        }
    }
    stiffness_matrix.add(loc2glb, local_matrix);
    rhs.add(loc2glb, local_rhs);
}


template<int dim>
void
StokesCylinder<dim>::solve() {
    std::cout << "Solving system" << std::endl;
    SparseDirectUMFPACK inverse;
    inverse.initialize(stiffness_matrix);
    inverse.vmult(solution, rhs);
}

template<int dim>
void
StokesCylinder<dim>::output_results() const {
    std::cout << "Output results" << std::endl;
    // Output results, see step-22
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(
            dim, DataComponentInterpretation::component_is_part_of_vector);
    dci.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             dci);

    data_out.build_patches();
    std::ofstream output("solution-d" + std::to_string(dim)
                         + "o" + std::to_string(element_order)
                         + "r" + std::to_string(n_refines) + ".vtk");
    data_out.write_vtk(output);

    // Output levelset function.
    DataOut<dim, DoFHandler<dim>> data_out_levelset;
    data_out_levelset.attach_dof_handler(levelset_dof_handler);
    data_out_levelset.add_data_vector(levelset, "levelset");
    data_out_levelset.build_patches();
    std::ofstream output_ls("levelset-d" + std::to_string(dim)
                            + "o" + std::to_string(element_order)
                            + "r" + std::to_string(n_refines) + ".vtk");
    data_out_levelset.write_vtk(output_ls);
}


template
class StokesCylinder<2>;

template
class StokesCylinder<3>;
