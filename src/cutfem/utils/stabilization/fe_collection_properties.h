#ifndef INCLUDE_CUTFEM_FE_COLLECTION_PROPERTIES_H_
#define INCLUDE_CUTFEM_FE_COLLECTION_PROPERTIES_H_

#include <deal.II/hp/fe_collection.h>

#include <algorithm>

namespace cutfem
{
  using namespace dealii;

  template <int dim>
  unsigned int
  get_max_element_order(const hp::FECollection<dim> &fe_collection)
  {
    Assert(fe_collection.size() > 0, ExcInternalError());
    unsigned int max_order = fe_collection[0].tensor_degree();
    for (unsigned int i = 1; i < fe_collection.size(); ++i)
      {
        max_order = std::max(max_order, fe_collection[i].tensor_degree());
      }
    return max_order;
  }

  template <int dim>
  unsigned int
  get_min_positive_element_order(const hp::FECollection<dim> &fe_collection)
  {
    Assert(fe_collection.size() > 0, ExcInternalError());
    unsigned int min_order = get_max_element_order(fe_collection);
    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      {
        const unsigned int order = fe_collection[i].tensor_degree();
        if (order > 0)
          {
            min_order = std::min(min_order, order);
          }
      }
    Assert(min_order > 0, ExcInternalError());
    return min_order;
  }

} // namespace cutfem

#endif /* INCLUDE_CUTFEM_FE_COLLECTION_PROPERTIES_H_ */
