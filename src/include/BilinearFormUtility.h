#ifndef BILINEAR_FORM_UTILITY
#define BILINEAR_FORM_UTILITY

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#include "DofOrdering.h"
#include "BF.h"
#include "CamelliaIntrepidExtendedTypes.h"
#include "IP.h"
#include "Mesh.h"
#include "RHS.h"
#include "BasisCache.h"

// Shards includes
#include "Shards_CellTopology.hpp"

using namespace std;

namespace Camellia
{
template <typename Scalar>
class BilinearFormUtility
{
private:
  static bool _warnAboutZeroRowsAndColumns;
public:
  // the "pre-stiffness" (rectangular) matrix methods:
  static void computeStiffnessMatrix(Intrepid::FieldContainer<Scalar> &stiffness, TBFPtr<Scalar> bilinearForm,
                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                     CellTopoPtr cellTopo, Intrepid::FieldContainer<double> &physicalCellNodes,
                                     Intrepid::FieldContainer<double> &cellSideParities);

  // the real one:
  //  static void computeStiffnessMatrix(Intrepid::FieldContainer<double> &stiffness, BFPtr bilinearForm,
  //                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
  //                                     Intrepid::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);

  static void computeStiffnessMatrixForCell(Intrepid::FieldContainer<Scalar> &stiffness, Teuchos::RCP<Mesh> mesh, int cellID);

  static void computeStiffnessMatrix(Intrepid::FieldContainer<Scalar> &stiffness, Intrepid::FieldContainer<Scalar> &innerProductMatrix,
                                     Intrepid::FieldContainer<Scalar> &optimalTestWeights);

  // this method is deprecated; use the next one
  static void computeRHS(Intrepid::FieldContainer<Scalar> &rhsVector, TBFPtr<Scalar> bilinearForm, RHS &rhs,
                         Intrepid::FieldContainer<Scalar> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
                         shards::CellTopology &cellTopo, Intrepid::FieldContainer<double> &physicalCellNodes);

  //  static void computeRHS(Intrepid::FieldContainer<double> &rhsVector, BFPtr bilinearForm, RHS &rhs,
  //                  Intrepid::FieldContainer<double> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
  //                  BasisCachePtr basisCache);

  static void transposeFCMatrices(Intrepid::FieldContainer<Scalar> &fcTranspose,
                                  const Intrepid::FieldContainer<Scalar> &fc);

  static bool checkForZeroRowsAndColumns(string name, Intrepid::FieldContainer<Scalar> &array, bool checkRows = true, bool checkCols = true);

  static void weightCellBasisValues(Intrepid::FieldContainer<double> &basisValues,
                                    const Intrepid::FieldContainer<double> &weights, int offset);

  static void setWarnAboutZeroRowsAndColumns( bool value );
  static bool warnAboutZeroRowsAndColumns();
};

extern template class BilinearFormUtility<double>;
}
#endif
