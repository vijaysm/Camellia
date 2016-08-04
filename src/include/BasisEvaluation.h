// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

//
//  BasisEvaluation.h
//  DPGTrilinos
//

#ifndef DPGTrilinos_BasisEvaluation_h
#define DPGTrilinos_BasisEvaluation_h

#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "VectorizedBasis.h"
#include "Basis.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "CamelliaIntrepidExtendedTypes.h"

namespace Camellia
{
class BasisEvaluation
{
public:  
  static FCPtr getValues(BasisPtr basis, Camellia::EOperator op,
                         const Intrepid::FieldContainer<double> &refPoints);
  static FCPtr getTransformedValues(BasisPtr basis, Camellia::EOperator op,
                                    const Intrepid::FieldContainer<double> &refPoints,
                                    int numCells,
                                    BasisCache* basisCache);
  
  // ! Deprecated.  Will be deleted once MultiBasis no longer depends on this
  static FCPtr getTransformedValues(BasisPtr basis, Camellia::EOperator op,
                                    const Intrepid::FieldContainer<double> &refPoints,
                                    int numCells,
                                    const Intrepid::FieldContainer<double> &cellJacobian,
                                    const Intrepid::FieldContainer<double> &cellJacobianInv,
                                    const Intrepid::FieldContainer<double> &cellJacobianDet);
  static FCPtr getTransformedVectorValuesWithComponentBasisValues(Camellia::VectorBasisPtr basis,
                                                                  Camellia::EOperator op,
                                                                  constFCPtr componentReferenceValuesTransformed);
  static FCPtr getTransformedValuesWithBasisValues(BasisPtr basis, Camellia::EOperator op, int spaceDim,
                                                   constFCPtr referenceValues, int numCells,
                                                   const Intrepid::FieldContainer<double> &cellJacobian,
                                                   const Intrepid::FieldContainer<double> &cellJacobianInv,
                                                   const Intrepid::FieldContainer<double> &cellJacobianDet);
  static FCPtr getTransformedValuesWithBasisValues(BasisPtr basis, Camellia::EOperator op,
                                                   constFCPtr referenceValues, int numCells,
                                                   BasisCache* basisCache);
  static FCPtr getValuesCrossedWithNormals(constFCPtr values,const Intrepid::FieldContainer<double> &sideNormals);
  static FCPtr getValuesDottedWithNormals(constFCPtr values,const Intrepid::FieldContainer<double> &sideNormals);
  static FCPtr getValuesTimesNormals(constFCPtr values,const Intrepid::FieldContainer<double> &sideNormals);
  static FCPtr getValuesTimesNormals(constFCPtr values,const Intrepid::FieldContainer<double> &sideNormals, int normalComponent);
  static FCPtr getVectorizedValues(constFCPtr values, int spaceDim);
  static Intrepid::EOperator relatedOperator(Camellia::EOperator op, Camellia::EFunctionSpace fs, int spaceDim, int &componentOfInterest);
  static FCPtr getComponentOfInterest(constFCPtr values, Camellia::EOperator op, Camellia::EFunctionSpace fs, int componentOfInterest);
};
}

#endif
