#ifndef DPG_EXACT_SOLUTION
#define DPG_EXACT_SOLUTION

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

/*
 *  ExactSolution.h
 *
 *  Created by Nathan Roberts on 7/5/11.
 */

#include "TypeDefs.h"

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "RHS.h"
#include "BC.h"

#include "Solution.h"

#include "BF.h"
#include "BasisCache.h"

namespace Camellia
{
template <typename Scalar>
class ExactSolution
{
protected:
  TBFPtr<Scalar> _bilinearForm;
  TBCPtr<Scalar> _bc;
  TRHSPtr<Scalar> _rhs;
  // TODO: Fix this for complex (use norm)
  void squaredDifference(Intrepid::FieldContainer<double> &diffSquared, Intrepid::FieldContainer<Scalar> &values1, Intrepid::FieldContainer<Scalar> &values2);

  int _H1Order;
  map< int, TFunctionPtr<Scalar> > _exactFunctions; // var ID --> function
public:
  ExactSolution();
  ExactSolution(TBFPtr<Scalar> bf, TBCPtr<Scalar> bc, TRHSPtr<Scalar> rhs, int H1Order = -1);
  TBFPtr<Scalar> bilinearForm();
  TBCPtr<Scalar> bc();
  TRHSPtr<Scalar> rhs();
  const map< int, TFunctionPtr<Scalar> > exactFunctions(); // not supported by legacy subclasses
  virtual bool functionDefined(int trialID); // not supported by legacy subclasses
  void setSolutionFunction( VarPtr var, TFunctionPtr<Scalar> varFunction );
  void solutionValues(Intrepid::FieldContainer<Scalar> &values, int trialID, BasisCachePtr basisCache);
  void solutionValues(Intrepid::FieldContainer<Scalar> &values,
                      int trialID,
                      Intrepid::FieldContainer<double> &physicalPoints);
  void solutionValues(Intrepid::FieldContainer<Scalar> &values,
                      int trialID,
                      Intrepid::FieldContainer<double> &physicalPoints,
                      Intrepid::FieldContainer<double> &unitNormals);
  virtual Scalar solutionValue(int trialID,
                               Intrepid::FieldContainer<double> &physicalPoint);
  virtual Scalar solutionValue(int trialID,
                               Intrepid::FieldContainer<double> &physicalPoint,
                               Intrepid::FieldContainer<double> &unitNormal);
  virtual int H1Order(); // return -1 for non-polynomial solutions
  // TODO: Fix this for complex
  double L2NormOfError(TSolutionPtr<Scalar> solution, int trialID, int cubDegree=-1);
  void L2NormOfError(Intrepid::FieldContainer<double> &errorSquaredPerCell, TSolutionPtr<Scalar> solution, ElementTypePtr elemTypePtr, int trialID, int sideIndex=VOLUME_INTERIOR_SIDE_ORDINAL, int cubDegree=-1, double solutionLift=0.0);

  virtual ~ExactSolution() {}
};

extern template class ExactSolution<double>;
}

#endif
