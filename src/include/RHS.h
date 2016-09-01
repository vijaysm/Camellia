#ifndef DPG_RHS
#define DPG_RHS

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

/*
 *  RHS.h
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"

#include "BasisCache.h"

#include "LinearTerm.h"

#include <vector>

using namespace std;

namespace Camellia
{
template <typename Scalar>
class TRHS
{
  bool _legacySubclass;

  TLinearTermPtr<Scalar> _lt;
  set<int> _varIDs;
public:
  TRHS(bool legacySubclass) : _legacySubclass(legacySubclass) {}
  virtual bool nonZeroRHS(int testVarID);
  virtual vector<Camellia::EOperator> operatorsForTestID(int testID);
  // TODO: change the API here so that values is the first argument (fitting a convention in the rest of the code)
  virtual void rhs(int testVarID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache, Intrepid::FieldContainer<Scalar> &values);
  virtual void rhs(int testVarID, int operatorIndex, const Intrepid::FieldContainer<double> &physicalPoints, Intrepid::FieldContainer<Scalar> &values);
  virtual void rhs(int testVarID, const Intrepid::FieldContainer<double> &physicalPoints, Intrepid::FieldContainer<Scalar> &values);
  // physPoints (numCells,numPoints,spaceDim)
  // values: either (numCells,numPoints) or (numCells,numPoints,spaceDim)

  virtual void integrateAgainstStandardBasis(Intrepid::FieldContainer<Scalar> &rhsVector, Teuchos::RCP<DofOrdering> testOrdering,
      BasisCachePtr basisCache);
  virtual void integrateAgainstOptimalTests(Intrepid::FieldContainer<Scalar> &rhsVector, const Intrepid::FieldContainer<Scalar> &optimalTestWeights,
      Teuchos::RCP<DofOrdering> testOrdering, BasisCachePtr basisCache);

  void addTerm( TLinearTermPtr<Scalar> rhsTerm );
  void addTerm( VarPtr v );

  TLinearTermPtr<Scalar> linearTerm(); // MUTABLE reference (change this, RHS will change!)
  TLinearTermPtr<Scalar> linearTermCopy(); // copy of RHS as a LinearTerm

  virtual ~TRHS() {}

  static TRHSPtr<Scalar> rhs()
  {
    return Teuchos::rcp(new TRHS<Scalar>(false) );
  }
};

extern template class TRHS<double>;
}

#endif
