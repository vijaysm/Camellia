//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef DPG_POISSON_EXACT_SOLUTION
#define DPG_POISSON_EXACT_SOLUTION

/*
 *  PoissonExactSolution.h
 *
 */

#include "ExactSolution.h"
#include "BC.h"
#include "RHS.h"

#include "Function.h"

using namespace Camellia;

class PoissonExactSolution : public ExactSolution<double>
{
public:
  enum PoissonExactSolutionType
  {
    POLYNOMIAL = 0,
    EXPONENTIAL,
    TRIGONOMETRIC
  };
protected:
  int _polyOrder;
  PoissonExactSolutionType _type;

  BFPtr _bf;
  FunctionPtr phi();
public:
  PoissonExactSolution(PoissonExactSolutionType type, int polyOrder=-2, bool useConformingTraces=false); // poly order here means that of phi -- -2 for non-polynomial types
// ExactSolution
  virtual int H1Order(); // here it means the H1 order (i.e. polyOrder+1)
  virtual void setUseSinglePointBCForPHI(bool value, IndexType vertexIndexForZeroValue);

  std::vector<double> getPointForBCImposition();

  static Teuchos::RCP<ExactSolution<double>> poissonExactPolynomialSolution(int polyOrder, bool useConformingTraces = true);
};

#endif