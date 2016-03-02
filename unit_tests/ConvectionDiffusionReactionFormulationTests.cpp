//
//  ConvectionDiffusionReactionFormulationTests
//  Camellia
//
//  Created by Nate Roberts on Mar 3, 2016
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "ConvectionDiffusionReactionFormulation.h"
#include "Function.h"
#include "MeshFactory.h"
#include "Solution.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
  void testSolve(int spaceDim, ConvectionDiffusionReactionFormulation::FormulationChoice formulationChoice,
                 Teuchos::FancyOStream &out, bool &success)
  {
    // solve for a polynomial manufactured solution:
    FunctionPtr x = Function::xn(1), y = Function::yn(1), z = Function::zn(1);
    FunctionPtr u_exact;
    int u_degree = 2;
    int beta_degree;
    // want beta to have zero divergence
    FunctionPtr beta;
    if (spaceDim == 1)
    {
      u_exact = x * x;
      beta = Function::constant(1.0);
      beta_degree = 0;
    }
    else if (spaceDim == 2)
    {
      u_exact = x * x + 2 * y * y;
      beta = Function::vectorize(y,x);
      beta_degree = 1;
    }
    else if (spaceDim == 3)
    {
      u_exact = x * x + 2 * y * y + 3 * z * z;
      beta = Function::constant({1,2,3});
      beta_degree = 0;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported spaceDim");
    }
    
    double alpha = 3.14159;
    double epsilon = 1e-3;
    ConvectionDiffusionReactionFormulation form(formulationChoice, spaceDim, beta, epsilon, alpha);
    
    FunctionPtr f = form.forcingFunction(u_exact);
    RHSPtr rhs = form.rhs(f);
    
    int H1Order = u_degree + beta_degree + 1; // discretization of flux involve beta * u
    int delta_k = 2;
    vector<double> meshDim(spaceDim,1.0);
    vector<int> elementWidths(spaceDim,1);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), meshDim, elementWidths, H1Order, delta_k);
    
    BCPtr bc = BC::bc();
    IPPtr ip;
    // enforce a Dirichlet BC everywhere
    if (formulationChoice == ConvectionDiffusionReactionFormulation::ULTRAWEAK)
    {
      bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), u_exact);
      ip = form.bf()->graphNorm();;
    }
    else if (formulationChoice == ConvectionDiffusionReactionFormulation::PRIMAL)
    {
      bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_exact);
      ip = form.bf()->naiveNorm(spaceDim);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported formulation choice");
    }
    
    SolutionPtr soln = Solution::solution(form.bf(), mesh, bc, rhs, ip);
    soln->setCubatureEnrichmentDegree(beta_degree); //set to match the degree of beta
    soln->solve();
    
    FunctionPtr u_soln = Function::solution(form.u(), soln);
    
    double tol = 5e-10; // the 3D solves do seem to want a looser tolerance (epsilon is fairly small, so maybe that's OK)
    double diff = (u_exact - u_soln)->l2norm(mesh);
    TEST_COMPARE(diff, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, ForcingFunction )
  {
    int spaceDim = 3;
    FunctionPtr x = Function::xn(1), y = Function::yn(1), z = Function::zn(1);
    FunctionPtr u_exact = x * x + 2 * y * y + 3 * z * z;
    FunctionPtr u_grad = Function::vectorize(2 * x, 4 * y, 6 * z);
    FunctionPtr u_laplacian = Function::constant(12.0);
    
    // want beta to have zero divergence
    FunctionPtr beta = Function::vectorize(y,z,x);
    double alpha = 3.14159;
    double epsilon = 1e-3;
    
    FunctionPtr f_expected = -epsilon * u_laplacian + beta * u_grad + alpha * u_exact;
    
    ConvectionDiffusionReactionFormulation form(ConvectionDiffusionReactionFormulation::ULTRAWEAK, spaceDim, beta, epsilon, alpha);
    
    FunctionPtr f_actual = form.forcingFunction(u_exact);
    
    // set up a single-element mesh for comparing f_actual with f_expected
    int H1Order = 2;
    int delta_k = 1;
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0,1.0}, {1,1,1}, H1Order, delta_k);
    double tol = 1e-15;
  
    double diff = (f_actual-f_expected)->l2norm(mesh);
    TEST_COMPARE(diff, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolvePrimal_1D )
  {
    int spaceDim = 1;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolvePrimal_2D )
  {
    int spaceDim = 2;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolvePrimal_3D_Slow )
  {
    int spaceDim = 3;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveUltraweak_1D )
  {
    int spaceDim = 1;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveUltraweak_2D )
  {
    int spaceDim = 2;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveUltraweak_3D_Slow )
  {
    int spaceDim = 3;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    testSolve(spaceDim, formulation, out, success);
  }
} // namespace
