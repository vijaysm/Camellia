//
//  ConvectionDiffusionReactionFormulationTests
//  Camellia
//
//  Created by Nate Roberts on Mar 3, 2016
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "ConvectionDiffusionReactionFormulation.h"
#include "Function.h"
#include "MeshFactory.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
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
} // namespace
