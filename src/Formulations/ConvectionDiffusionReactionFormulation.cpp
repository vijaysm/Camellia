//
//  ConvectionDiffusionReactionFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts Mar 3, 2016.
//
//

#include "ConvectionDiffusionReactionFormulation.h"

using namespace Camellia;

const string ConvectionDiffusionReactionFormulation::S_U = "u";
const string ConvectionDiffusionReactionFormulation::S_SIGMA = "sigma";

const string ConvectionDiffusionReactionFormulation::S_UHAT = "uhat";
const string ConvectionDiffusionReactionFormulation::S_SIGMA_N = "sigma_n";

const string ConvectionDiffusionReactionFormulation::S_V = "v";
const string ConvectionDiffusionReactionFormulation::S_TAU = "tau";

ConvectionDiffusionReactionFormulation::ConvectionDiffusionReactionFormulation(FormulationChoice formulation, int spaceDim,
                                                                               FunctionPtr beta, double epsilon, double alpha)
{
  _formulationChoice = formulation;
  _spaceDim = spaceDim;
  _beta = beta;
  _epsilon = epsilon;
  _alpha = alpha;

  Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
  Space uhat_space = HGRAD;
  Space sigmaSpace = (spaceDim > 1) ? VECTOR_L2 : L2;

  // fields
  VarPtr u;
  VarPtr sigma;

  // traces
  VarPtr uhat, sigma_n; // sigma_n is the normal trace of the whole thing under the divergence operator, not just sigma

  // tests
  VarPtr v;
  VarPtr tau;

  _vf = VarFactory::varFactory();

  TFunctionPtr<double> n = TFunction<double>::normal();
  TFunctionPtr<double> parity = TFunction<double>::sideParity();

  double sqrt_epsilon = sqrt(_epsilon);
  switch (_formulationChoice)
  {
    case ULTRAWEAK:
      u = _vf->fieldVar(S_U);
      sigma = _vf->fieldVar(S_SIGMA, sigmaSpace);
      
      if (spaceDim > 1)
        uhat = _vf->traceVar(S_UHAT, u, uhat_space);
      else
        uhat = _vf->fluxVar(S_UHAT, u, uhat_space); // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
      
      if (spaceDim > 1)
        sigma_n = _vf->fluxVar(S_SIGMA_N, (_beta * u-sigma) * (n * parity));
      else
        sigma_n = _vf->fluxVar(S_SIGMA_N, _beta * u-sigma);
      
      v = _vf->testVar(S_V, HGRAD);
      tau = _vf->testVar(S_TAU, tauSpace);
      
      _bf = Teuchos::rcp( new BF(_vf) );
      
      if (spaceDim==1)
      {
        // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
        _bf->addTerm(sigma, tau);
        _bf->addTerm(sqrt_epsilon * u, tau->dx());
        _bf->addTerm(-sqrt_epsilon * uhat, tau);
        
        _bf->addTerm(sqrt_epsilon * sigma - _beta * u, v->dx());
        _bf->addTerm(- sigma_n, v);
        _bf->addTerm(_alpha * u, v);
      }
      else
      {
        _bf->addTerm(sigma, tau);
        _bf->addTerm(sqrt_epsilon * u, tau->div());
        _bf->addTerm(-sqrt_epsilon * uhat, tau->dot_normal());
        
        _bf->addTerm(sqrt_epsilon * sigma -_beta * u, v->grad());
        _bf->addTerm(- sigma_n, v);
        _bf->addTerm(_alpha * u, v);
      }
      break;
    case PRIMAL:
      u = _vf->fieldVar(S_U, HGRAD);
      if (spaceDim == 1)
        sigma_n = _vf->fluxVar(S_SIGMA_N, (_beta * u - _epsilon * u->dx()));
      else
        sigma_n = _vf->fluxVar(S_SIGMA_N, (_beta * u - _epsilon * u->grad()) * (n * parity));
      
      v = _vf->testVar(S_V, HGRAD);
      
      _bf = Teuchos::rcp( new BF(_vf) );
      
      if (spaceDim==1)
      {
        // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
        _bf->addTerm(_epsilon * u->dx() - _beta * u, v->dx());
        _bf->addTerm(- sigma_n, v);
        _bf->addTerm(_alpha * u, v);
      }
      else
      {
        _bf->addTerm(_epsilon * u->grad() - _beta * u, v->grad());
        _bf->addTerm(-sigma_n, v);
        _bf->addTerm(_alpha * u, v);
      }
      break;
    case SUPG:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "SUPG formulation is not yet implemented");
  }
}

FunctionPtr ConvectionDiffusionReactionFormulation::forcingFunction(FunctionPtr u_exact)
{
  // -epsilon \Delta u + div (beta * u) + alpha u
  return -_epsilon * u_exact->grad(_spaceDim)->div() + (_beta * u_exact)->div() + _alpha * u_exact;
}

VarFactoryPtr ConvectionDiffusionReactionFormulation::vf()
{
  return _vf;
}

BFPtr ConvectionDiffusionReactionFormulation::bf()
{
  return _bf;
}

// field variables:
VarPtr ConvectionDiffusionReactionFormulation::u()
{
  return _vf->fieldVar(S_U);
}

VarPtr ConvectionDiffusionReactionFormulation::sigma()
{
  return _vf->fieldVar(S_SIGMA);
}

// traces:
VarPtr ConvectionDiffusionReactionFormulation::sigma_n()
{
  return _vf->fluxVar(S_SIGMA_N);
}

VarPtr ConvectionDiffusionReactionFormulation::uhat()
{
  return _vf->traceVar(S_UHAT);
}

// test variables:
VarPtr ConvectionDiffusionReactionFormulation::v()
{
  return _vf->testVar(S_V, HGRAD);
}

VarPtr ConvectionDiffusionReactionFormulation::tau()
{
  if (_spaceDim > 1)
    return _vf->testVar(S_TAU, HDIV);
  else
    return _vf->testVar(S_TAU, HGRAD);
}