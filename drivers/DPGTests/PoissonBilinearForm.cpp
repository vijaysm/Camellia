//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#include "PoissonBilinearForm.h"

// trial variable names:
const string PoissonBilinearForm::S_PHI = "\\phi";
const string PoissonBilinearForm::S_PSI_1 = "\\psi_1";
const string PoissonBilinearForm::S_PSI_2 = "\\psi_2";
const string PoissonBilinearForm::S_PHI_HAT = "\\hat{\\phi}";
const string PoissonBilinearForm::S_PSI_HAT_N ="\\hat{\\psi}_n";

// test variable names:
const string PoissonBilinearForm::S_Q = "q";
const string PoissonBilinearForm::S_TAU = "\\tau";

BFPtr PoissonBilinearForm::poissonBilinearForm(bool useConformingTraces)
{
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr tau = varFactory->testVar(S_TAU, HDIV);
  VarPtr q = varFactory->testVar(S_Q, HGRAD);

  Space phiHatSpace = useConformingTraces ? HGRAD : L2;
  VarPtr phi_hat = varFactory->traceVar(S_PHI_HAT, phiHatSpace);
  //  VarPtr phi_hat = varFactory->traceVar(S_GDAMinimumRuleTests_PHI_HAT, L2);
  //  cout << "WARNING: temporarily using L^2 discretization for \\widehat{\\phi}.\n";
  VarPtr psi_n = varFactory->fluxVar(S_PSI_HAT_N);

  VarPtr phi = varFactory->fieldVar(S_PHI, L2);
  VarPtr psi1 = varFactory->fieldVar(S_PSI_1, L2);
  VarPtr psi2 = varFactory->fieldVar(S_PSI_2, L2);

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  bf->addTerm(phi, tau->div());
  bf->addTerm(psi1, tau->x());
  bf->addTerm(psi2, tau->y());
  bf->addTerm(-phi_hat, tau->dot_normal());

  bf->addTerm(-psi1, q->dx());
  bf->addTerm(-psi2, q->dy());
  bf->addTerm(psi_n, q);

  return bf;
}
