//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef POISSON_BILINEAR_FORM_SPECIFICATION
#define POISSON_BILINEAR_FORM_SPECIFICATION

#include "BF.h"

using namespace Camellia;
using namespace Intrepid;

using namespace std;

class PoissonBilinearForm
{
public:
  // trial variable names:
  static const string S_PHI, S_PSI_1, S_PSI_2, S_PHI_HAT, S_PSI_HAT_N;
  // test variable names:
  static const string S_Q, S_TAU;

  static BFPtr poissonBilinearForm(bool useConformingTraces = true);
};
#endif
