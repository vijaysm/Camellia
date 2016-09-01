//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef TEST_BILINEAR_FORM_FLUX
#define TEST_BILINEAR_FORM_FLUX

#include "BF.h"

/*
 This test bilinear form just has b(u,v) = Int_dK (u_hat, v ),
 where u_hat is a trace (belongs formally to H^(1/2)), and \tau is a test
 function, belonging to H(div,K).
 */

using namespace Camellia;
using namespace Intrepid;

class TestBilinearFormFlux : public BF
{
private:
//  static const string & S_TEST;
//  static const string & S_TRIAL;

public:
  TestBilinearFormFlux();

  // implement the virtual methods declared in super:
  const string & testName(int testID);
  const string & trialName(int trialID);

  bool trialTestOperator(int trialID, int testID,
                         Camellia::EOperator &trialOperator, Camellia::EOperator &testOperator);

  void applyBilinearFormData(int trialID, int testID,
                             FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                             const FieldContainer<double> &points);

  Camellia::EFunctionSpace functionSpaceForTest(int testID);

  Camellia::EFunctionSpace functionSpaceForTrial(int trialID);

  bool isFluxOrTrace(int trialID);

};

#endif
