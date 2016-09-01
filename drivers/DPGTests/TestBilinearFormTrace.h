//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef TEST_BILINEAR_FORM_TRACE
#define TEST_BILINEAR_FORM_TRACE

#include "BF.h"

/*
 This test bilinear form just has b(u,v) = Int_dK (u_hat, \tau \cdot n),
 where u_hat is a trace (belongs formally to H^(1/2)), and \tau is a test
 function, belonging to H(div,K).
 */

class TestBilinearFormTrace : public BF
{
private:
  static const string & S_TEST;
  static const string & S_TRIAL;

public:
  TestBilinearFormTrace() : BF(true)   // true: is legacy subclass
  {
    _testIDs.push_back(0);
    _trialIDs.push_back(0);
  }

  // implement the virtual methods declared in super:
  const string & testName(int testID)
  {
    return S_TEST;
  }
  const string & trialName(int trialID)
  {
    return S_TRIAL;
  }

  bool trialTestOperator(int trialID, int testID,
                         Camellia::EOperator &trialOperator, Camellia::EOperator &testOperator)
  {
    trialOperator = Camellia::OP_VALUE;
    testOperator  = Camellia::OP_DOT_NORMAL;
    return true;
  }

  void applyBilinearFormData(int trialID, int testID,
                             FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                             const FieldContainer<double> &points)
  {
    // leave values as they are...
  }

  Camellia::EFunctionSpace functionSpaceForTest(int testID)
  {
    return Camellia::FUNCTION_SPACE_HDIV;
  }

  Camellia::EFunctionSpace functionSpaceForTrial(int trialID)
  {
    return Camellia::FUNCTION_SPACE_HGRAD;
  }

  bool isFluxOrTrace(int trialID)
  {
    return true;
  }

};

const string & TestBilinearFormTrace::S_TEST  = "test";
const string & TestBilinearFormTrace::S_TRIAL = "trace";

#endif
