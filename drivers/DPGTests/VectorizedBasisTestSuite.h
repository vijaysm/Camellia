//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef DPGTrilinos_VectorizedBasisTestSuite_h
#define DPGTrilinos_VectorizedBasisTestSuite_h

//
//  VectorizedBasisTestSuite.h
//  DPGTrilinos
//


#include "Vectorized_Basis.hpp"
#include "BasisFactory.h"
#include "TestSuite.h"

class VectorizedBasisTestSuite : public TestSuite
{

public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName();
  static bool testVectorizedBasis();
  static bool testVectorizedBasisTags();
  static bool testPoisson();
  static bool testHGRAD_2D();
};

#endif
