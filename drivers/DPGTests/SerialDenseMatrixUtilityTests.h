//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  SerialDenseSolveWrapperTests.h
//  Camellia-debug
//
//

#ifndef __Camellia_debug__SerialDenseSolveWrapperTests__
#define __Camellia_debug__SerialDenseSolveWrapperTests__

#include "TestSuite.h"

class SerialDenseMatrixUtilityTests : public TestSuite
{
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  bool testMultiplyMatrices();
  bool testSimpleSolve();
  bool testSolveMultipleRHS();
  bool testAddMatrices();

  std::string testSuiteName();
};


#endif /* defined(__Camellia_debug__SerialDenseSolveWrapperTests__) */
