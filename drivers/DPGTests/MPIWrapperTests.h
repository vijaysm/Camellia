//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MPIWrapperTests.h
//  Camellia-debug
//
//

#ifndef __Camellia_MPIWrapperTests__
#define __Camellia_MPIWrapperTests__

#include "TestSuite.h"
#include "MPIWrapper.h"

class MPIWrapperTests : public TestSuite
{
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);

  bool testSimpleSum();
  bool testentryWiseSum();

  std::string testSuiteName();
};

#endif /* defined(__Camellia_MPIWrapperTests__) */
