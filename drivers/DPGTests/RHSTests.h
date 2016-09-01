//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
//
//  RHSTests.h
//  Camellia
//

#ifndef Camellia_RHSTests_h
#define Camellia_RHSTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace Camellia;

#include "Mesh.h"
#include "RHS.h"
#include "Var.h"

class RHSTests : public TestSuite
{
  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP<RHS> _rhs;
  Teuchos::RCP<RHS> _rhsEasy;

  VarPtr _v, _tau;

  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);

  bool testComputeRHSLegacy(); // test copied from DPGTests
  bool testIntegrateAgainstStandardBasis();
  bool testRHSEasy();
  bool testTrivialRHS();

  std::string testSuiteName();
};

#endif
