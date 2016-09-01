//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ElementTests.h
//  Camellia
//

#ifndef Camellia_ElementTests_h
#define Camellia_ElementTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Mesh.h"

class ElementTests : public TestSuite
{
  FieldContainer<double> _testPoints1D;

  Teuchos::RCP<Mesh> _mesh; // a 2x2 mesh refined in SW, and then in the SE of the SW
  ElementPtr _sw, _se, _nw, _ne, _sw_se, _sw_ne, _sw_se_se, _sw_se_ne;

  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);

  bool testNeighborPointMapping();
  bool testParentPointMapping();

  std::string testSuiteName();
};

#endif
