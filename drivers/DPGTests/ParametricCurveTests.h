//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ParametricCurveTests.h
//  Camellia-debug
//
//

#ifndef Camellia_debug_ParametricCurveTests_h
#define Camellia_debug_ParametricCurveTests_h

#include "ParametricCurve.h"
#include "TestSuite.h"

class ParametricCurveTests : public TestSuite
{
  void setup();
  void teardown() {}
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  bool testBubble();
  bool testCircularArc();
  bool testGradientWrapper();
  bool testLine();
  bool testParametricCurveRefinement(); // tests the kind of thing that will happen to parametric curves during mesh refinement
  bool testPolygon();
  bool testProjectionBasedInterpolation();
  bool testTransfiniteInterpolant();

  std::string testSuiteName();
};

#endif
