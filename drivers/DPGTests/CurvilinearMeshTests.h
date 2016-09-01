//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CurvilinearMeshTests.h
//  Camellia-debug
//
//

#ifndef __Camellia_debug__CurvilinearMeshTests__
#define __Camellia_debug__CurvilinearMeshTests__

#include "TestSuite.h"

class CurvilinearMeshTests : public TestSuite
{
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);

  bool testAutomaticStraightEdgesMatchVertices();
  bool testEdgeLength();
  bool testCylinderMesh();
  bool testH1Projection();
  bool testPointsRemainInsideElement();
  bool testTransformationJacobian();
  bool testStraightEdgeMesh();

  std::string testSuiteName();
};


#endif /* defined(__Camellia_debug__CurvilinearMeshTests__) */
