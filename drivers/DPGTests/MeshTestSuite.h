//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef MESH_TEST_SUITE
#define MESH_TEST_SUITE

/*
 *  MeshTestSuite.h
 *
 */

#include "DofOrderingFactory.h"
#include "Mesh.h"

#include "TestSuite.h"

using namespace Camellia;

class MeshTestSuite : public TestSuite
{
private:
//  static bool checkMeshDofConnectivities(Teuchos::RCP<Mesh> mesh);
  static bool checkDofOrderingHasNoOverlap(Teuchos::RCP<DofOrdering> dofOrdering);
  static bool vectorPairsEqual( vector< pair<unsigned,unsigned> > &first, vector< pair<unsigned,unsigned> > &second);
  static void printParities(Teuchos::RCP<Mesh> mesh);
  // checkDofOrderingHasNoOverlap returns true if no two (varID,basisOrdinal,sideIndex) tuples map to same dofIndex
public:
//  static bool checkMeshConsistency(Teuchos::RCP<Mesh> mesh);
  static bool neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, const FieldContainer<double> &testPointsRefCoords,
                                        bool reportErrors = false);

  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName()
  {
    return "MeshTestSuite";
  }
  static bool testBuildMesh();
  static bool testMeshSolvePointwise();
  static bool testExactSolution(bool checkL2Error);
  static bool testSacadoExactSolution();
  static bool testPoissonConvergence();
  static bool testBasisRefinement();
  static bool testFluxNorm();
  static bool testFluxIntegration();
  static bool testDofOrderingFactory();
  static bool testPRefinement();
  static bool testSinglePointBC();
  static bool testSolutionForMultipleElementTypes();
  static bool testSolutionForSingleElementUpgradedSide();
  static bool testHRefinement();
  static bool testHRefinementForConfusion();
  static bool testHUnrefinementForConfusion();
  static bool testRefinementPattern();
  static bool testPointContainment();

  // added by Jesse
  static bool testJesseMultiBasisRefinement();
  static bool testJesseAnisotropicRefinement();
  static bool testPRefinementAdjacentCells();
};

#endif
