
#include "PatchBasisTests.h"

#include "SolutionTests.h"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_CellTools.hpp"

#include "BasisFactory.h"

#include "StokesBilinearForm.h"
#include "BasisEvaluation.h"

#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionProblemLegacy.h"
#include "MathInnerProduct.h"

#include "MeshTestUtility.h" // used for checkMeshConsistency
#include "MeshTestSuite.h"
#include "MeshFactory.h"

class PatchBasisInflowSquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol);
    bool yMatch = (abs(y) < tol);
    return xMatch || yMatch;
  }
};

class PatchBasisInflowFunction : public Function
{
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    double tol = 1e-11;
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0; i<cellIDs.size(); i++)
    {
      for (int j = 0; j<numPoints; j++)
      {
        double x = points(i,j,0);
        double y = points(i,j,1);
        values(i,j) = 0.0;
        if (abs(y)<tol)
        {
          values(i,j) = 1.0;
        }
        if (abs(x)<tol)
        {
          values(i,j) = -1.0;
        }
      }
    }
  }
};

typedef Teuchos::RCP< FieldContainer<double> > FCPtr;

// for some reason, we throw an exception (at least in debug mode) if we don't
// explicitly initialize the _mesh variable
PatchBasisTests::PatchBasisTests() : _mesh(Teuchos::rcp((Mesh *)NULL)) {}

void PatchBasisTests::runTests(int &numTestsRun, int &numTestsPassed)
{
  setup();
  if (testPatchBasis1D())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

//  try {
  setup();
  if (testSimpleRefinement())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testSolveUniformMesh())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testMultiLevelRefinement())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testChildPRefinementSimple())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testChildPRefinementMultiLevel())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testNeighborPRefinementSimple())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testNeighborPRefinementMultiLevel())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
//  } catch (...) {
//    cout << "PatchBasisTests: caught exception while running tests.\n";
//    teardown();
//  }
}

bool PatchBasisTests::basisValuesAgreeWithPermutedNeighbor(Teuchos::RCP<Mesh> mesh)
{
  // for every side (PatchBasis or no), compute values for that side, and values for its neighbor along
  // the same physical points.  (Imitate the comparison between parent and child, only remember that
  // the neighbor involves a flip: (-1,1) --> (1,-1).)

  return MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints1D);
}

bool PatchBasisTests::doPRefinementAndTestIt(ElementPtr elem, const string &testName)
{
  bool success = true;

  if (elem->isChild())
  {
    if ( ! childPolyOrdersAgreeWithParent(elem) )
    {
      cout << testName << ": before refinement, parent and child don't agree on p-order.\n";
      return false;
    }
  }

  vector< map< int, int> > elemPOrdersBeforeRefinement; // includes all fields and fluxes
  getPolyOrders(elemPOrdersBeforeRefinement,elem);

//  cout << "trialOrdering for cell " << elem->cellID() << " before p-refinement:\n";
//  cout << *(elem->elementType()->trialOrderPtr);

  vector<GlobalIndexType> cellsToRefine;
  cellsToRefine.push_back(elem->cellID());
  _mesh->pRefine(cellsToRefine);

//  cout << "trialOrdering for cell " << elem->cellID() << " after p-refinement:\n";
//  cout << *(elem->elementType()->trialOrderPtr);

  if (elem->isChild())
  {
    if ( ! childPolyOrdersAgreeWithParent(elem) )
    {
      cout << testName << ": after refinement, parent and child don't agree on p-order.\n";
      return false;
    }
  }

  // (check both that p-refinement was done in child, and that meshLooksGood())
  vector< map< int, int> > elemPOrdersAfterRefinement; // map from varID to p-order
  getPolyOrders(elemPOrdersAfterRefinement,elem);

  if ( ! pRefined( elemPOrdersBeforeRefinement, elemPOrdersAfterRefinement ) )
  {
    cout << testName << ": after p-refinement, child doesn't have increased p-order.\n";
    success = false;
  }

  if ( !meshLooksGood() )
  {
    success = false;
  }

  if ( !refinementsHaveNotIncreasedError() )
  {
    success = false;
  }

  if ( !success )
  {
    cout << "Failed " << testName << ".\n";
  }
  return success;
}

bool PatchBasisTests::childPolyOrdersAgreeWithParent(ElementPtr child)
{
  vector< map< int, int> > elemPOrdersAlongSharedSidesBeforeRefinement; // map from varID to p-order
  vector< map< int, int> > parentPOrdersAlongSharedSidesBeforeRefinement;

  getPolyOrdersAlongSharedSides(elemPOrdersAlongSharedSidesBeforeRefinement,
                                parentPOrdersAlongSharedSidesBeforeRefinement,
                                child);
  return polyOrdersAgree( elemPOrdersAlongSharedSidesBeforeRefinement, parentPOrdersAlongSharedSidesBeforeRefinement );
}

void PatchBasisTests::getPolyOrders(vector< map<int, int> > &polyOrderMapVector, ElementPtr elem)
{
  // polyOrderMapVector has the following maps (in the order given):
  // - fieldID -> p-order for field variable
  // - for each sideIndex: fluxID -> p-order for flux/trace variable
  polyOrderMapVector.clear();
  vector<int>::iterator varIt;
  map<int, int> polyOrders;
  for (varIt = _fieldIDs.begin(); varIt != _fieldIDs.end(); varIt++)
  {
    int fieldID = *varIt;
    int polyOrder = BasisFactory::basisFactory()->basisPolyOrder(elem->elementType()->trialOrderPtr->getBasis(fieldID));
    polyOrders[fieldID] = polyOrder;
  }
  polyOrderMapVector.push_back(polyOrders);
  int numSides = elem->numSides();
  for (int sideIndex=0; sideIndex<numSides; sideIndex++)
  {
    polyOrders.clear();
    vector<int>::iterator varIt;
    for (varIt = _fluxIDs.begin(); varIt != _fluxIDs.end(); varIt++)
    {
      int fluxID = *varIt;
      int polyOrder =  BasisFactory::basisFactory()->basisPolyOrder(elem->elementType()->trialOrderPtr->getBasis(fluxID,sideIndex));
      polyOrders[fluxID] = polyOrder;
    }
    polyOrderMapVector.push_back(polyOrders);
  }
}

void PatchBasisTests::getPolyOrdersAlongSharedSides(vector< map<int, int> > &childPOrderMapForSide,
    vector< map<int, int> > &parentPOrderMapForSide,
    ElementPtr child)
{
  childPOrderMapForSide.clear();
  parentPOrderMapForSide.clear();
  ElementPtr parent = child->getParent();
  int numSides = child->numSides();
  for (int sideIndex=0; sideIndex<numSides; sideIndex++)
  {
    int parentSideIndex = child->parentSideForSideIndex(sideIndex);
    if (parentSideIndex >= 0)   // they share this side
    {
      map<int, int> childPOrdersForSide;
      map<int, int> parentPOrdersForSide;
      vector<int>::iterator varIt;
      for (varIt = _fluxIDs.begin(); varIt != _fluxIDs.end(); varIt++)
      {
        int fluxID = *varIt;
        int childPolyOrder = child->elementType()->trialOrderPtr->getBasis(fluxID,sideIndex)->getDegree();
        int parentPolyOrder = parent->elementType()->trialOrderPtr->getBasis(fluxID,parentSideIndex)->getDegree();
        childPOrdersForSide[fluxID] = childPolyOrder;
        parentPOrdersForSide[fluxID] = parentPolyOrder;
      }
      parentPOrderMapForSide.push_back(childPOrdersForSide);
      parentPOrderMapForSide.push_back(parentPOrdersForSide);
    }
  }
}

void PatchBasisTests::hRefineAllActiveCells(Teuchos::RCP<Mesh> mesh)
{
  set<GlobalIndexType> cellIDsToRefine = mesh->getActiveCellIDsGlobal();
  mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());
}

void PatchBasisTests::makeSimpleRefinement()
{
  vector<GlobalIndexType> cellIDsToRefine;
  //cout << "refining SW element (cellID " << _sw->cellID() << ")\n";
  cellIDsToRefine.push_back(_sw->cellID()); // this is cellID 0, as things are right now implemented
  _mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());

  /*_________________________________
    |               |               |
    |               |               |
    |               |               |
    |       1       |       3       |
    |               |               |
    |               |               |
    |               |               |
    ---------------------------------
    |       |       |               |
    |   7   |   6   |               |
    |       |       |               |
    |-------0-------|       2       |
    |       |       |               |
    |   4   |   5   |               |
    |       |       |               |
    ---------------------------------*/

}

void PatchBasisTests::makeMultiLevelRefinement()
{
  makeSimpleRefinement();

  vector<GlobalIndexType> cellIDsToRefine;
  // now, find the southeast element in the refined element, and refine it
  // the southeast element should have (0.375, 0.125) at its center
  FieldContainer<double> point(1,2);
  point(0,0) = 0.375;
  point(0,1) = 0.125;
  ElementPtr elem = _mesh->elementsForPoints(point)[0];
  cellIDsToRefine.push_back(elem->cellID());
  _mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());

  /* _________________________________
     |               |               |
     |               |               |
     |               |               |
     |       1       |       3       |
     |               |               |
     |               |               |
     |               |               |
     ---------------------------------
     |       |       |               |
     |   7   |   6   |               |
     |       |       |               |
     |-------0-------|       2       |
     |       | 11| 10|               |
     |   4   |---5---|               |
     |       | 8 | 9 |               |
     ---------------------------------*/

}

bool PatchBasisTests::meshLooksGood()
{
  bool looksGood = true;
  if ( !patchBasisCorrectlyAppliedInMesh(_mesh,_fluxIDs,_fieldIDs) )
  {
    cout << "patchBasisCorrectlyAppliedInMesh returned false.\n";
    looksGood = false;
  }
  if ( !patchBasesAgreeWithParentInMesh() )
  {
    cout << "patchBasesAgreeWithParentInMesh returned false.\n";
    looksGood = false;
  }
  if ( !MeshTestUtility::checkMeshConsistency(_mesh) )
  {
    cout << "MeshTestUtility::checkMeshConsistency() returned false.\n";
    looksGood = false;
  }
  if ( !basisValuesAgreeWithPermutedNeighbor(_mesh) )
  {
    cout << "basisValuesAgreeWithPermutedNeighbor returned false.\n";
    looksGood = false;
  }
  return looksGood;
}

bool PatchBasisTests::patchBasisCorrectlyAppliedInMesh(Teuchos::RCP<Mesh> mesh, vector<int> fluxIDs, vector<int> fieldIDs)
{
  // checks that the right elements have some PatchBasis in the right places
  const set<GlobalIndexType>* myCellIDs = &mesh->cellIDsInPartition();

  // depending on our debugging needs, could revise this to return more information
  // about the nature and extent of the incorrectness when correct == false.

  bool correct = true;

  for (GlobalIndexType cellID : *myCellIDs)
  {
    ElementPtr elem = mesh->getElement(cellID);
    vector<int>::iterator varIt;
    for (varIt = fluxIDs.begin(); varIt != fluxIDs.end(); varIt++)
    {
      int fluxID = *varIt;
      int numSides = elem->numSides();
      for (int sideIndex=0; sideIndex<numSides; sideIndex++)
      {
        BasisPtr basis = elem->elementType()->trialOrderPtr->getBasis(fluxID,sideIndex);
        bool hasPatchBasis = BasisFactory::basisFactory()->isPatchBasis(basis);
        bool shouldHavePatchBasis;
        // check who the (ancestor's) neighbor is on this side:
        int sideIndexInNeighbor;
        ElementPtr neighbor = mesh->ancestralNeighborForSide(elem,sideIndex,sideIndexInNeighbor);

        // check whether the neighbor relationship is symmetric:
        if (neighbor.get() == NULL)
        {
          shouldHavePatchBasis = false;
        }
        else if (neighbor->getNeighborCellID(sideIndexInNeighbor) != elem->cellID())
        {
          // i.e. neighbor's neighbor is our parent/ancestor--so we should have a PatchBasis
          shouldHavePatchBasis = true;
        }
        else
        {
          shouldHavePatchBasis = false;
        }
        if (shouldHavePatchBasis != hasPatchBasis)
        {
          correct = false;
        }
      }
    }
    for (varIt = fieldIDs.begin(); varIt != fieldIDs.end(); varIt++)
    {
      int fieldID = *varIt;
      bool shouldHavePatchBasis = false; // false for all fields
      BasisPtr basis = elem->elementType()->trialOrderPtr->getBasis(fieldID);
      bool hasPatchBasis = BasisFactory::basisFactory()->isPatchBasis(basis);
      if (shouldHavePatchBasis != hasPatchBasis)
      {
        correct = false;
      }
    }
  }
  return correct;
}

bool PatchBasisTests::patchBasesAgreeWithParentInMesh()
{
  // checks that those elements with PatchBases compute values that agree with their parents

  // iterate through all elements (including inactive!), looking for those that have PatchBases.
  //  - anytime a PatchBasis is found, take the _testPoints1D as the subcell reference for the PatchBasis
  //  - check that parent and child agree along the shared edge to within a very small tolerance (1e-15, say)

  double tol = 1e-15;
  bool valuesAgree = true;
  int numElements = _mesh->numElements();
  for (int cellID=0; cellID < numElements; cellID++)
  {
    ElementPtr elem = _mesh->getElement(cellID);

    Teuchos::RCP< DofOrdering > trialOrdering = elem->elementType()->trialOrderPtr;
    set<int> varIDs = trialOrdering->getVarIDs();
    for (set<int>::iterator varIDIt = varIDs.begin(); varIDIt != varIDs.end(); varIDIt++)
    {
      int varID = *varIDIt;
      const vector<int>* sidesForVar = &trialOrdering->getSidesForVarID(varID);
      for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
      {
        int sideIndex = *sideIt;
        BasisPtr basis = trialOrdering->getBasis(varID,sideIndex);
        if (BasisFactory::basisFactory()->isPatchBasis(basis))
        {
          // get parent basis:
          ElementPtr parent = elem->getParent();
          int parentSideIndex = elem->parentSideForSideIndex(sideIndex);
          BasisPtr parentBasis = parent->elementType()->trialOrderPtr->getBasis(varID, parentSideIndex);

          FieldContainer<double> parentTestPoints(_testPoints1D);

          elem->getSidePointsInParentRefCoords(parentTestPoints,sideIndex,_testPoints1D);

          // evaluate testPoints and parentTestPoints in respective bases
          FCPtr parentValues = BasisEvaluation::getValues(parentBasis,OP_VALUE,parentTestPoints);
          FCPtr childValues =  BasisEvaluation::getValues(basis,OP_VALUE,_testPoints1D);

          // check that they agree
          TEUCHOS_TEST_FOR_EXCEPTION(parentValues->size() != childValues->size(), std::invalid_argument,
                                     "parentValues and childValues don't have the same size--perhaps parentBasis and child don't have the same order?");
          double maxDiff;
          if ( !fcsAgree(*parentValues,*childValues,tol,maxDiff) )
          {
            valuesAgree = false;
            cout << "For child cellID " << cellID << " on side " << sideIndex << ", ";
            cout << "parent values and childValues differ; maxDiff = " << maxDiff << "\n";
          }
//          for (int i=0; i<parentValues->size(); i++) {
//            double diff = abs((*parentValues)[i]-(*childValues)[i]);
//            if (diff > tol) {
//              cout << "For child cellID " << cellID << " on side " << sideIndex << ", ";
//              cout << "parent value != childValue (" << (*parentValues)[i] << " != " << (*childValues)[i] << ")\n";
//              valuesAgree = false;
//            }
//          }
        }
      }
    }
  }

  return valuesAgree;
}

bool PatchBasisTests::polyOrdersAgree(const vector< map<int, int> > &pOrderMapVector1,
                                      const vector< map<int, int> > &pOrderMapVector2)
{
  vector< map<int, int> >::const_iterator mapVectorIt1;
  vector< map<int, int> >::const_iterator mapVectorIt2 = pOrderMapVector2.begin();
  for (mapVectorIt1 = pOrderMapVector1.begin(); mapVectorIt1 != pOrderMapVector1.end(); mapVectorIt1++)
  {
    map<int, int> map1 = *mapVectorIt1;
    map<int, int> map2 = *mapVectorIt2;
    map<int, int>::iterator map1It;
    for (map1It=map1.begin(); map1It != map1.end(); map1It++)
    {
      pair<int,int> entry = *map1It;
      if (map2[entry.first] != entry.second)
      {
        return false;
      }
    }
    mapVectorIt2++;
  }
  return true;
}

void PatchBasisTests::pRefineAllActiveCells()
{
  set<GlobalIndexType> cellIDsToRefine = _mesh->getActiveCellIDsGlobal();
  _mesh->pRefine(cellIDsToRefine);
}

bool PatchBasisTests::pRefined(const vector< map<int, int> > &pOrderMapForSideBefore,
                               const vector< map<int, int> > &pOrderMapForSideAfter)
{
  vector< map<int, int> >::const_iterator beforeVectorIt;
  vector< map<int, int> >::const_iterator afterVectorIt = pOrderMapForSideAfter.begin();
  for (beforeVectorIt = pOrderMapForSideBefore.begin(); beforeVectorIt != pOrderMapForSideBefore.end(); beforeVectorIt++)
  {
    map<int, int> beforeMap = *beforeVectorIt;
    map<int, int> afterMap = *afterVectorIt;
    map<int, int>::iterator beforeMapIt;
    for (beforeMapIt=beforeMap.begin(); beforeMapIt != beforeMap.end(); beforeMapIt++)
    {
      pair<int,int> entry = *beforeMapIt;
      int sideIndex = entry.first;
      int pOrderAfter = afterMap[sideIndex];
      int pOrderBefore = entry.second;
      if (pOrderAfter != (pOrderBefore + 1))
      {
        return false;
      }
    }
    afterVectorIt++;
  }
  return true;
}

void PatchBasisTests::setup()
{

  _useMumps = false; // false because Jesse reports trouble with MUMPS

  /**** SUPPORT FOR TESTS THAT PATCHBASIS COMPUTES THE CORRECT VALUES *****/
  // for tests, we'll do a simple division of a line segment into thirds
  // (for now, PatchBasis only supports 1D bases--sufficient for 2D DPG meshes)
  // setup bases:
  int polyOrder = 3;
  _parentBasis = BasisFactory::basisFactory()->getBasis( polyOrder, shards::Line<2>::key, Camellia::FUNCTION_SPACE_HGRAD );
  FieldContainer<double> nodesLeft(2,1), nodesMiddle(2,1), nodesRight(2,1);
  nodesLeft(0,0)   = -1.0;
  nodesLeft(1,0)   = -1.0 / 3.0;
  nodesMiddle(0,0) = -1.0 / 3.0;
  nodesMiddle(1,0) = 1.0 / 3.0;
  nodesRight(0,0)  = 1.0 / 3.0;
  nodesRight(1,0)  = 1.0;
  _patchBasisLeft   = BasisFactory::basisFactory()->getPatchBasis(_parentBasis,nodesLeft);
  _patchBasisMiddle = BasisFactory::basisFactory()->getPatchBasis(_parentBasis,nodesMiddle);
  _patchBasisRight  = BasisFactory::basisFactory()->getPatchBasis(_parentBasis,nodesRight);

  double refCellLeft = -1.0;
  double refCellRight = 1.0;

  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};

  _testPoints1D = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++)
  {
    _testPoints1D(i, 0) = x[i];
  }

  _testPoints1DLeftParent   = FieldContainer<double>(NUM_POINTS_1D,1);
  _testPoints1DMiddleParent = FieldContainer<double>(NUM_POINTS_1D,1);
  _testPoints1DRightParent  = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++)
  {
    double offset = (x[i] - refCellLeft) / 3.0;
    _testPoints1DLeftParent(i,0)   = -1.0       + offset;
    _testPoints1DMiddleParent(i,0) = -1.0 / 3.0 + offset;
    _testPoints1DRightParent(i,0)  =  1.0 / 3.0 + offset;
  }

  /**** SUPPORT FOR TESTS THAT PATCHBASIS IS CORRECTLY ASSIGNED WITHIN MESH *****/
  // first, build a simple mesh

  FieldContainer<double> quadPoints(4,2);

  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;

  int H1Order = 3;
  int delta_p = 3; // for tests
  int horizontalCells = 2;
  int verticalCells = 2;

  double eps = 1.0; // not really testing for sharp gradients right now--just want to see if things basically work
  double beta_x = 1.0;
  double beta_y = 1.0;
  // _convectionExactSolution = Teuchos::rcp( new ConfusionManufacturedSolution(eps,beta_x,beta_y) );

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr v = varFactory->testVar("v", HGRAD);

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());

  // define trial variables
  VarPtr beta_n_u = varFactory->fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory->fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );

  FunctionPtr n = Function::normal();
  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);

  _mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, convectionBF, H1Order, H1Order+delta_p);

  RHSPtr rhs = RHS::rhs();
  BCPtr bc = BC::bc();
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new PatchBasisInflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new NegatedSpatialFilter(inflowBoundary) );

  FunctionPtr uIn;
  uIn = Teuchos::rcp(new PatchBasisInflowFunction); // uses a discontinuous piecewise-constant basis function on left and bottom sides of square
  bc->addDirichlet(beta_n_u, inflowBoundary, beta*n*uIn);

  _convectionSolution = Teuchos::rcp( new Solution(_mesh, bc, rhs, ip) );

  _mesh->registerSolution(_convectionSolution);

  // the right way to determine the southwest element, etc. is as follows:
  FieldContainer<double> points(4,2);
  // southwest center:
  points(0,0) = 0.25;
  points(0,1) = 0.25;
  // southeast center:
  points(1,0) = 0.75;
  points(1,1) = 0.25;
  // northwest center:
  points(2,0) = 0.25;
  points(2,1) = 0.75;
  // northeast center:
  points(3,0) = 0.75;
  points(3,1) = 0.75;
  vector<ElementPtr> elements = _mesh->elementsForPoints(points);

  _sw = elements[0];
  _se = elements[1];
  _nw = elements[2];
  _ne = elements[3];

//  cout << "SW nodes:\n" << _mesh->physicalCellNodesForCell(_sw->cellID());
//  cout << "SE nodes:\n" << _mesh->physicalCellNodesForCell(_se->cellID());
//  cout << "NW nodes:\n" << _mesh->physicalCellNodesForCell(_nw->cellID());
//  cout << "NE nodes:\n" << _mesh->physicalCellNodesForCell(_ne->cellID());

  _convectionSolution->solve(_useMumps);

  //  for (vector<int>::iterator fieldIt=_fieldIDs.begin(); fieldIt != _fieldIDs.end(); fieldIt++) {
  //    int fieldID = *fieldIt;
  //    double err = _convectionExactSolution->L2NormOfError(*(_convectionSolution.get()),fieldID);
  //    _convectionL2ErrorForOriginalMesh[fieldID] = err;
  //  }

  _convectionEnergyErrorForOriginalMesh = _convectionSolution->energyErrorTotal();

//  _convectionSolution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_patchBasis_before_refinement.m");

  _mesh->setUsePatchBasis(true);

  _fluxIDs = convectionBF->trialBoundaryIDs();
  _fieldIDs = convectionBF->trialVolumeIDs();

}

bool PatchBasisTests::refinementsHaveNotIncreasedError()
{
  return refinementsHaveNotIncreasedError(_convectionSolution);
}

bool PatchBasisTests::refinementsHaveNotIncreasedError(Teuchos::RCP<Solution> solution)
{
  double tol = 1e-11;

  bool success = true;

  solution->solve(_useMumps);

  double err = _convectionSolution->energyErrorTotal();
  double diff = err - _convectionEnergyErrorForOriginalMesh;
  if (diff > tol)
  {
    cout << "PatchBasisTests: increase in error after refinement " << diff << " > tol " << tol << ".\n";

//    solution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_patchBasis.m");
//    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "confusion_u_hat_patchBasis.m");

    success = false;
  }

//  for (vector<int>::iterator fieldIt=_fieldIDs.begin(); fieldIt != _fieldIDs.end(); fieldIt++) {
//    int fieldID = *fieldIt;
//    double err = _convectionExactSolution->L2NormOfError(*(_convectionSolution.get()),fieldID);
//    double originalErr = _convectionL2ErrorForOriginalMesh[fieldID];
//    if (err - originalErr > tol) {
//      cout << "PatchBasisTests: increase in error after refinement " << err - originalErr << " > tol " << tol << " for ";
//      cout << _convectionExactSolution->bilinearForm()->trialName(fieldID) << endl;
//
//      solution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_patchBasis.m");
//      solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "confusion_u_hat_patchBasis.m");
//
//      success = false;
//    }
//  }

  return success;
}


void PatchBasisTests::teardown()
{
  _testPoints1D.resize(0);
  _testPoints1DLeftParent.resize(0);
  _testPoints1DMiddleParent.resize(0);
  _testPoints1DRightParent.resize(0);
  _parentBasis = Teuchos::rcp((Camellia::Basis<> *)NULL);
  _patchBasisLeft = Teuchos::rcp((PatchBasis<> *)NULL);
  _patchBasisMiddle = Teuchos::rcp((PatchBasis<> *)NULL);
  _patchBasisRight = Teuchos::rcp((PatchBasis<> *)NULL);

  _mesh = Teuchos::rcp((Mesh *)NULL);
  _sw = Teuchos::rcp((Element *)NULL);
  _se = Teuchos::rcp((Element *)NULL);
  _nw = Teuchos::rcp((Element *)NULL);
  _ne = Teuchos::rcp((Element *)NULL);

}

bool PatchBasisTests::testPatchBasis1D()
{
  bool success = true;

  double tol = 1e-15;
  int numPoints = _testPoints1D.size();
  int numFields = _parentBasis->getCardinality();
  FieldContainer<double> valuesLeft(numFields,numPoints),   expectedValuesLeft(numFields,numPoints);
  FieldContainer<double> valuesMiddle(numFields,numPoints), expectedValuesMiddle(numFields,numPoints);
  FieldContainer<double> valuesRight(numFields,numPoints),  expectedValuesRight(numFields,numPoints);

  // get the expected values
  _parentBasis->getValues(expectedValuesLeft,   _testPoints1DLeftParent,   Intrepid::OPERATOR_VALUE);
  _parentBasis->getValues(expectedValuesMiddle, _testPoints1DMiddleParent, Intrepid::OPERATOR_VALUE);
  _parentBasis->getValues(expectedValuesRight,  _testPoints1DRightParent,  Intrepid::OPERATOR_VALUE);

  // get the actual values:
  _patchBasisLeft  ->getValues(valuesLeft,   _testPoints1D, Intrepid::OPERATOR_VALUE);
  _patchBasisMiddle->getValues(valuesMiddle, _testPoints1D, Intrepid::OPERATOR_VALUE);
  _patchBasisRight ->getValues(valuesRight,  _testPoints1D, Intrepid::OPERATOR_VALUE);

  for (int fieldIndex=0; fieldIndex < numFields; fieldIndex++)
  {
    for (int pointIndex=0; pointIndex < numPoints; pointIndex++)
    {
      double diff = abs(valuesLeft(fieldIndex,pointIndex) - expectedValuesLeft(fieldIndex,pointIndex));
      if (diff > tol)
      {
        success = false;
        cout << "expected value of left basis: " << expectedValuesLeft(fieldIndex,pointIndex) << "; actual: " << valuesLeft(fieldIndex,pointIndex) << endl;
      }

      diff = abs(valuesMiddle(fieldIndex,pointIndex) - expectedValuesMiddle(fieldIndex,pointIndex));
      if (diff > tol)
      {
        success = false;
        cout << "expected value of middle basis: " << expectedValuesMiddle(fieldIndex,pointIndex) << "; actual: " << valuesMiddle(fieldIndex,pointIndex) << endl;
      }

      diff = abs(valuesRight(fieldIndex,pointIndex) - expectedValuesRight(fieldIndex,pointIndex));
      if (diff > tol)
      {
        success = false;
        cout << "expected value of right basis: " << expectedValuesRight(fieldIndex,pointIndex) << "; actual: " << valuesRight(fieldIndex,pointIndex) << endl;
      }
    }
  }

  return success;
}

bool PatchBasisTests::testSimpleRefinement()
{
  // refine in the sw, and then check that the right elements have PatchBases
  bool success = true;

//  if ( ! SolutionTests::solutionCoefficientsAreConsistent(_convectionSolution) ) {
//    cout << "BEFORE simple refinement, solution coefficients are inconsistent.\n";
//  }

//  cout << "Before testSimpleRefinement, boundary: " << endl;
//  for (int cellID=0; cellID<_mesh->numElements(); cellID++) {
//    cout << "cellID " << cellID << ":";
//    for (int sideIndex=0; sideIndex<4; sideIndex++) {
//      if (_mesh->boundary().boundaryElement(cellID,sideIndex)) {
//        cout << sideIndex << " ";
//      }
//    }
//    cout << endl;
//  }

  // the _nw and _se element's dofs should not change: let's store them and check this
  FieldContainer<double> nwDofsBefore = _convectionSolution->allCoefficientsForCellID(_nw->cellID());
  FieldContainer<double> seDofsBefore = _convectionSolution->allCoefficientsForCellID(_se->cellID());

//  cout << "cellID 1, trial ordering:\n";
//  cout << *(_mesh->getElement(1)->elementType()->trialOrderPtr);

//  cout << "cellID 0 dof coefficients before refinement:\n" << _convectionSolution->allCoefficientsForCellID(0);

  makeSimpleRefinement();

//  cout << "cellID 7, trial ordering:\n";
//  cout << *(_mesh->getElement(7)->elementType()->trialOrderPtr);

  FieldContainer<double> nwDofsAfter = _convectionSolution->allCoefficientsForCellID(_nw->cellID());
  FieldContainer<double> seDofsAfter = _convectionSolution->allCoefficientsForCellID(_se->cellID());

//  cout << "cellID 5 dof coefficients:\n" << _convectionSolution->allCoefficientsForCellID(5);

//  cout << "cellID 2 dof coefficients:\n" << _convectionSolution->allCoefficientsForCellID(2);

  double tol = 1e-15;
  double maxDiff = 0;
  if (! fcsAgree(nwDofsBefore, nwDofsAfter, tol, maxDiff))
  {
    success = false;
    cout << "nw dofs before and after sw refinement don't match. Maxdiff " << maxDiff << endl;
  }
  if (! fcsAgree(seDofsBefore, seDofsAfter, tol, maxDiff))
  {
    success = false;
    cout << "se dofs before and after sw refinement don't match. Maxdiff " << maxDiff << endl;
  }

//  cout << "After testSimpleRefinement, boundary: " << endl;
//  for (int cellID=0; cellID<_mesh->numElements(); cellID++) {
//    cout << "cellID " << cellID << ":";
//    for (int sideIndex=0; sideIndex<4; sideIndex++) {
//      if (_mesh->boundary().boundaryElement(cellID,sideIndex)) {
//        cout << sideIndex << " ";
//      }
//    }
//    cout << endl;
//  }

  // the following test would be reasonable, except that we don't do the projection onto fluxes and traces yet...
//  bool verbose = false;
//  if ( ! SolutionTests::solutionCoefficientsAreConsistent(_convectionSolution, verbose) ) {
//    cout << "After simple refinement, solution coefficients are inconsistent.\n";
//    success = false;
//  }

  if ( !meshLooksGood() || (! refinementsHaveNotIncreasedError()) )
  {
    success = false;
    cout << "Failed testSimpleRefinement.\n";
  }

  return success;
}

bool PatchBasisTests::testMultiLevelRefinement()
{
  // refine in the sw, then refine in its se, and check the mesh
  bool success = true;
  makeMultiLevelRefinement();

  if ( !meshLooksGood() || (! refinementsHaveNotIncreasedError()))
  {
    success = false;
    cout << "Failed testMultiLevelRefinement.\n";
  }

  return success;
}

bool PatchBasisTests::testChildPRefinementSimple()
{
  // in same mesh as the simple h-refinement test, p-refine the child.  Check that its parent also gets p-refined...
  makeSimpleRefinement();

  bool success = true;

  // the child we'd like to p-refine is the upper-right quadrant of the lower-left cell of the original mesh.
  // since we're on a unit square, that element contains the point (0.375, 0.375)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.375;
  cellPoint(0,1) = 0.375;
  ElementPtr child = _mesh->elementsForPoints(cellPoint)[0];

  return doPRefinementAndTestIt(child,"testChildPRefinementSimple");
}

bool PatchBasisTests::testChildPRefinementMultiLevel()
{
  // in same mesh as the multi-level h-refinement test, p-refine the child.  Check that its parent and grandparent also get p-refined...
  bool success = true;
  makeMultiLevelRefinement();

  // the child we'd like to p-refine is NE quad. of the SE quad. of the SW element of the original mesh.
  // since we're on a unit square, that element contains the point (0.4375, 0.1875)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.4375;
  cellPoint(0,1) = 0.1875;
  ElementPtr child = _mesh->elementsForPoints(cellPoint)[0];

  return doPRefinementAndTestIt(child,"testChildPRefinementMultiLevel");
}

bool PatchBasisTests::testNeighborPRefinementSimple()
{
  // in same mesh as the simple h-refinement test, p-refine a big neighbor.  Check that its parent also gets p-refined...
  makeSimpleRefinement();

  // the neighbor we'd like to p-refine is SE quad of the original mesh
  // since we're on a unit square, that element contains the point (0.75, 0.25)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.75;
  cellPoint(0,1) = 0.25;
  ElementPtr neighbor = _mesh->elementsForPoints(cellPoint)[0];

  return doPRefinementAndTestIt(neighbor,"testNeighborPRefinementSimple");
}

bool PatchBasisTests::testNeighborPRefinementMultiLevel()
{
  // in same mesh as the multi-level h-refinement test, p-refine a big neighbor.  Check that its parent and grandparent also get p-refined...
  makeMultiLevelRefinement();

  // the neighbor we'd like to p-refine is SE quad of the original mesh
  // since we're on a unit square, that element contains the point (0.75, 0.25)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.75;
  cellPoint(0,1) = 0.25;
  ElementPtr neighbor = _mesh->elementsForPoints(cellPoint)[0];

  return doPRefinementAndTestIt(neighbor,"testNeighborPRefinementMultiLevel");
}

bool PatchBasisTests::testSolveUniformMesh()
{
  // TODO: clean up this test, and make it a proper test... (Right now, a container for debug code!)
  int H1Order = 2;
  int horizontalCells = 2;
  int verticalCells = 2;

  FieldContainer<double> quadPoints(4,2);

  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;

  // setup the solution objects:
  int polyOrder = H1Order - 1;

  double eps = 1.0, beta_x = 1.0, beta_y = 1.0;
  BFPtr confusionBF = ConfusionBilinearForm::confusionBF(eps,beta_x,beta_y);

  Teuchos::RCP<Mesh> multiBasisMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, confusionBF, H1Order, H1Order);

  hRefineAllActiveCells(multiBasisMesh);

  IPPtr ip = Teuchos::rcp( new MathInnerProduct(confusionBF) );

  Teuchos::RCP<ConfusionProblemLegacy> confusionProblem = Teuchos::rcp( new ConfusionProblemLegacy(confusionBF, beta_x, beta_y) );

  Teuchos::RCP<Solution> mbSolution = Teuchos::rcp(new Solution(multiBasisMesh, confusionProblem, confusionProblem, ip));
//  cout << "solving MultiBasis...\n";
  mbSolution->solve(_useMumps);

//  mbSolution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_multiBasis.m");
//  mbSolution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "confusion_u_hat_multiBasis.m");

//  cout << "MultiBasis localToGlobalMap:\n";
//  multiBasisMesh->printLocalToGlobalMap();

  Teuchos::RCP<Mesh> patchBasisMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, confusionBF, H1Order, H1Order);
  patchBasisMesh->setUsePatchBasis(true);

  Teuchos::RCP<Solution> pbSolution = Teuchos::rcp(new Solution(patchBasisMesh, confusionProblem, confusionProblem, ip));

  bool success = true;

  hRefineAllActiveCells(patchBasisMesh);

//  cout << "solving PatchBasis...\n";
  pbSolution->solve(_useMumps);

//  pbSolution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_patchBasis.m");
//  pbSolution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "confusion_u_hat_patchBasis.m");

//  cout << "PatchBasis localToGlobalMap:\n";
//  patchBasisMesh->printLocalToGlobalMap();

  set<GlobalIndexType> allActiveCells = _mesh->getActiveCellIDsGlobal();
  
  vector<GlobalIndexType> cellsToRefine;
  // just refine first active element
  cellsToRefine.push_back(*allActiveCells.begin());
  patchBasisMesh->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());

  if ( !patchBasisCorrectlyAppliedInMesh(patchBasisMesh,_fluxIDs,_fieldIDs) )
  {
    cout << "patchBasisCorrectlyAppliedInMesh returned false.\n";
    success = false;
  }

  pbSolution->solve(_useMumps);

//  pbSolution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_patchBasis_refined.m");
//  pbSolution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "confusion_u_hat_patchBasis_refined.m");

  return success;
}
