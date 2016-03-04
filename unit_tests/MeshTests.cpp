//
//  MeshTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/19/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "GlobalDofAssignment.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "StokesVGPFormulation.h"

#include <cstdio>

using namespace Camellia;
using namespace Intrepid;

namespace
{
MeshPtr makeTestMesh( int spaceDim, bool spaceTime )
{
  MeshPtr mesh;
  if ((spaceDim == 1) && spaceTime)
  {
    int tensorialDegree = 1;
    CellTopoPtr line_x_time = CellTopology::cellTopology(CellTopology::line(), tensorialDegree);

    vector<double> v00 = {-1,-1};
    vector<double> v10 = { 1,-1};
    vector<double> v20 = { 2,-1};
    vector<double> v01 = {-1, 1};
    vector<double> v11 = { 1, 1};
    vector<double> v21 = { 2, 1};

    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00); // 0
    spaceTimeVertices.push_back(v10); // 1
    spaceTimeVertices.push_back(v20); // 2
    spaceTimeVertices.push_back(v01); // 3
    spaceTimeVertices.push_back(v11); // 4
    spaceTimeVertices.push_back(v21); // 5

    vector<unsigned> spaceTimeLine1VertexList;
    vector<unsigned> spaceTimeLine2VertexList;
    spaceTimeLine1VertexList.push_back(0);
    spaceTimeLine1VertexList.push_back(1);
    spaceTimeLine1VertexList.push_back(3);
    spaceTimeLine1VertexList.push_back(4);
    spaceTimeLine2VertexList.push_back(1);
    spaceTimeLine2VertexList.push_back(2);
    spaceTimeLine2VertexList.push_back(4);
    spaceTimeLine2VertexList.push_back(5);

    vector< vector<unsigned> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeLine1VertexList);
    spaceTimeElementVertices.push_back(spaceTimeLine2VertexList);

    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(line_x_time);
    spaceTimeCellTopos.push_back(line_x_time);

    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );

    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr v = varFactory->testVar("v", HGRAD);

    // define trial variables
    VarPtr uhat = varFactory->fluxVar("uhat");

    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = BF::bf(varFactory);

    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 3, pToAdd = 1;
    mesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );
  }
  else if (!spaceTime)
  {
    int H1Order = 2;
    vector<int> elemCounts(spaceDim,2);
    vector<double> dims(spaceDim,1.0);
    
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    mesh = MeshFactory::rectilinearMesh(form.bf(), dims, elemCounts, H1Order);
  }
  else
  {
    // TODO: handle other space-time mesh options for non-1D spatial meshes
  }
  return mesh;
}

  TEUCHOS_UNIT_TEST( Mesh, ConstructSingleCellMeshSerialComm )
  {
    // the purpose of this test is just to ensure that construction for a serial communicator works
    // without any MPI communication (there used to be some hard-coded MPI_COMM_WORLDs in
    // the mesh partitioning and dof assignments).
    MPIWrapper::CommWorld()->Barrier(); // for setting a breakpoint for debugging
    
    int globalRank = MPIWrapper::CommWorld()->MyPID();

    int spaceDim = 2;
    int H1Order = 2;
    vector<int> elemCounts(spaceDim,2);
    vector<double> dims(spaceDim,1.0);
    
    bool conformingTraces = false; // non-conformity allows us to easily determine how many global dofs to expect on the single-element mesh
    PoissonFormulation form(spaceDim,conformingTraces);
    
    MeshPtr originalMesh = MeshFactory::rectilinearMesh(form.bf(), dims, elemCounts, H1Order);
    
    if (globalRank==0)
    {
      GlobalIndexType coarseCellID = *originalMesh->getActiveCellIDs().begin();
      DofOrderingPtr trialOrdering = originalMesh->getElementType(coarseCellID)->trialOrderPtr;
      int localDofs = trialOrdering->totalDofs();
      
      MeshPtr singleCellMesh = Teuchos::rcp( new Mesh(originalMesh, coarseCellID, MPIWrapper::CommSerial()) );
      
      int globalDofs = singleCellMesh->numGlobalDofs();
      
      TEUCHOS_TEST_EQUALITY(localDofs, globalDofs, out, success);
    }
  }
  
  TEUCHOS_UNIT_TEST( Mesh, EnforceRegularityInteriorTriangles )
  {
    int spaceDim = 2;
    int H1Order = 2;
    bool useConformingTraces = true;
    
    int delta_k = spaceDim;
    
    vector<vector<double>> vertices = {{0,0},{1,0},{0.5,1}};
    vector<vector<IndexType>> elementVertices = {{0,1,2}};
    CellTopoPtr triangle = CellTopology::triangle();
    
    MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, {triangle}) );
    MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(geometry));
    
    // create a problematic mesh of a particular sort: refine once, then refine the interior element.  Then refine the interior element of the refined element.
    IndexType cellIDToRefine = 0, nextCellIndex = 1;
    int interiorChildOrdinal = 1; // interior child has index 1 in children
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(triangle);
    meshTopo->refineCell(cellIDToRefine, refPattern, nextCellIndex);
    nextCellIndex += refPattern->numChildren();
    
    vector<CellPtr> children = meshTopo->getCell(cellIDToRefine)->children();
    cellIDToRefine = children[interiorChildOrdinal]->cellIndex();
    meshTopo->refineCell(cellIDToRefine, refPattern, nextCellIndex);
    nextCellIndex += refPattern->numChildren();
    
    children = meshTopo->getCell(cellIDToRefine)->children();
    cellIDToRefine = children[interiorChildOrdinal]->cellIndex();
    meshTopo->refineCell(cellIDToRefine, refPattern, nextCellIndex);
    nextCellIndex += refPattern->numChildren();
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, poissonForm.bf(), H1Order, delta_k) );
    
    // thus far, we have done 3 refinements, each of which added 3 elements.  Expect to have 10 elements
    int numActiveElementsExpected = 10;
    int numActiveElements = mesh->numActiveElements();
    TEST_EQUALITY(numActiveElements, numActiveElementsExpected);
    
    // The above mesh will cause some cascading constraints, which the new getBasisMap() can't
    // handle.  We have added logic to deal with this case to Mesh::enforceOneIrregularity().
    mesh->enforceOneIrregularity();
    
    // The strategy above should induce refinements on the topmost level.
    // 3 refinements, each of which adds 3 elements to the active count: expect 19 elements
    numActiveElementsExpected = 19;
    
    numActiveElements = mesh->numActiveElements();
    TEST_EQUALITY(numActiveElements, numActiveElementsExpected);
  }
  
TEUCHOS_UNIT_TEST( Mesh, ParitySpaceTime1D )
{
  int spaceDim = 1;
  bool spaceTime = true;
  MeshPtr spaceTimeMesh = makeTestMesh(spaceDim, spaceTime);

  set<GlobalIndexType> cellIDs = spaceTimeMesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    FieldContainer<double> parities = spaceTimeMesh->globalDofAssignment()->cellSideParitiesForCell(cellID);
    CellPtr cell = spaceTimeMesh->getTopology()->getCell(cellID);
    for (int sideOrdinal=0; sideOrdinal<cell->getSideCount(); sideOrdinal++)
    {
      double parity = parities[sideOrdinal];
      if (cell->getNeighbor(sideOrdinal,spaceTimeMesh->getTopology()) == Teuchos::null)
      {
        // where there is no neighbor, the parity should be 1.0
        TEST_EQUALITY(parity, 1.0);
      }
      else
      {
        pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal,spaceTimeMesh->getTopology());
        GlobalIndexType neighborCellID = neighborInfo.first;
        unsigned neighborSide = neighborInfo.second;
        FieldContainer<double> neighborParities = spaceTimeMesh->globalDofAssignment()->cellSideParitiesForCell(neighborCellID);
        double neighborParity = neighborParities[neighborSide];
        TEST_EQUALITY(parity, -neighborParity);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( Mesh, NormalSpaceTime1D )
{
  int spaceDim = 1;
  bool spaceTime = true;
  MeshPtr spaceTimeMesh = makeTestMesh(spaceDim, spaceTime);

  double tol = 1e-15;
  set<GlobalIndexType> cellIDs = spaceTimeMesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(spaceTimeMesh,cellID);

    CellPtr cell = spaceTimeMesh->getTopology()->getCell(cellID);
    for (int sideOrdinal=0; sideOrdinal<cell->getSideCount(); sideOrdinal++)
    {
      FieldContainer<double> sideNormalsSpaceTime = basisCache->getSideBasisCache(sideOrdinal)->getSideNormalsSpaceTime();
      int numPoints = sideNormalsSpaceTime.dimension(1);

      // check that the normals are unit length:
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        double lengthSquared = 0;
        for (int d=0; d<spaceTimeMesh->getDimension(); d++)
        {
          lengthSquared += sideNormalsSpaceTime(0,ptOrdinal,d) * sideNormalsSpaceTime(0,ptOrdinal,d);
        }
        double length = sqrt(lengthSquared);
        TEST_FLOATING_EQUALITY(length,1.0,tol);
      }

      if (cell->getNeighbor(sideOrdinal,spaceTimeMesh->getTopology()) != Teuchos::null)
      {
        // then we also want to check that pointwise the normals are opposite each other
        pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal,spaceTimeMesh->getTopology());
        GlobalIndexType neighborCellID = neighborInfo.first;
        unsigned neighborSide = neighborInfo.second;
        BasisCachePtr neighborBasisCache = BasisCache::basisCacheForCell(spaceTimeMesh,neighborCellID);
        FieldContainer<double> neighborSideNormals = neighborBasisCache->getSideBasisCache(neighborSide)->getSideNormalsSpaceTime();

        // NOTE: here we implicitly assume that the normals at each point will be the same, because we don't
        //       do anything to make neighbors' physical points come in the same order.  For now, this is true
        //       of our test meshes.
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          for (int d=0; d<spaceTimeMesh->getDimension(); d++)
          {
            double cell_d = sideNormalsSpaceTime(0,ptOrdinal,d);
            double neighbor_d = neighborSideNormals(0,ptOrdinal,d);
            TEST_FLOATING_EQUALITY(cell_d, -neighbor_d, tol);
          }
        }
      }
    }
  }
}

void testSaveAndLoad2D(BFPtr bf, Teuchos::FancyOStream &out, bool &success)
{
  int H1Order = 2;
  vector<int> elemCounts = {3,2};
  vector<double> dims = {1.0,2.0};

  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dims, elemCounts, H1Order);

  string meshFile = "SavedMesh.HDF5";
  mesh->saveToHDF5(meshFile);

  MeshPtr loadedMesh = MeshFactory::loadFromHDF5(bf, meshFile);
  TEST_EQUALITY(loadedMesh->globalDofCount(), mesh->globalDofCount());

  // delete the file we created
  remove(meshFile.c_str());

  // just to confirm that we can manipulate the loaded mesh:
  set<GlobalIndexType> cellsToRefine;
  cellsToRefine.insert(0);
  loadedMesh->pRefine(cellsToRefine);
}
  
  TEUCHOS_UNIT_TEST( Mesh, ProjectFieldSolution )
  {
    double tol = 1e-15;
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 2;
    vector<int> elemCounts = {3,2};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,2.0}, elemCounts, H1Order);
    
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    map<int, FunctionPtr> solutionMap;
    FunctionPtr exactFxn = Function::constant(1.0);
    VarPtr phi = form.phi();
    solutionMap[phi->ID()] = exactFxn;
    
    solution->projectOntoMesh(solutionMap);
    
    FunctionPtr solnFxn = Function::solution(phi, solution, false);
    double err = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(err, <, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Mesh, ProjectSolutionOnRefinement )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 2;
    vector<int> elemCounts = {3,2};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,2.0}, elemCounts, H1Order);

    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    mesh->registerSolution(solution);
    
    map<int, FunctionPtr> solutionMap;
    FunctionPtr x = Function::xn();
    solutionMap[form.phi()->ID()] = x;
    solutionMap[form.psi()->ID()] = Function::constant({1.0,0.0});
    solutionMap[form.phi_hat()->ID()] = x;
    FunctionPtr n = Function::normal();
    FunctionPtr n_parity = Function::normal() * Function::sideParity();
    solutionMap[form.psi_n_hat()->ID()] = Function::constant({1.0,0.0}) * n_parity;
    
    solution->projectOntoMesh(solutionMap);
    
    // sanity check: make sure that the difference *before* refinement is 0
    double tol = 1e-14;
    for (auto entry : solutionMap)
    {
      int varID = entry.first;
      FunctionPtr exactFxn = entry.second;
      VarPtr var = form.bf()->varFactory()->trial(varID);
      FunctionPtr solnFxn = Function::solution(var, solution, false);
      double err = (solnFxn - exactFxn)->l2norm(mesh);
      TEUCHOS_TEST_COMPARE(err, <, tol, out, success);
      out << "Before refinement, err for variable " << var->name() << ": " << err << endl;
    }
    
    // now, refine uniformly:
    RefinementStrategy::hRefineUniformly(mesh);
    
    for (auto entry : solutionMap)
    {
      int varID = entry.first;
      FunctionPtr exactFxn = entry.second;
      VarPtr var = form.bf()->varFactory()->trial(varID);
      FunctionPtr solnFxn = Function::solution(var, solution, false);
      double err = (solnFxn - exactFxn)->l2norm(mesh);
      TEUCHOS_TEST_COMPARE(err, <, tol, out, success);
      out << "After refinement, err for variable " << var->name() << ": " << err << endl;
    }
  }
  

TEUCHOS_UNIT_TEST( Mesh, SaveAndLoadPoissonConforming )
{
  int spaceDim = 2;
  bool conformingTraces = true;
  PoissonFormulation form(spaceDim,conformingTraces);
  testSaveAndLoad2D(form.bf(), out, success);
}

TEUCHOS_UNIT_TEST( Mesh, SaveAndLoadStokesConforming )
{
  int spaceDim = 2;
  bool conformingTraces = true;
  double mu = 1.0;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim,mu,conformingTraces);
  testSaveAndLoad2D(form.bf(), out, success);
}
} // namespace
