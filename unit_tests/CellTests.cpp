//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CellTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/18/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Intrepid_FieldContainer.hpp"

#include "CamelliaCellTools.h"
#include "Cell.h"
#include "MeshFactory.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
TEUCHOS_UNIT_TEST( Cell, FindSubcellOrdinalInSide )
{
  double width=1.0, height=1.0;
  int horizontalElements=2, verticalElements=1;
  MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(width,height,horizontalElements,verticalElements);

  int sideDim = meshTopo->getDimension() - 1;
  for (auto cellIndex : meshTopo->getLocallyKnownActiveCellIndices())
  {
    CellPtr cell = meshTopo->getCell(cellIndex);
    CellTopoPtr cellTopo = cell->topology();
    for (int sideOrdinal=0; sideOrdinal<cell->getSideCount(); sideOrdinal++)
    {
      IndexType sideEntityIndex = cell->getEntityIndices(sideDim)[sideOrdinal];
      for (int subcdim=0; subcdim <= sideDim; subcdim++)
      {
        int subcCount = meshTopo->getSubEntityCount(sideDim, sideEntityIndex, subcdim);
        for (int subcOrdinalInGlobalSide=0; subcOrdinalInGlobalSide<subcCount; subcOrdinalInGlobalSide++)
          // subcOrdinalGlobal as opposed to as ordered by the cell side
        {
          IndexType subcEntityIndex = meshTopo->getSubEntityIndex(sideDim, sideEntityIndex, subcdim, subcOrdinalInGlobalSide);
          IndexType subcOrdinalInCellExpected = cell->findSubcellOrdinal(subcdim, subcEntityIndex);
          IndexType subcOrdinalInCellSide = cell->findSubcellOrdinalInSide(subcdim, subcEntityIndex, sideOrdinal);
          IndexType subcOrdinalInCellActual = CamelliaCellTools::subcellOrdinalMap(cellTopo, sideDim, sideOrdinal, subcdim, subcOrdinalInCellSide);
          TEST_EQUALITY(subcOrdinalInCellActual, subcOrdinalInCellExpected);
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( Cell, Neighbors1D )
{
  int numCells = 8;
  double xLeft = 0, xRight = 1;
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology( xLeft, xRight, numCells );

  int numBoundarySides = 0;

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    CellPtr cell = meshTopo->getCell(cellIndex);

    TEST_ASSERT(cell->getSideCount() == 2);

    for (int sideOrdinal = 0; sideOrdinal < cell->getSideCount(); sideOrdinal++)
    {
      pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal,meshTopo);
      if (neighborInfo.first == -1)
      {
        numBoundarySides++;
      }
      else
      {
        CellPtr neighbor = meshTopo->getCell(neighborInfo.first);
        unsigned sideOrdinalInNeighbor = neighborInfo.second;
        pair<GlobalIndexType, unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(sideOrdinalInNeighbor,meshTopo);
        TEST_ASSERT(neighborNeighborInfo.first == cellIndex);
      }
    }
  }
  TEST_ASSERT(numBoundarySides == 2);
}
  
  TEUCHOS_UNIT_TEST( Cell, VertexNeighbors_Triangles )
  {
    bool useTriangles = true;
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,1};
    
    MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                                             elementCounts[0], elementCounts[1], useTriangles);
    
    GlobalIndexType cellID = 1, neighborCellID = 0;
    CellPtr cell = meshTopo->getCell(cellID);
    unsigned vertexDim = 0;
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(vertexDim, meshTopo);
    
    TEST_EQUALITY(neighborIDs.size(), 1);
    TEST_EQUALITY(*neighborIDs.begin(), neighborCellID);
    
    // something similar, but now use 2x2 quad mesh divided into triangles
    // the "0" triangle (SE of bottom-left quad) shares a vertex with all but the top left triangle
    cellID = 0;
    elementCounts = {2,2};
    
    meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                             elementCounts[0], elementCounts[1], useTriangles);
    
    cell = meshTopo->getCell(cellID);
    neighborIDs = cell->getActiveNeighborIndices(vertexDim, meshTopo);
    
    // "-2" below is to exclude cell 0 as well as the top left triangle
    int neighborsExpected = elementCounts[0] * elementCounts[1] * 2 - 2;
    
    TEST_EQUALITY(neighborIDs.size(), neighborsExpected);
  }
  
  TEUCHOS_UNIT_TEST( Cell, VertexNeighbors_HangingNodeTriangle )
  {
    bool useTriangles = true;
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,1};
    MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                                             elementCounts[0], elementCounts[1], useTriangles);
    IndexType firstChildCellIndex = meshTopo->cellCount();
    IndexType cellToRefine = 0;
    RefinementPatternPtr triangleRefPattern = RefinementPattern::regularRefinementPatternTriangle();
    meshTopo->refineCell(cellToRefine, triangleRefPattern, firstChildCellIndex);
    
    // find the edge the two original cells share:
    IndexType sharedEdgeEntityIndex = -1;
    CellPtr cell0 = meshTopo->getCell(0);
    CellPtr cell1 = meshTopo->getCell(1);
    int edgeCount = cell0->topology()->getEdgeCount();
    int vertexDim = 0;
    int edgeDim = 1;
    for (int edgeOrdinal=0; edgeOrdinal<edgeCount; edgeOrdinal++)
    {
      if (cell0->getNeighbor(edgeOrdinal, meshTopo) != Teuchos::null)
      {
        sharedEdgeEntityIndex = cell0->entityIndex(edgeDim, edgeOrdinal);
      }
    }
    // find the new vertex created by breaking that shared edge
    vector<IndexType> edgeChildren = meshTopo->getChildEntities(edgeDim, sharedEdgeEntityIndex);
    TEUCHOS_TEST_FOR_EXCEPT(edgeChildren.size() != 2);
    vector<IndexType> firstChildNodes = meshTopo->getEntityVertexIndices(edgeDim, edgeChildren[0]);
    vector<IndexType> secondChildNodes = meshTopo->getEntityVertexIndices(edgeDim, edgeChildren[1]);
    
    IndexType newVertexIndex;
    if ((firstChildNodes[0] == secondChildNodes[0]) || (firstChildNodes[0] == secondChildNodes[1]))
    {
      newVertexIndex = firstChildNodes[0];
    }
    else if ((firstChildNodes[1] == secondChildNodes[0]) || (firstChildNodes[1] == secondChildNodes[1]))
    {
      newVertexIndex = firstChildNodes[1];
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "new vertex not found!");
    }
    // which active cells match the new vertex?  All such cells should be included among cell1's vertex neighbors
    set<pair<IndexType,unsigned>> matchingCellEntries = meshTopo->getCellsContainingEntity(vertexDim, newVertexIndex);
    set<IndexType> matchingCellIDs;
    for (auto entry : matchingCellEntries)
    {
      matchingCellIDs.insert(entry.first);
    }
    
    set<GlobalIndexType> neighborIDs = cell1->getActiveNeighborIndices(vertexDim, meshTopo);

    for (IndexType newVertexCellID : matchingCellIDs)
    {
      if (neighborIDs.find(newVertexCellID) == neighborIDs.end())
      {
        success = false;
        out << newVertexCellID << " not found in neighborIDs.\n";
      }
    }
    print(out, "matchingCellIDs", matchingCellIDs);
    print(out, "neighborIDs", neighborIDs);
  }
} // namespace
