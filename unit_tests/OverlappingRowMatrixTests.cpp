//
//  OverlappingRowMatrixTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/24/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "CamelliaDebugUtility.h"
#include "Mesh.h"
#include "MeshFactory.h"
#include "OverlappingRowMatrix.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

using namespace Camellia;
using namespace Intrepid;

namespace
{
  void testNeighborRelationshipIsSymmetric(MeshPtr mesh, int dimensionToUseForNeighbor, int overlapLevel, bool hierarchical,
                                           Teuchos::FancyOStream &out, bool &success)
  {
    set<GlobalIndexType> activeCells = mesh->getActiveCellIDs();
    
    GlobalIndexType maxCellID = *max_element(activeCells.begin(), activeCells.end());
    
    FieldContainer<int> adjacency(maxCellID+1,maxCellID+1); // we assume that in our tests we don't ever get too many cells!
    
    for (GlobalIndexType cellID : activeCells)
    {
      set<GlobalIndexType> neighbors = OverlappingRowMatrix::overlappingCells(cellID, mesh, overlapLevel,hierarchical,
                                                                              dimensionToUseForNeighbor);
      for (GlobalIndexType neighborID: neighbors)
      {
        adjacency(cellID,neighborID) = 1;
      }
    }
    
    for (int i=0; i<=maxCellID; i++)
    {
      for (int j=i; j<=maxCellID; j++)
      {
        TEST_EQUALITY(adjacency(i,j),adjacency(j,i));
        
        if (adjacency(i,j) != adjacency(j,i))
        {
          if (adjacency(i,j) == 1)
            out << "cell " << i << " sees " << " cell " << j << " as a neighbor, but cell " << j << " doesn't agree!\n";
          else
            out << "cell " << j << " sees " << " cell " << i << " as a neighbor, but cell " << i << " doesn't agree!\n";
        }
      }
    }
  }
  
  set<GlobalIndexType> getOverlapCells(MeshPtr mesh, int overlap, bool hierarchical)
  {
    set<GlobalIndexType> cells = mesh->cellIDsInPartition();
    
    // go outward #overlap levels of cells
    set<GlobalIndexType> ghostCells;
    set<GlobalIndexType> lastGhostCells = cells;
    for (int i=0; i<overlap; i++)
    {
      for (GlobalIndexType cellID : lastGhostCells)
      {
        CellPtr cell = mesh->getTopology()->getCell(cellID);
        vector< CellPtr > neighbors = cell->getNeighbors(mesh->getTopology());
        for (CellPtr neighbor : neighbors)
        {
          if (cells.find(neighbor->cellIndex()) == cells.end())
          {
            ghostCells.insert(neighbor->cellIndex());
            cells.insert(neighbor->cellIndex());
          }
        }
      }
      lastGhostCells = ghostCells;
    }
    
    if (hierarchical)
    {
      set<GlobalIndexType> neighborCells = cells;
      cells = mesh->cellIDsInPartition();
      
      // go upward #overlap levels of cells:
      set<GlobalIndexType> ancestralCells = cells;
      set<GlobalIndexType> lastAncestralCells = cells;
      for (int i=0; i<overlap; i++)
      {
        ancestralCells.clear();
        for (set<GlobalIndexType>::iterator cellIDIt = lastAncestralCells.begin();
             cellIDIt != lastAncestralCells.end(); cellIDIt++)
        {
          GlobalIndexType cellID = *cellIDIt;
          CellPtr cell = mesh->getTopology()->getCell(cellID);
          if (cell->getParent() == Teuchos::null)
          {
            // no parent: cell is its own ancestor
            ancestralCells.insert(cellID);
          }
          else
          {
            ancestralCells.insert(cell->getParent()->cellIndex());
          }
        }
        lastAncestralCells = ancestralCells;
      }
      
      //          Camellia::print("ancestralCells", ancestralCells);
      
      // once we've determined our partition's cell ancestors, we want to intersect their
      // descendants with the non-hierarchical ghost region (neighborCells)
      cells.clear();
      for (set<GlobalIndexType>::iterator cellIDIt = ancestralCells.begin();
           cellIDIt != ancestralCells.end(); cellIDIt++)
      {
        GlobalIndexType cellID = *cellIDIt;
        CellPtr cell = mesh->getTopology()->getCell(cellID);
        set<IndexType> descendants = cell->getDescendants(mesh->getTopology());
        
        for (IndexType descendant : descendants)
        {
          if (neighborCells.find(descendant) != neighborCells.end())
          {
            cells.insert(descendant);
          }
        }
      }
      //      Camellia::print("cells", cells);
      //      Camellia::print("neighborCells", neighborCells);
    }
    return cells;
  }
  
  void testOverlapCells(bool hierarchical, Teuchos::FancyOStream &out, bool &success)
  {
    // test that checks that the dof ordinals for overlap are correctly identified
    // when OverlappingRowMatrix is constructed by passing in a Mesh.
    
    // to fully exploit this test, should be run in context of 1, 2, 4, 16 MPI ranks.
    
    Teuchos::ParameterList pl;
    
    int spaceDim = 2;
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim,useConformingTraces);
    bool useStaticCondensation = false;
    
    int H1Order = 1;
    int delta_k = 2;
    int horizontalCells = 1;
    int verticalCells = 1;
    double width = 1.0;
    double height = 1.0;
    
    BFPtr poissonBilinearForm = form.bf();
    
    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", useConformingTraces);
    pl.set("x0",(double)0);
    pl.set("y0",(double)0);
    pl.set("width", width);
    pl.set("height",height);
    
    MeshPtr mesh = MeshFactory::quadMesh(pl);
    
    int sideDim = mesh->getTopology()->getDimension() - 1;
    
    // refine uniformly twice:
    mesh->hRefine(mesh->getTopology()->getActiveCellIndices());
    mesh->hRefine(mesh->getTopology()->getActiveCellIndices());
    
    SolutionPtr soln = Solution::solution(mesh, BC::bc(), RHS::rhs(), form.bf()->graphNorm());
    soln->setUseCondensedSolve(useStaticCondensation);
    
    for (int overlap=0; overlap<3; overlap++)
    {
      set<GlobalIndexType> expectedCells = getOverlapCells(mesh, overlap, hierarchical);
      set<GlobalIndexType> actualCells;
      
      set<GlobalIndexType> cellsInPartition = mesh->cellIDsInPartition();
      for (GlobalIndexType cellID : cellsInPartition)
      {
        set<GlobalIndexType> cellNeighbors = OverlappingRowMatrix::overlappingCells(cellID, mesh, overlap, hierarchical,
                                                                                    sideDim);
        actualCells.insert(cellNeighbors.begin(),cellNeighbors.end());
      }
      
      TEST_EQUALITY(expectedCells.size(), actualCells.size());
      
      if (expectedCells.size() == actualCells.size())
      {
        set<GlobalIndexType>::iterator actualIt = actualCells.begin();
        
        for (set<GlobalIndexType>::iterator expectedIt = expectedCells.begin();
             expectedIt != expectedCells.end(); expectedIt++, actualIt++)
        {
          GlobalIndexType expectedIndex = *expectedIt;
          GlobalIndexType actualIndex = *actualIt;
          TEST_EQUALITY(expectedIndex, actualIndex);
        }
      }
      else
      {
        set<GlobalIndexType> myCells = mesh->cellIDsInPartition();
        Camellia::print("myCells", myCells);
        Camellia::print("expectedCells", expectedCells);
        Camellia::print("actualCells", actualCells);
      }
    }
  }
  
  
  void testOverlapDofOrdinals(bool hierarchical, bool useStaticCondensation, Teuchos::FancyOStream &out, bool &success)
  {
    // test that checks that the dof ordinals for overlap are correctly identified
    // when OverlappingRowMatrix is constructed by passing in a Mesh.
    
    // to fully exploit this test, should be run in context of 1, 2, 4, 16 MPI ranks.
    
    Teuchos::ParameterList pl;
    
    int spaceDim = 2;
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim,useConformingTraces);
    
    int H1Order = 1;
    int delta_k = 2;
    int horizontalCells = 1;
    int verticalCells = 1;
    double width = 1.0;
    double height = 1.0;
    
    BFPtr poissonBilinearForm = form.bf();
    
    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", useConformingTraces);
    pl.set("x0",(double)0);
    pl.set("y0",(double)0);
    pl.set("width", width);
    pl.set("height",height);
    
    MeshPtr mesh = MeshFactory::quadMesh(pl);
    
    // refine uniformly twice:
    mesh->hRefine(mesh->getTopology()->getActiveCellIndices());
    mesh->hRefine(mesh->getTopology()->getActiveCellIndices());
    
    SolutionPtr soln = Solution::solution(mesh, BC::bc(), RHS::rhs(), form.bf()->graphNorm());
    
    soln->setUseCondensedSolve(useStaticCondensation);
    
    soln->initializeStiffnessAndLoad();
    soln->populateStiffnessAndLoad();
    
    Teuchos::RCP<DofInterpreter> dofInterpreter = soln->getDofInterpreter();
    
    Teuchos::RCP<Epetra_RowMatrix> stiffness = soln->getStiffnessMatrix();
    
    set<GlobalIndexType> myCells = mesh->cellIDsInPartition();
//    print("myCells", myCells);
    
    for (int overlap=0; overlap<3; overlap++)
    {
      set<GlobalIndexType> cells = getOverlapCells(mesh, overlap, hierarchical);
      
      vector<GlobalIndexType> cellsVector(cells.begin(),cells.end());
//      print("test cells", cells);
      set<GlobalIndexType> expectedGlobalDofIndices = dofInterpreter->importGlobalIndicesForCells(cellsVector);
      
      //        set<GlobalIndexType> expectedGlobalDofIndices;
      //        for (set<GlobalIndexType>::iterator cellIDIt = cells.begin(); cellIDIt != cells.end(); cellIDIt++) {
      //          GlobalIndexType cellID = *cellIDIt;
      //          set<GlobalIndexType> cellIndices = dofInterpreter->globalDofIndicesForCell(cellID);
      //          expectedGlobalDofIndices.insert(cellIndices.begin(), cellIndices.end());
      //        }
      
      // DEBUGGING output...
      //        if (useStaticCondensation) {
      //          ostringstream description;
      //          description << "expectedGlobalDofIndices for overlap " << overlap << " on rank " << rank;
      //          Camellia::print(description.str().c_str(), expectedGlobalDofIndices);
      //        }
      
      OverlappingRowMatrix rowMatrix(stiffness, overlap, mesh, dofInterpreter, hierarchical);
      set<GlobalIndexType> actualGlobalDofIndices = rowMatrix.RowIndices();
      
      TEST_EQUALITY(expectedGlobalDofIndices.size(), actualGlobalDofIndices.size());
      
      if (expectedGlobalDofIndices.size() == actualGlobalDofIndices.size())
      {
        set<GlobalIndexType>::iterator actualIt = actualGlobalDofIndices.begin();
        
        for (set<GlobalIndexType>::iterator expectedIt = expectedGlobalDofIndices.begin();
             expectedIt != expectedGlobalDofIndices.end(); expectedIt++, actualIt++)
        {
          GlobalIndexType expectedIndex = *expectedIt;
          GlobalIndexType actualIndex = *actualIt;
          TEST_EQUALITY(expectedIndex, actualIndex);
        }
      }
      else
      {
        print("expectedGlobalDofIndices", expectedGlobalDofIndices);
        print("actualGlobalDofIndices", actualGlobalDofIndices);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( OverlappingRowMatrix, OverlapCells )
  {
    bool hierarchical = false;
    testOverlapCells(hierarchical, out, success);
  }
  
  TEUCHOS_UNIT_TEST( OverlappingRowMatrix, HierarchicalOverlapCells )
  {
    bool hierarchical = true;
    testOverlapCells(hierarchical, out, success);
  }
  
  TEUCHOS_UNIT_TEST( OverlappingRowMatrix, OverlapDofOrdinals )
  {
    bool hierarchical = false;
    bool useStaticCondensation = false;
    testOverlapDofOrdinals(hierarchical, useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( OverlappingRowMatrix, HierarchicalOverlapDofOrdinals )
  {
    bool hierarchical = true;
    bool useStaticCondensation = false;
    testOverlapDofOrdinals(hierarchical, useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( OverlappingRowMatrix, OverlapDofOrdinalsCondensed )
  {
    bool hierarchical = false;
    bool useStaticCondensation = true;
    testOverlapDofOrdinals(hierarchical, useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( OverlappingRowMatrix, HierarchicalOverlapDofOrdinalsCondensed )
  {
    bool hierarchical = true;
    bool useStaticCondensation = true;
    testOverlapDofOrdinals(hierarchical, useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( OverlappingRowMatrix, OneNeighborsAreSymmetric )
  {
    int H1Order = 1;
    int delta_k = 2;
    int horizontalCells = 2;
    int verticalCells = 1;
    double width = 1.0;
    double height = 1.0;
    
    int spaceDim = 2;
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim,useConformingTraces);
    
    BFPtr poissonBilinearForm = form.bf();
    
    Teuchos::ParameterList pl;
    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", useConformingTraces);
    pl.set("x0",(double)0);
    pl.set("y0",(double)0);
    pl.set("width", width);
    pl.set("height",height);
    
    MeshPtr mesh = MeshFactory::quadMesh(pl);
    
    mesh->hRefine(set<GlobalIndexType>{1}); // create a hanging node
    
    int d = 1;
    int overlapLevel = 1;
    int hierarchical = false;
    
    // TODO: test some other cases...
    testNeighborRelationshipIsSymmetric(mesh, d, overlapLevel, hierarchical, out, success);
  }
  
  TEUCHOS_UNIT_TEST( OverlappingRowMatrix, TestOverlapValues )
  {
    // test checks that the rows in the overlapping matrix match what's expected
    // to fully exploit this test, should be run in context of 1, 2, 4, 16 MPI ranks.
    
    int rank = Teuchos::GlobalMPISession::getRank();
    
    int n = 16;
    FieldContainer<double> denseMatrix(n,n);
    double value = 0;
    // fill with arbitrary data, taking care to give each entry a unique value, to maximize chances of detecting failure
    FieldContainer<int> allColumnIndices(n);
    for (int i=0; i<n; i++)
    {
      allColumnIndices(i) = i;
      for (int j=0; j<n; j++)
      {
        denseMatrix(i,j) = value;
        value += 1.0;
      }
    }
    
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    //cout << "rank: " << rank << " of " << numProcs << endl;
#else
    Epetra_SerialComm Comm;
#endif
    
    Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
    
    set<GlobalIndexType> myGlobalDofs;
    
    Epetra_Map map(n, 0, Comm); // define distribution
    Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, map, n));
    for (int LID=0; LID<map.NumMyElements(); LID++)
    {
      int GID = map.GID(LID);
      matrix->InsertGlobalValues(GID, n, &denseMatrix(GID,0), &allColumnIndices(0));
      myGlobalDofs.insert(GID);
    }
    matrix->FillComplete();
    
    double floatingTol = 1e-15;
    
    // sanity check for this test: make sure that entries in matrix match denseMatrix
    for (int LID=0; LID<map.NumMyElements(); LID++)
    {
      int GID = map.GID(LID);
      FieldContainer<double> rowData(n);
      int numEntries;
      FieldContainer<int> indices(n);
      matrix->ExtractGlobalRowCopy(GID, n, numEntries, &rowData(0), &indices(0));
      TEST_EQUALITY(n, numEntries); // row should be full
      for (int j=0; j<numEntries; j++)
      {
        int colIndex = indices(j);
        TEST_FLOATING_EQUALITY(rowData(j), denseMatrix(GID,colIndex), floatingTol);
      }
    }
    
    // define some arbitrary overlap distributions for testing
    vector< set<GlobalIndexType> > overlapDistributions;
    {
      set<GlobalIndexType> dist1 = myGlobalDofs;
      overlapDistributions.push_back(dist1);
      
      set<GlobalIndexType> dist2 = myGlobalDofs;
      dist1.insert(0);
      dist1.insert(1);
      overlapDistributions.push_back(dist2);
      
      set<GlobalIndexType> dist3 = myGlobalDofs;
      dist2.insert(n-2);
      dist2.insert(n-1);
      overlapDistributions.push_back(dist3);
    }
    
    int maxOverlapLevel = 1;
    for (vector< set<GlobalIndexType> >::iterator distributionIt = overlapDistributions.begin(); distributionIt != overlapDistributions.end(); distributionIt++)
    {
      set<GlobalIndexType> myRowIndices = *distributionIt;
      OverlappingRowMatrix overlapMatrix(matrix, maxOverlapLevel, myRowIndices);
      
      int expectedLocalRowCount = myRowIndices.size();
      int actualLocalRowCount = overlapMatrix.Map().NumMyElements();
      
      TEST_EQUALITY(expectedLocalRowCount, actualLocalRowCount);
      
      for (int localRowOrdinal=0; localRowOrdinal < actualLocalRowCount; localRowOrdinal++)
      {
        GlobalIndexType row = overlapMatrix.Map().GID(localRowOrdinal);
        FieldContainer<double> rowData(n);
        int numEntries;
        FieldContainer<int> indices(n);
        overlapMatrix.ExtractMyRowCopy(localRowOrdinal, n, numEntries, &rowData(0), &indices(0));
        TEST_EQUALITY(myRowIndices.size(), numEntries); // should have entries for every locally known column
        if (myRowIndices.size() == numEntries)
        {
          for (int j=0; j<numEntries; j++)
          {
            int col_lid = indices(j);
            int colIndex = overlapMatrix.RowMatrixColMap().GID(col_lid);
            TEST_FLOATING_EQUALITY(rowData(j), denseMatrix(row,colIndex), floatingTol);
          }
        }
        else
        {
          cout << "On rank " << rank << ", global row " << row << ", n != numEntries; " << n << " != " << numEntries << endl;
        }
      }
    }
  }
} // namespace