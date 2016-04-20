//
//  TestTemplate
//  Camellia
//
//  Created by Nate Roberts on 3/31/16.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "CellDataMigration.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "PoissonFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;
using namespace std;

namespace
{
  typedef pair<RefinementBranch,vector<GlobalIndexType>> LabeledRefinementBranch; // first cellID indicates the root cell ID; after that, each cellID indicates first child of each refinement
  typedef pair<LabeledRefinementBranch, vector<vector<double>> > RootedLabeledRefinementBranch; // second contains vertex coordinates for root cell

  bool verticesEquals(vector<vector<double>> &vertices1, vector<vector<double>> &vertices2)
  {
    double tol = 1e-15;
    if (vertices1.size() != vertices2.size()) return false;
    for (int i=0; i<vertices1.size(); i++)
    {
      if (vertices1[i].size() != vertices2[i].size()) return false;
      for (int j=0; j<vertices1[i].size(); j++)
      {
        double diff = abs(vertices1[i][j]-vertices2[i][j]);
        if (diff > tol) return false;
      }
    }
    return true;
  }
  
  bool labelsEquals(vector<GlobalIndexType> &labels1, vector<GlobalIndexType> &labels2)
  {
    if (labels1.size() != labels2.size()) return false;
    for (int i=0; i<labels1.size(); i++)
    {
      if (labels1[i] != labels2[i]) return false;
    }
    return true;
  }
  
  bool refBranchEquals(RefinementBranch &refBranch1, RefinementBranch &refBranch2)
  {
    if (refBranch1.size() != refBranch2.size()) return false;
    for (int i=0; i<refBranch1.size(); i++)
    {
      RefinementPattern* refPattern1 = refBranch1[i].first;
      RefinementPattern* refPattern2 = refBranch2[i].first;
      if (refPattern1->getKey() != refPattern2->getKey()) return false;
      unsigned childOrdinal1 = refBranch1[i].second;
      unsigned childOrdinal2 = refBranch2[i].second;
      if (childOrdinal1 != childOrdinal2) return false;
    }
    return true;
  }
  
  bool labeledRefBranchEquals(LabeledRefinementBranch &refBranch1, LabeledRefinementBranch &refBranch2)
  {
    bool refBranchesMatch = refBranchEquals(refBranch1.first, refBranch2.first);
    bool labelsMatch = labelsEquals(refBranch1.second, refBranch2.second);
    return refBranchesMatch && labelsMatch;
  }
  
  bool rootedRefBranchEquals(RootedLabeledRefinementBranch &refBranch1, RootedLabeledRefinementBranch &refBranch2)
  {
    bool labeledRefBranchesMatch = labeledRefBranchEquals(refBranch1.first, refBranch2.first);
    bool verticesMatch = verticesEquals(refBranch1.second, refBranch2.second);
    return verticesMatch && labeledRefBranchesMatch;
  }
  
  bool rootedRefBranchVectorEquals(vector<RootedLabeledRefinementBranch> &refBranches1, vector<RootedLabeledRefinementBranch> &refBranches2)
  {
    if (refBranches1.size() != refBranches2.size()) return false;
    
    for (int i=0; i<refBranches1.size(); i++)
    {
      if (! rootedRefBranchEquals(refBranches1[i], refBranches2[i]))
      {
        return false;
      }
    }
    
    return true;
  }
  
  void testPackAndUnpackGeometry(MeshPtr mesh, Teuchos::FancyOStream &out, bool &success)
  {
    /* For every active cell in mesh, do the following:
     1. Get the data size.
     2. Allocate a buffer that is 3x the data size, initialized with -1 values.
     3. Pack the buffer.
     4. Check that the written size is exactly the advertised size.
     5. Unpack from the buffer (returns a vector<RootedLabeledRefinementBranch>).
     6. Compare the return value with CellDataMigration::getCellHaloGeometry().
     */
    
    MPIWrapper::CommWorld()->Barrier();
    
    const set<GlobalIndexType>* myCellIndices = &mesh->cellIDsInPartition();
    
    for (GlobalIndexType cellID : *myCellIndices)
    {
      int size = CellDataMigration::geometryDataSize(mesh.get(), cellID);
      vector<char> dataBuffer(size*3,-1);
      char* dataLocation = &dataBuffer[0];
      CellDataMigration::packGeometryData(mesh.get(), cellID, dataLocation, size);
      int sizeWritten = dataLocation - &dataBuffer[0];
      TEST_EQUALITY(sizeWritten, size);
      const char *constDataLocation = &dataBuffer[0];
      vector<RootedLabeledRefinementBranch> unpackedBranch;
      CellDataMigration::unpackGeometryData(mesh.get(), cellID, constDataLocation, size, unpackedBranch);
      int sizeRead = constDataLocation - &dataBuffer[0];
      TEST_EQUALITY(sizeRead, size);
      vector<RootedLabeledRefinementBranch> expectedBranch;
      CellDataMigration::getCellHaloGeometry(mesh.get(), cellID, expectedBranch);
      TEST_ASSERT(rootedRefBranchVectorEquals(unpackedBranch, expectedBranch));
    }
  }
  
  TEUCHOS_UNIT_TEST( CellDataMigration, PackAndUnpackPureGeometry_1D )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 1;
    bool useConformingTraces = true;
    int H1Order = 2;
    int meshWidth = 2;
    PoissonFormulation form(spaceDim,useConformingTraces,PoissonFormulation::ULTRAWEAK);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0}, {meshWidth}, H1Order);
    
    // do a few refinements:
    // start with a non-uniform one, then refine uniformly from there
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    cellsToRefine = mesh->getActiveCellIDsGlobal();
    mesh->hRefine(cellsToRefine);
    
    cellsToRefine = mesh->getActiveCellIDsGlobal();
    mesh->hRefine(cellsToRefine);
    
    testPackAndUnpackGeometry(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( CellDataMigration, PackAndUnpackPureGeometry_2D )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    bool useConformingTraces = true;
    int H1Order = 2;
    int meshWidth = 2;
    PoissonFormulation form(spaceDim,useConformingTraces,PoissonFormulation::ULTRAWEAK);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {meshWidth,meshWidth}, H1Order);
    
    // do a few refinements:
    // start with a non-uniform one, then refine uniformly from there
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);

    cellsToRefine = mesh->getActiveCellIDsGlobal();
    mesh->hRefine(cellsToRefine);
    
    testPackAndUnpackGeometry(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( CellDataMigration, PackAndUnpackPureGeometry_3D )
  {
    int spaceDim = 3;
    bool useConformingTraces = true;
    int H1Order = 2;
    int meshWidth = 2;
    PoissonFormulation form(spaceDim,useConformingTraces,PoissonFormulation::ULTRAWEAK);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0,1.0}, {meshWidth,meshWidth,meshWidth}, H1Order);
    
    // do a few refinements:
    // start with a non-uniform one, then refine uniformly from there
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    cellsToRefine = mesh->getActiveCellIDsGlobal();
    mesh->hRefine(cellsToRefine);
    
    testPackAndUnpackGeometry(mesh, out, success);
  }
} // namespace
