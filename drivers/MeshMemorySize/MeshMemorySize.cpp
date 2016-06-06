#include "MeshTopology.h"
#include "MPIWrapper.h"
#include "PoissonFormulation.h"

#include <Teuchos_GlobalMPISession.hpp>

using namespace Camellia;
using namespace std;

vector<double> makeVertex(double v0)
{
  vector<double> v;
  v.push_back(v0);
  return v;
}

vector<double> makeVertex(double v0, double v1)
{
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  return v;
}

vector<double> makeVertex(double v0, double v1, double v2)
{
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  v.push_back(v2);
  return v;
}

vector< vector<double> > quadPoints(double x0, double y0, double width, double height)
{
  vector< vector<double> > v(4);
  v[0] = makeVertex(x0,y0);
  v[1] = makeVertex(x0 + width,y0);
  v[2] = makeVertex(x0 + width,y0 + height);
  v[3] = makeVertex(x0,y0 + height);
  return v;
}

vector< vector<double> > hexPoints(double x0, double y0, double z0, double width, double height, double depth)
{
  vector< vector<double> > v(8);
  v[0] = makeVertex(x0,y0,z0);
  v[1] = makeVertex(x0 + width,y0,z0);
  v[2] = makeVertex(x0 + width,y0 + height,z0);
  v[3] = makeVertex(x0,y0 + height,z0);
  v[4] = makeVertex(x0,y0,z0+depth);
  v[5] = makeVertex(x0 + width,y0,z0 + depth);
  v[6] = makeVertex(x0 + width,y0 + height,z0 + depth);
  v[7] = makeVertex(x0,y0 + height,z0 + depth);
  return v;
}

MeshTopologyPtr makeQuadMesh(double x0, double y0, double width, double height,
                             unsigned horizontalCells, unsigned verticalCells)
{
  unsigned spaceDim = 2;
  Teuchos::RCP<MeshTopology> meshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
  double dx = width / horizontalCells;
  double dy = height / verticalCells;
  CellTopoPtrLegacy quadTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
  for (unsigned i=0; i<horizontalCells; i++)
  {
    double x = x0 + dx * i;
    for (unsigned j=0; j<verticalCells; j++)
    {
      double y = y0 + dy * j;
      vector< vector<double> > vertices = quadPoints(x, y, dx, dy);
      meshTopo->addCell(quadTopo, vertices);
    }
  }
  return meshTopo;
}

MeshTopologyPtr makeHexMesh(double x0, double y0, double z0, double width, double height, double depth,
                            unsigned horizontalCells, unsigned verticalCells, unsigned depthCells)
{
  unsigned spaceDim = 3;
  Teuchos::RCP<MeshTopology> meshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
  double dx = width / horizontalCells;
  double dy = height / verticalCells;
  double dz = depth / depthCells;
  CellTopoPtrLegacy hexTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ) );
  for (unsigned i=0; i<horizontalCells; i++)
  {
    double x = x0 + dx * i;
    for (unsigned j=0; j<verticalCells; j++)
    {
      double y = y0 + dy * j;
      for (unsigned k=0; k<depthCells; k++)
      {
        double z = z0 + dz * k;
        vector< vector<double> > vertices = hexPoints(x, y, z, dx, dy, dz);
        meshTopo->addCell(hexTopo, vertices);
      }
    }
  }
  return meshTopo;
}

void refineUniformly(MeshTopologyPtr meshTopo)
{
  vector<IndexType> cellIndices = meshTopo->getActiveCellIndicesGlobal();
  for (IndexType cellID : cellIndices)
  {
    meshTopo->refineCell(cellID, RefinementPattern::regularRefinementPatternHexahedron(), meshTopo->cellCount());
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);

  Epetra_CommPtr Comm = MPIWrapper::CommWorld();
  int rank = Comm->MyPID();
  
  {
    // 2D
    int horizontalCells = 128;
    int verticalCells = 128;
    MeshTopologyPtr meshTopo = makeQuadMesh(0,0,1,1,horizontalCells,verticalCells);

    if (rank == 0) cout << "Approximate size of " << horizontalCells << " x " << verticalCells << " meshTopo is ";

    long long memoryFootprintInBytes = meshTopo->approximateMemoryFootprint();
    double memoryFootprintInMegabytes = (double)memoryFootprintInBytes / (1024 * 1024);
    cout << setprecision(4) << memoryFootprintInMegabytes << " MB.\n";

    meshTopo->printApproximateMemoryReport();

    CellPtr cell = meshTopo->getCell(0);
    cell->printApproximateMemoryReport();

    meshTopo = Teuchos::null;
    cell = Teuchos::null;
  }
  {
    // 3D
    int horizontalCells = 32;
    int verticalCells = 32;
    int depthCells = 32;

    double x0 = 0.0, y0 = 0.0, z0 = 0.0;
    int width = 1.0, height = 1.0, depth = 1.0;
    MeshTopologyPtr meshTopo = makeHexMesh(x0, y0, z0, width, height, depth,
                                       horizontalCells, verticalCells, depthCells);

    if (rank == 0) cout << "Approximate size of " << horizontalCells << " x " << verticalCells << " x " << depthCells << " meshTopo is ";

    long long memoryFootprintInBytes = meshTopo->approximateMemoryFootprint();
    double memoryFootprintInMegabytes = (double)memoryFootprintInBytes / (1024 * 1024);
    if (rank == 0) cout << setprecision(4) << memoryFootprintInMegabytes << " MB.\n";

    if (rank==0) meshTopo->printApproximateMemoryReport();

    CellPtr cell = meshTopo->getCell(0);
    if (rank==0) cell->printApproximateMemoryReport();

    meshTopo = Teuchos::null;
    cell = Teuchos::null;
  }

  {
    // 2D
    int horizontalCells = 1;
    int verticalCells = 1;
    MeshTopologyPtr meshTopo = makeQuadMesh(0,0,1,1,horizontalCells,verticalCells);

    while (horizontalCells < 128)
    {
      // uniform refinements

      RefinementPatternPtr regularQuadRefPattern = RefinementPattern::regularRefinementPatternQuad();

      vector<IndexType> activeCells = meshTopo->getActiveCellIndicesGlobal();
      for (IndexType cellID : activeCells)
      {
        meshTopo->refineCell(cellID, regularQuadRefPattern, meshTopo->cellCount());
      }

      horizontalCells *= 2;
      verticalCells *= 2;
    }

    if (rank == 0) cout << "Approximate size of " << horizontalCells << " x " << verticalCells << " meshTopo produced by refinements is ";

    long long memoryFootprintInBytes = meshTopo->approximateMemoryFootprint();
    double memoryFootprintInMegabytes = (double)memoryFootprintInBytes / (1024 * 1024);
    if (rank==0) cout << setprecision(4) << memoryFootprintInMegabytes << " MB.\n";

    if (rank==0) meshTopo->printApproximateMemoryReport();

    CellPtr cell = meshTopo->getCell(0);
    cell->printApproximateMemoryReport();

    meshTopo = Teuchos::null;
    cell = Teuchos::null;
  }
  {
    // 3D
    int horizontalCells = 32;
    int verticalCells = 32;
    int depthCells = 32;

    double x0 = 0.0, y0 = 0.0, z0 = 0.0;
    int width = 1.0, height = 1.0, depth = 1.0;
    MeshTopologyPtr meshTopo = makeHexMesh(x0, y0, z0, width, height, depth,
                                       horizontalCells, verticalCells, depthCells);

    if (rank==0) cout << "Approximate size of " << horizontalCells << " x " << verticalCells << " x " << depthCells << " meshTopo is ";

    long long memoryFootprintInBytes = meshTopo->approximateMemoryFootprint();
    double memoryFootprintInMegabytes = (double)memoryFootprintInBytes / (1024 * 1024);
    if (rank==0) cout << setprecision(4) << memoryFootprintInMegabytes << " MB.\n";

    if (rank==0) meshTopo->printApproximateMemoryReport();

    CellPtr cell = meshTopo->getCell(0);
    if (rank==0) cell->printApproximateMemoryReport();

    meshTopo = Teuchos::null;
    cell = Teuchos::null;
  }

  {
    // 3D
    int horizontalCells = 1;
    int verticalCells = 1;
    int depthCells = 1;

    double x0 = 0.0, y0 = 0.0, z0 = 0.0;
    int width = 1.0, height = 1.0, depth = 1.0;
    MeshTopologyPtr meshTopo = makeHexMesh(x0, y0, z0, width, height, depth,
                                       horizontalCells, verticalCells, depthCells);

    while (horizontalCells < 32)
    {
      // uniform refinements

      RefinementPatternPtr regularHexRefPattern = RefinementPattern::regularRefinementPatternHexahedron();

      vector<IndexType> activeCells = meshTopo->getActiveCellIndicesGlobal();
      for (IndexType cellID : activeCells)
      {
        meshTopo->refineCell(cellID, regularHexRefPattern, meshTopo->cellCount());
      }

      horizontalCells *= 2;
      verticalCells *= 2;
      depthCells *= 2;
    }

    if (rank==0) cout << "Approximate size of " << horizontalCells << " x " << verticalCells << " x " << depthCells << " meshTopo arrived at by refinements is ";

    long long memoryFootprintInBytes = meshTopo->approximateMemoryFootprint();
    double memoryFootprintInMegabytes = (double)memoryFootprintInBytes / (1024 * 1024);
    if (rank==0) cout << setprecision(4) << memoryFootprintInMegabytes << " MB.\n";

    if (rank==0) meshTopo->printApproximateMemoryReport();

//    CellPtr cell = meshTopo->getCell(0);
//    cell->printApproximateMemoryReport();

//    Epetra_CommPtr Comm = MPIWrapper::CommWorld();
//    int minRankWithActiveCell = Comm->NumProc();
    
//    IndexType someActiveCell = *meshTopo->cellIDsInPartition().begin();
//    int vertexDim = 0;
//    meshTopo->pruneToInclude({someActiveCell}, vertexDim);
//    
//    cout << "Approximate size of same meshTopo, but pruned to include first active cell, is ";
//    memoryFootprintInBytes = meshTopo->approximateMemoryFootprint();
//    double memoryFootprintInKilobytes = (double)memoryFootprintInBytes / 1024;
//    cout << setprecision(4) << memoryFootprintInKilobytes << " KB.\n";
//    meshTopo->printApproximateMemoryReport();
    
    bool conformingTraces = true;
    int spaceDim = meshTopo->getDimension();
    int H1Order = 2, delta_k = 2;
    
    PoissonFormulation form(spaceDim, conformingTraces);
    MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, form.bf(), H1Order, delta_k) );
    
    memoryFootprintInBytes = meshTopo->approximateMemoryFootprint();
    memoryFootprintInMegabytes = (double)memoryFootprintInBytes / (1024 * 1024);
    
    double maximumMemoryFootprintInMegabytes;
    Comm->MaxAll(&memoryFootprintInMegabytes, &maximumMemoryFootprintInMegabytes, 1);
    double minimumMemoryFootprintInMegabytes;
    Comm->MinAll(&memoryFootprintInMegabytes, &minimumMemoryFootprintInMegabytes, 1);
    
    if (rank == 0) cout << "After distributing, maximum memory footprint: " << maximumMemoryFootprintInMegabytes << " MB";
    if (rank == 0) cout << " (minimum: " << minimumMemoryFootprintInMegabytes << " MB).\n";
    
    if (rank == 0)
    {
      cout << "Rank 0 Memory Report, after distributing:\n";
      meshTopo->printApproximateMemoryReport();
    }
    
    meshTopo = Teuchos::null;
  }

}