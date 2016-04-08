//
//  CellDataMigration.h
//  Camellia-debug
//
//  Created by Nate Roberts on 7/15/14.
//
//

#ifndef __Camellia_debug__CellDataMigration__
#define __Camellia_debug__CellDataMigration__

#include "Mesh.h"

namespace Camellia
{
  typedef pair<RefinementBranch,vector<GlobalIndexType>> LabeledRefinementBranch; // first cellID indicates the root cell ID; after that, each cellID indicates first child of each refinement
  typedef pair<LabeledRefinementBranch, vector<vector<double>> > RootedLabeledRefinementBranch; // second contains vertex coordinates for root cell
  
class CellDataMigration
{
public:
  // methods for Zoltan's benefit
  static int dataSize(Mesh* mesh, GlobalIndexType cellID);
  static void packData(Mesh* mesh, GlobalIndexType cellID, bool packParentDofs, char *dataBuffer, int size);
  static void unpackData(Mesh* mesh, GlobalIndexType cellID, const char *dataBuffer, int size);
  
  // methods used in the above, made public for testing
  static int geometryDataSize(Mesh* mesh, GlobalIndexType cellID);
  static void packGeometryData(Mesh* mesh, GlobalIndexType cellID, char* &dataLocation, int size);
  static void unpackGeometryData(Mesh* mesh, GlobalIndexType cellID, const char* &dataLocation, int size,
                                 vector<RootedLabeledRefinementBranch> &rootedLabeledBranches);
  
  static int solutionDataSize(Mesh* mesh, GlobalIndexType cellID);
  static void packSolutionData(Mesh* mesh, GlobalIndexType cellID, bool packParentDofs, char* &dataLocation, int size);
  static void unpackSolutionData(Mesh* mesh, GlobalIndexType cellID, const char* &dataLocation, int size);
  
  static void getCellGeometry(MeshTopology* mesh, GlobalIndexType cellID, set<GlobalIndexType> &knownCells,
                              RootedLabeledRefinementBranch &rootedLabeledRefBranch);
  static void getCellHaloGeometry(Mesh *mesh, GlobalIndexType cellID, vector<RootedLabeledRefinementBranch> &cellHaloBranches);
  static void getCellHaloGeometry(MeshTopology* meshTopo, unsigned minDimensionForContinuity,
                                  GlobalIndexType cellID, vector<RootedLabeledRefinementBranch> &cellHaloBranches);
  
  static void addMigratedGeometry(MeshTopology* meshTopo, const vector<RootedLabeledRefinementBranch> &rootedLabeledBranches);
};
}

#endif /* defined(__Camellia_debug__CellDataMigration__) */
