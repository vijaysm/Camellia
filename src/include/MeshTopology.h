//
//  MeshTopology.h
//  Camellia-debug
//
//  Created by Nate Roberts on 12/2/13.
//
//

#ifndef Camellia_debug_MeshTopology_h
#define Camellia_debug_MeshTopology_h

#include "Shards_CellTopology.hpp"
#include "Intrepid_FieldContainer.hpp"

#include "Cell.h"
#include "EntitySet.h"
#include "MeshGeometry.h"
#include "MeshTopologyView.h"
#include "PeriodicBC.h"
#include "RangeList.h"
#include "RefinementPattern.h"
#include "SpatialFilter.h"
#include "TypeDefs.h"

using namespace std;

namespace Camellia
{
  class MeshTransformationFunction;
  class GlobalDofAssignment;
  struct MeshGeometryInfo;

class MeshTopology : public MeshTopologyView
{
  // _Comm and _ownedCellIndices are used when MeshTopology is distributed
  Epetra_CommPtr _Comm;
  std::set<GlobalIndexType> _ownedCellIndices;
  int _pruningOrdinal = -1;
  
  unsigned _spaceDim; // dimension of the mesh
  
  IndexType _nextCellIndex; // until we actually support cell coarsenings, this will be the same as the global cell count
  IndexType _activeCellCount;

  map< vector<double>, IndexType > _vertexMap; // maps into indices in the vertices list -- here just for vertex identification (i.e. so we don't add the same vertex twice)
  vector< vector<double> > _vertices; // vertex locations

  EntityHandle _initialTimeEntityHandle = -1; // for space-time MeshTopologies: track the handle for the entity set corresponding to the space-time sides at the initial time.
  
  map< EntityHandle, EntitySetPtr > _entitySets;
  map< string, vector<pair<EntityHandle, int> > > _tagSetsInteger; // tags with integer value, applied to EntitySets.
  
  vector< PeriodicBCPtr > _periodicBCs;
  map<IndexType, set< pair<int, int> > > _periodicBCIndicesMatchingNode; // pair: first = index in _periodicBCs; second: 0 or 1, indicating first or second part of the identification matches.  IndexType is the vertex index.
  map< pair<IndexType, pair<int,int> >, IndexType > _equivalentNodeViaPeriodicBC;
  map<IndexType, IndexType> _canonicalVertexPeriodic; // key is a vertex *not* in _knownEntities; the value is the matching vertex in _knownEntities

  // the following entity vectors are indexed on dimension of the entities
  vector< vector< vector<IndexType> > > _entities; // vertices, edges, faces, solids, etc., up to dimension (_spaceDim - 1).  Innermost container is sorted by value of IndexType (nodes). The outer two indices are entityDim, entityIndex.
  vector< map< vector<IndexType>, IndexType > > _knownEntities; // map keys are vectors of sorted vertex indices, values are entity indices in _entities[d]
  vector< vector< vector<IndexType> > > _canonicalEntityOrdering;
  vector< vector< vector< pair<IndexType, unsigned> > > > _activeCellsForEntities; // inner vector entries are sorted (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)--I'm vascillating on whether this should contain entries for active ancestral cells.  Today, I think it should not.  I think we should have another set of activeEntities.  Things in that list either themselves have active cells or an ancestor that has an active cell.  So if your parent is inactive and you don't have any active cells of your own, then you know you can deactivate.
  vector< vector< vector<IndexType> > > _sidesForEntities; // vector indices: dimension d, entity index; innermost container stores entity indices of dimension _spaceDim-1 belonging to cells that contain the indicated entity, sorted by index.
  map< IndexType, pair< pair<IndexType, unsigned>, pair<IndexType, unsigned> > > _cellsForSideEntities; // key: sideEntityIndex.  value.first is (cellIndex1, sideOrdinal1), value.second is (cellIndex2, sideOrdinal2).  On initialization, (cellIndex2, sideOrdinal2) == ((IndexType)-1,(IndexType)-1).
  set<IndexType> _boundarySides; // entities of dimension _spaceDim-1 on the mesh boundary
  vector< map< IndexType, vector< pair<IndexType, unsigned> > > > _parentEntities; // map from entity to its possible parents.  Not every entity has a parent.  We support entities having multiple parents.  Such things will be useful in the context of anisotropic refinements.  The pair entries here are (parentEntityIndex, refinementOrdinal), where the refinementOrdinal is the index into the _childEntities[d][parentEntityIndex] vector.

  vector< map< IndexType, pair<IndexType, unsigned> > > _generalizedParentEntities; // map from entity to its nearest generalized parent.  map entries are (parentEntityIndex, parentEntityDimension).  Generalized parents may be higher-dimensional or equal-dimensional to the child entity.
  vector< map< IndexType, vector< pair< RefinementPatternPtr, vector<IndexType> > > > > _childEntities; // map from parent to child entities, together with the RefinementPattern to get from one to the other.
  vector< map<Camellia::CellTopologyKey, RangeList<IndexType>>> _entityCellTopologyKeys;

//  vector< CellPtr > _cells;
  map<GlobalIndexType, CellPtr> _cells; // the cells known on this MPI rank.  Right now, all cells are stored on every rank; soon, this will not be true anymore.

  // these guys presently only support 2D:
  set< IndexType > _cellIDsWithCurves;
  map< pair<IndexType, IndexType>, ParametricCurvePtr > _edgeToCurveMap;
  Teuchos::RCP<MeshTransformationFunction> _transformationFunction; // for dealing with those curves

  //  set<IndexType> activeDescendants(IndexType d, IndexType entityIndex);
  //  set<IndexType> activeDescendantsNotInSet(IndexType d, IndexType entityIndex, const set<IndexType> &excludedSet);
  IndexType addCell(IndexType cellIndex, CellTopoPtrLegacy cellTopo, const vector<IndexType> &cellVertices, IndexType parentCellIndex = -1);
  IndexType addCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<IndexType> &cellVertices, IndexType parentCellIndex = -1);
  void addCellForSide(IndexType cellIndex, unsigned sideOrdinal, IndexType sideEntityIndex);
  void addEdgeCurve(pair<IndexType,IndexType> edge, ParametricCurvePtr curve);
  //  IndexType addEntity(const shards::CellTopology &entityTopo, const vector<IndexType> &entityVertices, unsigned &entityPermutation); // returns the entityIndex
  IndexType addEntity(CellTopoPtr entityTopo, const vector<IndexType> &entityVertices, unsigned &entityPermutation); // returns the entityIndex

  void deactivateCell(CellPtr cell);
  set<IndexType> descendants(unsigned d, IndexType entityIndex) const;

  //  pair< IndexType, set<IndexType> > determineEntityConstraints(unsigned d, IndexType entityIndex);
  void addChildren(IndexType firstChildCellIndex, CellPtr cell, const vector< CellTopoPtr > &childTopos,
                   const vector< vector<IndexType> > &childVertices);

  void determineGeneralizedParentsForRefinement(CellPtr cell, RefinementPatternPtr refPattern);
  
  IndexType getVertexIndexAdding(const vector<double> &vertex, double tol);
  vector<IndexType> getVertexIndices(const Intrepid::FieldContainer<double> &vertices);
  vector<IndexType> getVertexIndices(const vector< vector<double> > &vertices);
  map<unsigned, IndexType> getVertexIndicesMap(const Intrepid::FieldContainer<double> &vertices);
  set<IndexType> getEntitiesForSide(IndexType sideEntityIndex, unsigned d) const;
  void init(unsigned spaceDim);
  void printVertex(IndexType vertexIndex) const;
  void printVertices(set<IndexType> vertexIndices) const;
  void refineCellEntities(CellPtr cell, RefinementPatternPtr refPattern); // ensures that the appropriate child entities exist, and parental relationships are recorded in _parentEntities
  void setEntityGeneralizedParent(unsigned entityDim, IndexType entityIndex, unsigned parentDim, IndexType parentEntityIndex);

  GlobalDofAssignment* _gda; // for cubature degree lookups

  map<string, long long> approximateMemoryCosts() const; // for each private variable

  void addSideForEntity(unsigned entityDim, IndexType entityIndex, IndexType sideEntityIndex); // maintains _sidesForEntities container

  // ! private method for deep-copying Cells during MeshToplogy::deepCopy()
  void deepCopyCells();

  CellPtr addCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector< vector<double> > &cellVertices);
  CellPtr addCell(IndexType cellIndex, CellTopoPtr cellTopo, const Intrepid::FieldContainer<double> &cellVertices);
  CellPtr addCell(IndexType cellIndex, CellTopoPtrLegacy cellTopo, const vector< vector<double> > &cellVertices);
  
public:
  MeshTopology(unsigned spaceDim, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());
  MeshTopology(MeshGeometryPtr meshGeometry, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());
  MeshTopology(Epetra_CommPtr Comm, const MeshGeometryInfo &meshGeometryInfo);
  virtual ~MeshTopology() {}

  CellPtr addCell(CellTopoPtr cellTopo, const vector< vector<double> > &cellVertices);
  CellPtr addCell(CellTopoPtr cellTopo, const Intrepid::FieldContainer<double> &cellVertices);
  CellPtr addCell(CellTopoPtrLegacy cellTopo, const vector< vector<double> > &cellVertices);
  
  // ! adds cell with the specified cellIndex, which must be less than _nextCellIndex
  CellPtr addMigratedCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<vector<double>> &cellVertices);
  
  void addVertex(const std::vector<double>& vertex);

  void applyTag(std::string tagName, int tagID, EntitySetPtr entitySet);
  
  // ! This method only gets within a factor of 2 or so, but can give a rough estimate (in bytes)
  long long approximateMemoryFootprint() const;

  const MeshTopology* baseMeshTopology() const; // returns this
  
  EntitySetPtr createEntitySet();
  EntitySetPtr getEntitySet(EntityHandle entitySetHandle) const;
  vector<EntityHandle> getEntityHandlesForCell(IndexType cellIndex) const;
  
  // ! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs and the
  Teuchos::RCP<MeshTopology> deepCopy() const;

  // ! This method only gets within a factor of 2 or so, but can give rough estimates
  void printApproximateMemoryReport() const;

  bool entityHasParent(unsigned d, IndexType entityIndex) const;
  bool entityHasGeneralizedParent(unsigned d, IndexType entityIndex) const;
  bool entityHasChildren(unsigned d, IndexType entityIndex) const;
  IndexType getActiveCellCount(unsigned d, IndexType entityIndex) const;

  vector< pair<IndexType,unsigned> > getActiveCellIndices(unsigned d, IndexType entityIndex) const; // first entry in pair is the cellIndex, the second is the index of the entity in that cell (the subcord).
  CellPtr getCell(IndexType cellIndex) const;
  //  vector< pair< unsigned, unsigned > > getCellNeighbors(unsigned cellIndex, unsigned sideIndex); // second entry in return is the sideIndex in neighbor (note that in context of h-refinements, one or both of the sides may be broken)
  //  pair< CellPtr, unsigned > getCellAncestralNeighbor(unsigned cellIndex, unsigned sideIndex);
  bool cellHasCurvedEdges(IndexType cellIndex) const;

  bool cellContainsPoint(GlobalIndexType cellID, const std::vector<double> &point, int cubatureDegree) const;
  std::vector<IndexType> cellIDsForPoints(const Intrepid::FieldContainer<double> &physicalPoints) const;

  bool entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent) const;
  bool entityIsGeneralizedAncestor(unsigned ancestorDimension, IndexType ancestor,
                                   unsigned descendentDimension, IndexType descendent) const;

  // ! If the MeshTopology is distributed, returns the Comm object used.  Otherwise, returns Teuchos::null, which is meant to indicate that the MeshTopology is replicated on every MPI rank on which it is used.
  Epetra_CommPtr Comm() const;
  
  virtual bool isDistributed() const;
  
  CellPtr findCellWithVertices(const vector< vector<double> > &cellVertices) const;

  vector<IndexType> getChildEntities(unsigned d, IndexType entityIndex) const;
  set<IndexType> getChildEntitiesSet(unsigned d, IndexType entityIndex) const;
  IndexType getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex) const;
  pair<IndexType, unsigned> getConstrainingEntity(unsigned d, IndexType entityIndex) const;
  IndexType getEntityIndex(unsigned d, const set<IndexType> &nodeSet) const;
  IndexType getEntityCount(unsigned d) const;

  pair<IndexType,unsigned> getEntityGeneralizedParent(unsigned d, IndexType entityIndex) const; // returns (parentEntityIndex, parentDimension)

  IndexType getEntityParent(unsigned d, IndexType entityIndex, unsigned parentOrdinal=0) const;
  unsigned getEntityParentCount(unsigned d, IndexType entityIndex) const;
  IndexType getEntityParentForSide(unsigned d, IndexType entityIndex, IndexType parentSideEntityIndex) const;   // returns the entity index for the parent (which might be the entity itself) of entity (d,entityIndex) that is a subcell of side parentSideEntityIndex
  vector<IndexType> getEntityVertexIndices(unsigned d, IndexType entityIndex) const;
  CellTopoPtr getEntityTopology(unsigned d, IndexType entityIndex) const;
  IndexType getFaceEdgeIndex(unsigned faceIndex, unsigned edgeOrdinalInFace) const;

  unsigned getCellCountForSide(IndexType sideEntityIndex) const; // 1 or 2
  pair<IndexType, unsigned> getFirstCellForSide(IndexType sideEntityIndex) const;
  pair<IndexType, unsigned> getSecondCellForSide(IndexType sideEntityIndex) const;

  vector<IndexType> getCanonicalEntityNodesViaPeriodicBCs(unsigned d, const vector<IndexType> &myEntityNodes) const; // if there are periodic BCs for this entity, this converts the provided nodes to the ones listed in the canonical ordering (allows permutation determination) -- this method is meant to be called internally, and from Cell.

  set< pair<IndexType, unsigned> > getCellsContainingEntity(unsigned d, IndexType entityIndex) const;
  vector<IndexType> getCellsForSide(IndexType sideEntityIndex) const;
  vector< IndexType > getSidesContainingEntity(unsigned d, IndexType entityIndex) const;

//  RefinementBranch getSideConstraintRefinementBranch(IndexType sideEntityIndex); // Returns a RefinementBranch that goes from the constraining side to the side indicated.

  unsigned getDimension() const;
  unsigned getSubEntityCount(unsigned int d, IndexType entityIndex, unsigned subEntityDim) const;
  IndexType getSubEntityIndex(unsigned d, IndexType entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal) const;
  unsigned getSubEntityPermutation(unsigned d, IndexType entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal) const;
  bool getVertexIndex(const vector<double> &vertex, IndexType &vertexIndex, double tol=1e-14) const;
  std::vector<IndexType> getVertexIndicesMatching(const vector<double> &vertexInitialCoordinates, double tol=1e-14) const;
  const std::vector<double>& getVertex(IndexType vertexIndex) const;
  
  bool isBoundarySide(IndexType sideEntityIndex) const;
  bool isValidCellIndex(IndexType cellIndex) const;
  
  Intrepid::FieldContainer<double> physicalCellNodesForCell(unsigned cellIndex, bool includeCellDimension = false) const;
  void refineCell(IndexType cellIndex, RefinementPatternPtr refPattern, IndexType firstChildCellIndex);
  
  // ! Returns the global cell count.
  IndexType cellCount() const;

  // ! Returns the global active cell count.
  IndexType activeCellCount() const;

  //  pair<IndexType,IndexType> leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity(unsigned d, unsigned constrainingEntityIndex);

  void setGlobalDofAssignment(GlobalDofAssignment* gda); // for cubature degree lookups

  // ! MPI-collective method for distributed MeshTopology.  The vector returned is ordered by the MPI rank that owns the cell indices (i.e. cell indices belonging to a single rank will be contiguous in this container).
  vector<IndexType> getActiveCellIndicesGlobal() const;
  
  const set<IndexType> &getLocallyKnownActiveCellIndices() const;
  // for a distributed MeshTopology, returns the set of cell indices owned by this MPI rank.  For a replicated MeshTopology, returns an empty set.
  const set<IndexType> &getMyActiveCellIndices() const;
  set< pair<IndexType, unsigned> > getActiveBoundaryCells() const; // (cellIndex, sideOrdinal)
  vector<double> getCellCentroid(IndexType cellIndex) const;
  
  const set<IndexType> &getRootCellIndicesLocal() const;
  set<IndexType> getRootCellIndicesGlobal() const;
  
  vector<EntitySetPtr> getEntitySetsForTagID(string tagName, int tagID) const;
  
  // ! Returns an EntitySet corresponding to the initial time, for a space-time MeshTopology.  Requires that setEntitySetInitialTime() has been called previously.  (This is usually done in MeshFactory.)
  EntitySetPtr getEntitySetInitialTime() const;
  
  // ! Specifies an EntitySet corresponding to the initial time, for a space-time MeshTopology.  Called by MeshFactory.
  void setEntitySetInitialTime(EntitySetPtr entitySet);

  // ! Returns boundary sides whose vertices all match the specified SpatialFilter
  vector<IndexType> getBoundarySidesThatMatch(SpatialFilterPtr spatialFilter) const;
  
  // ! maxConstraint made public for the sake of MeshTopologyView; not intended for general use
  IndexType maxConstraint(unsigned d, IndexType entityIndex1, IndexType entityIndex2) const;
  
  pair<IndexType,IndexType> owningCellIndexForConstrainingEntity(unsigned d, unsigned constrainingEntityIndex) const;

  // 2D only:
  vector< ParametricCurvePtr > parametricEdgesForCell(IndexType cellID, bool neglectCurves) const;
  void setEdgeToCurveMap(const map< pair<IndexType, IndexType>, ParametricCurvePtr > &edgeToCurveMap, MeshPtr mesh);

  void printConstraintReport(unsigned d) const;
  void printEntityVertices(unsigned d, IndexType entityIndex) const;

  void printAllEntities() const;
  
  // ! Removes all entities that do not belong to the "halo" of the cells indicated, stores ownedCellIndices, and sets the Comm object.
  void pruneToInclude(Epetra_CommPtr Comm, const std::set<GlobalIndexType> &ownedCellIndices,
                      unsigned dimForNeighborRelation);
  // ! Returns a counter, incremented when pruneToInclude was last called.  Allows caching of things that depend on ownedCellIndices (used in MeshTopologyView's determination of *its* ownedCellIndices).
  int pruningOrdinal() const;

  // ! Generates a transformation function if edge-to-curve map has been set.  Particularly useful for cases where a null Mesh was previously provided, resulting in a null transformation function.
  void initializeTransformationFunction(MeshPtr mesh);
  
  // not sure this should ultimately be exposed -- using it now to allow correctly timed call to updateCells()
  // (will be transitioning from having MeshTransformationFunction talk to Mesh to having it talk to MeshTopology)
  Teuchos::RCP<MeshTransformationFunction> transformationFunction() const;

  // ! This method exposed for the sake of tests
  vector< pair<IndexType,unsigned> > getConstrainingSideAncestry(IndexType sideEntityIndex) const;   // pair: first is the sideEntityIndex of the ancestor; second is the refinementIndex of the refinement to get from parent to child (see _parentEntities and _childEntities)

  // ! Fills the provided container with the vertices for the requested cell
  void verticesForCell(Intrepid::FieldContainer<double>& vertices, IndexType cellID) const;
  
  const map<EntityHandle, EntitySetPtr>& getEntitySets() const;
  const map<string, vector<pair<EntityHandle, int>>>& getTagSetsInteger() const; // tags with integer value, applied to EntitySets.
  
  MeshTopologyViewPtr getView(const std::set<IndexType> &activeCells) const;
  
  // distributed read/write methods (for HDF5 support, e.g.)
  // ! returns the size, in bytes, of the serialization of this rank's view of the MeshTopology object.
  virtual int dataSize() const;
  // ! reads a distributed MeshTopology
  static MeshTopologyPtr read(Epetra_CommPtr comm, const char* &dataLocation, int size);
  // ! writes a distributed MeshTopology
  virtual void write(char* &dataLocation, int size) const;
};
}

#endif
