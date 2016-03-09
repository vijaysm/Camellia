//
//  MeshTopologyView.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/23/15.
//
//

#include "MeshTopologyView.h"

#include "CamelliaDebugUtility.h"
#include "MeshTopology.h"

using namespace Camellia;
using namespace std;

template<typename A>
long long approximateSetSizeLLVM(set<A> &someSet)   // in bytes
{
  // 48 bytes for the set itself; nodes are 32 bytes + sizeof(pair<A,B>) each
  // if A and B are containers, this won't count their contents...
  
  set<int> emptySet;
  int SET_OVERHEAD = sizeof(emptySet);
  
  int MAP_NODE_OVERHEAD = 32; // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal
  
  return SET_OVERHEAD + (MAP_NODE_OVERHEAD + sizeof(A)) * someSet.size();
}

// ! Constructor for use by MeshTopology and any other subclasses
MeshTopologyView::MeshTopologyView()
{
  
}

// ! Constructor that defines a view in terms of an existing MeshTopology and a set of cells selected to be active.
MeshTopologyView::MeshTopologyView(MeshTopologyPtr meshTopoPtr, const std::set<IndexType> &activeCellIDs)
{
  // for now (at least), we disallow empty MeshTopologyViews:
  TEUCHOS_TEST_FOR_EXCEPTION(activeCellIDs.size() == 0, std::invalid_argument, "Empty activeCellIDs is not allowed in MeshTopologyView constructor!");
  _meshTopo = meshTopoPtr;
  _activeCells = activeCellIDs;
  buildLookups();
}

// ! This method only gets within a factor of 2 or so, but can give a rough estimate
long long MeshTopologyView::approximateMemoryFootprint()
{
  // size of pointers plus size of sets:
  long long footprint = sizeof(_meshTopo);
  footprint += approximateSetSizeLLVM(_activeCells);
  footprint += approximateSetSizeLLVM(_rootCells);
  footprint += approximateSetSizeLLVM(_allKnownCells);
  footprint += sizeof(_gda);
  return  footprint;
}

IndexType MeshTopologyView::cellCount()
{
  return _allKnownCells.size();
}

std::vector<IndexType> MeshTopologyView::cellIDsForPoints(const Intrepid::FieldContainer<double> &physicalPoints)
{
  std::vector<IndexType> descendentIDs = _meshTopo->cellIDsForPoints(physicalPoints);
  vector<IndexType> myIDs;
  for (IndexType descendentCellID : descendentIDs)
  {
    while ((descendentCellID != -1) && (_activeCells.find(descendentCellID) == _activeCells.end()))
    {
      CellPtr descendentCell = _meshTopo->getCell(descendentCellID);
      CellPtr parentCell = descendentCell->getParent();
      if (parentCell == Teuchos::null)
        descendentCellID = -1;
      else
        descendentCellID = parentCell->cellIndex();
    }
    myIDs.push_back(descendentCellID);
  }
  return myIDs;
}

void MeshTopologyView::buildLookups()
{
  set<IndexType> visitedIndices;
  for (IndexType cellID : _activeCells)
  {
    CellPtr cell = _meshTopo->getCell(cellID);
    while ((cell->getParent() != Teuchos::null) && (visitedIndices.find(cellID) == visitedIndices.end()))
    {
      visitedIndices.insert(cellID);
      cell = cell->getParent();
      cellID = cell->cellIndex();
    }
    if (cell->getParent() == Teuchos::null)
    {
      _rootCells.insert(cellID);
    }
  }
  _allKnownCells.insert(_rootCells.begin(),_rootCells.end());
  _allKnownCells.insert(visitedIndices.begin(),visitedIndices.end());
}

// ! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs).  Not supported for MeshTopologyViews with _meshTopo defined (i.e. those that are themselves defined in terms of another MeshTopology object).
Teuchos::RCP<MeshTopology> MeshTopologyView::deepCopy()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "deepCopy() not supported by MeshTopologyView; this method is defined for potential subclass support.");
}

bool MeshTopologyView::entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent)
{
  return _meshTopo->entityIsAncestor(d, ancestor, descendent);
}

bool MeshTopologyView::entityIsGeneralizedAncestor(unsigned int ancestorDimension, IndexType ancestor, unsigned int descendentDimension, IndexType descendent)
{
  return _meshTopo->entityIsGeneralizedAncestor(ancestorDimension, ancestor, descendentDimension, descendent);
}

const set<IndexType> &MeshTopologyView::getActiveCellIndices()
{
  return _activeCells;
}

IndexType MeshTopologyView::getActiveCellCount(unsigned d, IndexType entityIndex)
{
  // first entry in pair is the cellIndex, the second is the ordinal of the entity in that cell (the subcord).
  set<IndexType> activeCellIndices;
  
  vector<IndexType> sideIndices = this->getSidesContainingEntity(d, entityIndex);
  for (IndexType sideEntityIndex : sideIndices)
  {
    vector<IndexType> cells = this->getActiveCellsForSide(sideEntityIndex);
    for (IndexType cellIndex : cells)
    {
      activeCellIndices.insert(cellIndex);
    }
  }
  return activeCellIndices.size();
}

vector< pair<IndexType,unsigned> > MeshTopologyView::getActiveCellIndices(unsigned d, IndexType entityIndex)
{
  // first entry in pair is the cellIndex, the second is the ordinal of the entity in that cell (the subcord).
  set<pair<IndexType,unsigned>> activeCellIndicesSet;
  
  vector<IndexType> sideIndices = this->getSidesContainingEntity(d, entityIndex);
  for (IndexType sideEntityIndex : sideIndices)
  {
    vector<IndexType> cells = this->getActiveCellsForSide(sideEntityIndex);
    for (IndexType cellIndex : cells)
    {
      // one of our active cells contains the entity
      CellPtr cell = _meshTopo->getCell(cellIndex);
      unsigned subcord = cell->findSubcellOrdinal(d, entityIndex);
      activeCellIndicesSet.insert({cellIndex,subcord});
    }
  }
  vector<pair<IndexType,unsigned>> activeCellIndicesVector(activeCellIndicesSet.begin(),activeCellIndicesSet.end());
  return activeCellIndicesVector;
}

vector<IndexType> MeshTopologyView::getActiveCellsForSide(IndexType sideEntityIndex)
{
  vector<IndexType> cellsForSide = getCellsForSide(sideEntityIndex);
  
  vector<IndexType> activeCells;
  for (IndexType cellIndex : cellsForSide)
  {
    if (_activeCells.find(cellIndex) != _activeCells.end())
    {
      activeCells.push_back(cellIndex);
    }
  }
  return activeCells;
}

CellPtr MeshTopologyView::getCell(IndexType cellIndex)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!isValidCellIndex(cellIndex), std::invalid_argument, "Invalid cellIndex!");
  return _meshTopo->getCell(cellIndex);
}

vector<double> MeshTopologyView::getCellCentroid(IndexType cellIndex)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!isValidCellIndex(cellIndex), std::invalid_argument, "Invalid cellIndex!");
  return _meshTopo->getCellCentroid(cellIndex);
}

// getCellsContainingEntity() copied from MeshTopology; could possibly eliminate it in MeshTopology
// ! pairs are (cellIndex, sideOrdinal) where the sideOrdinal is a side that contains the entity
set< pair<IndexType, unsigned> > MeshTopologyView::getCellsContainingEntity(unsigned d, unsigned entityIndex)   // not *all* cells, but within any refinement branch, the most refined cell that contains the entity will be present in this set.  The unsigned value is the ordinal of a *side* in the cell containing this entity.  There may be multiple sides in a cell that contain the entity; this method will return just one entry per cell.
{
  if (d==getDimension())
  {
    // entityIndex is a cell; the side then is contained within the cell; we'll flag this fact by setting the side ordinal to -1.
    return {{entityIndex,-1}};
  }
  vector<IndexType> sidesForEntity = getSidesContainingEntity(d, entityIndex);
  typedef pair<IndexType,unsigned> CellPair;
  set< CellPair > cells;
  set< IndexType > cellIndices;  // container to keep track of which cells we've already counted -- we only return one (cell, side) pair per cell that contains the entity...
  int sideDim = getDimension() - 1;
  for (IndexType sideEntityIndex : sidesForEntity)
  {
    vector<IndexType> cellsForSide = getCellsForSide(sideEntityIndex);
    TEUCHOS_TEST_FOR_EXCEPTION((cellsForSide.size() == 0) || (cellsForSide.size() > 2), std::invalid_argument, "Unexpected cell count for side.");
    
    for (IndexType cellIndex : cellsForSide)
    {
      if (cellIndices.find(cellIndex) != cellIndices.end()) continue; // already have an entry for this cell
      CellPtr cell = _meshTopo->getCell(cellIndex);
      unsigned sideSubcord = cell->findSubcellOrdinal(sideDim, sideEntityIndex);
      cells.insert({cellIndex,sideSubcord});
      cellIndices.insert(cellIndex);
    }
  }
  return cells;
}

vector<IndexType> MeshTopologyView::getCellsForSide(IndexType sideEntityIndex)
{
  vector<IndexType> cells;
  for (int whichCell=0; whichCell<2; whichCell++)
  {
    IndexType cellIndex;
    if (whichCell == 0)
      cellIndex = _meshTopo->getFirstCellForSide(sideEntityIndex).first;
    else
      cellIndex =  _meshTopo->getSecondCellForSide(sideEntityIndex).first;
    while ((cellIndex != -1) && !isValidCellIndex(cellIndex))
    {
      // we should check whether there is a valid ancestor of the cell that also contains the side
      // (this can happen both in 1D and with anisotropic refinements)
      CellPtr cell = _meshTopo->getCell(cellIndex);
      CellPtr parent = cell->getParent();
      if (parent != Teuchos::null)
      {
        // check whether the parent contains the side:
        int sideDim = _meshTopo->getDimension() - 1;
        unsigned sideOrdinal = parent->findSubcellOrdinal(sideDim, sideEntityIndex);
        if (sideOrdinal != -1) cellIndex = parent->cellIndex();
        else cellIndex = -1;
      }
      else
      {
        cellIndex = -1;
      }
    }
    if (isValidCellIndex(cellIndex)) cells.push_back(cellIndex);
  }
  return cells;
}

pair<IndexType, unsigned> MeshTopologyView::getConstrainingEntity(unsigned d, IndexType entityIndex)
{
  // copying from MeshTopology's implementation:
  unsigned sideDim = getDimension() - 1;
  
  pair<IndexType, unsigned> constrainingEntity; // we store the highest-dimensional constraint.  (This will be the maximal constraint.)
  constrainingEntity.first = entityIndex;
  constrainingEntity.second = d;
  
  IndexType generalizedAncestorEntityIndex = entityIndex;
  for (unsigned generalizedAncestorDim=d; generalizedAncestorDim <= sideDim; )
  {
    IndexType possibleConstrainingEntityIndex = getConstrainingEntityIndexOfLikeDimension(generalizedAncestorDim, generalizedAncestorEntityIndex);
    if (possibleConstrainingEntityIndex != generalizedAncestorEntityIndex)
    {
      constrainingEntity.second = generalizedAncestorDim;
      constrainingEntity.first = possibleConstrainingEntityIndex;
    }
    else
    {
      // if the generalized parent has no constraint of like dimension, then either the generalized parent is the constraint, or there is no constraint of this dimension
      // basic rule: if there exists a side belonging to an active cell that contains the putative constraining entity, then we constrain
      // I am a bit vague on whether this will work correctly in the context of anisotropic refinements.  (It might, but I'm not sure.)  But first we are targeting isotropic.
      vector<IndexType> sidesForEntity;
      if (generalizedAncestorDim==sideDim)
      {
        sidesForEntity.push_back(generalizedAncestorEntityIndex);
      }
      else
      {
        sidesForEntity = getSidesContainingEntity(generalizedAncestorDim, generalizedAncestorEntityIndex);
      }
      for (vector<IndexType>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++)
      {
        IndexType sideEntityIndex = *sideEntityIt;
        if (getActiveCellsForSide(sideEntityIndex).size() > 0)
        {
          constrainingEntity.second = generalizedAncestorDim;
          constrainingEntity.first = possibleConstrainingEntityIndex;
          break;
        }
      }
    }
    while (_meshTopo->entityHasParent(generalizedAncestorDim, generalizedAncestorEntityIndex))   // parent of like dimension
    {
      generalizedAncestorEntityIndex = _meshTopo->getEntityParent(generalizedAncestorDim, generalizedAncestorEntityIndex);
    }
    if (_meshTopo->entityHasGeneralizedParent(generalizedAncestorDim, generalizedAncestorEntityIndex))
    {
      pair< IndexType, unsigned > generalizedParent = _meshTopo->getEntityGeneralizedParent(generalizedAncestorDim, generalizedAncestorEntityIndex);
      generalizedAncestorEntityIndex = generalizedParent.first;
      generalizedAncestorDim = generalizedParent.second;
    }
    else     // at top of refinement tree -- break out of for loop
    {
      break;
    }
  }
  return constrainingEntity;
}

// copied from MeshTopology; once that's a subclass of MeshTopologyView, could possibly eliminate it in MeshTopology
IndexType MeshTopologyView::getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex)
{
  unsigned constrainingEntityIndex = entityIndex;
  
  if (d==0)   // one vertex can't constrain another...
  {
    return entityIndex;
  }
  
  // 3-9-16: I've found an example in which the below fails in a 2-irregular mesh
  // I think the following, simpler thing will work just fine.  (It does pass tests!)
  IndexType ancestorEntityIndex = entityIndex;
  while (_meshTopo->entityHasParent(d, ancestorEntityIndex))
  {
    ancestorEntityIndex = _meshTopo->getEntityParent(d, ancestorEntityIndex);
    if (getActiveCellCount(d, ancestorEntityIndex) > 0)
    {
      constrainingEntityIndex = ancestorEntityIndex;
    }
  }
  
  //  vector<unsigned> sidesForEntity;
  //  unsigned sideDim = _spaceDim - 1;
  //  if (d==sideDim)
  //  {
  //    sidesForEntity.push_back(entityIndex);
  //  }
  //  else
  //  {
  //    sidesForEntity = _sidesForEntities[d][entityIndex];
  //  }
  //  for (vector<unsigned>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++)
  //  {
  //    unsigned sideEntityIndex = *sideEntityIt;
  //    vector< pair<unsigned,unsigned> > sideAncestry = getConstrainingSideAncestry(sideEntityIndex);
  //    unsigned constrainingEntityIndexForSide = entityIndex;
  //    if (sideAncestry.size() > 0)
  //    {
  //      // need to find the subcellEntity for the constraining side that overlaps with the one on our present side
  //      for (vector< pair<unsigned,unsigned> >::iterator entryIt=sideAncestry.begin(); entryIt != sideAncestry.end(); entryIt++)
  //      {
  //        // need to map constrained entity index from the current side to its parent in sideAncestry
  //        unsigned parentSideEntityIndex = entryIt->first;
  //        if (_parentEntities[d].find(constrainingEntityIndexForSide) == _parentEntities[d].end())
  //        {
  //          // no parent for this entity (may be that it was a refinement-interior edge, e.g.)
  //          break;
  //        }
  //        constrainingEntityIndexForSide = getEntityParentForSide(d,constrainingEntityIndexForSide,parentSideEntityIndex);
  //        sideEntityIndex = parentSideEntityIndex;
  //      }
  //    }
  //    constrainingEntityIndex = maxConstraint(d, constrainingEntityIndex, constrainingEntityIndexForSide);
  //  }
  return constrainingEntityIndex;
}

// getConstrainingSideAncestry() copied from MeshTopology; once that's a subclass of MeshTopologyView, could possibly eliminate it in MeshTopology
// pair: first is the sideEntityIndex of the ancestor; second is the refinementIndex of the refinement to get from parent to child (see _parentEntities and _childEntities)
vector< pair<IndexType,unsigned> > MeshTopologyView::getConstrainingSideAncestry(unsigned int sideEntityIndex)
{
  // three possibilities: 1) compatible side, 2) side is parent, 3) side is child
  // 1) and 2) mean unconstrained.  3) means constrained (by parent)
  unsigned sideDim = getDimension() - 1;
  vector< pair<unsigned, unsigned> > ancestry;
  if (_meshTopo->isBoundarySide(sideEntityIndex))
  {
    return ancestry; // sides on boundary are unconstrained...
  }
  
  vector< pair<unsigned,unsigned> > sideCellEntries = getActiveCellIndices(sideDim, sideEntityIndex); //_activeCellsForEntities[sideDim][sideEntityIndex];
  int activeCellCountForSide = sideCellEntries.size();
  if (activeCellCountForSide == 2)
  {
    // compatible side
    return ancestry; // will be empty
  }
  else if ((activeCellCountForSide == 0) || (activeCellCountForSide == 1))
  {
    // then we're either parent or child of an active side
    // if we are a parent, then *this* sideEntityIndex is unconstrained, and we can return an empty ancestry.
    // if we are a child, then we should find and return an ancestral path that ends in an active side
    IndexType ancestorIndex = sideEntityIndex;
    // the possibility of multiple parents is there for the sake of anisotropic refinements.  We don't fully support
    // these yet, but may in the future.
    while (_meshTopo->entityHasParent(sideDim, ancestorIndex))
    {
      int entityParentCount = _meshTopo->getEntityParentCount(sideDim, ancestorIndex);
      IndexType entityParentIndex = -1;
      for (int entityParentOrdinal=0; entityParentOrdinal<entityParentCount; entityParentOrdinal++)
      {
        entityParentIndex = _meshTopo->getEntityParent(sideDim, ancestorIndex, entityParentOrdinal);
        if (getActiveCellIndices(sideDim, entityParentIndex).size() > 0)
        {
          // active cell; we've found our final ancestor
          ancestry.push_back({entityParentIndex, entityParentOrdinal});
          return ancestry;
        }
      }
      // if we get here, then (parentEntityIndex, entityParentCount-1) refers to the last of the possible parents, which by convention must be a regular refinement (more precisely, one whose subentities are at least as fine as all previous possible parents)
      // this is therefore an acceptable entry in our ancestry path.
      ancestry.push_back({entityParentIndex, entityParentCount-1});
      ancestorIndex = entityParentIndex;
    }
    // if no such ancestral path exists, then we are a parent, and are unconstrained (return empty ancestry)
    ancestry.clear();
    return ancestry;
  }
  else
  {
    cout << "MeshTopologyView internal error: # active cells for side is not 0, 1, or 2\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MeshTopologyView internal error: # active cells for side is not 0, 1, or 2\n");
  }
}

unsigned MeshTopologyView::getDimension() const
{
  return _meshTopo->getDimension();
}

std::vector<IndexType> MeshTopologyView::getEntityVertexIndices(unsigned d, IndexType entityIndex)
{
  return _meshTopo->getEntityVertexIndices(d,entityIndex);
}

const set<IndexType> & MeshTopologyView::getRootCellIndices()
{
  if (_rootCells.size() == 0)
  {
    buildLookups(); // but for the present, we do this on construction anyway
  }
  return _rootCells;
}

vector<IndexType> MeshTopologyView::getSidesContainingEntity(unsigned d, IndexType entityIndex)
{
  unsigned sideDim = getDimension() - 1;
  vector<IndexType> meshTopoSides;
  if (d == sideDim) meshTopoSides = {entityIndex};
  else meshTopoSides = _meshTopo->getSidesContainingEntity(d, entityIndex);
  
  vector<IndexType> viewSides; // meshTopoSides, filtered for sides that belong to valid cells
  for (IndexType topoSideIndex : meshTopoSides)
  {
    if (this->getCellsForSide(topoSideIndex).size() > 0)
    {
      viewSides.push_back(topoSideIndex);
    }
  }
  return viewSides;
}

const std::vector<double>& MeshTopologyView::getVertex(IndexType vertexIndex)
{
  return _meshTopo->getVertex(vertexIndex);
}

bool MeshTopologyView::getVertexIndex(const vector<double> &vertex, IndexType &vertexIndex, double tol)
{
  return _meshTopo->getVertexIndex(vertex, vertexIndex, tol);
}

vector<IndexType> MeshTopologyView::getVertexIndicesMatching(const vector<double> &vertexInitialCoordinates, double tol)
{
  return _meshTopo->getVertexIndicesMatching(vertexInitialCoordinates, tol);
}

bool MeshTopologyView::isParent(IndexType cellIndex)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!isValidCellIndex(cellIndex), std::invalid_argument, "cellIndex is invalid!");
  return _activeCells.find(cellIndex) == _activeCells.end();
}

bool MeshTopologyView::isValidCellIndex(IndexType cellIndex)
{
  return _allKnownCells.find(cellIndex) != _allKnownCells.end();
}

Intrepid::FieldContainer<double> MeshTopologyView::physicalCellNodesForCell(unsigned cellIndex, bool includeCellDimension)
{
  return _meshTopo->physicalCellNodesForCell(cellIndex,includeCellDimension);
}

void MeshTopologyView::printActiveCellAncestors()
{
  for (IndexType cellID : getActiveCellIndices())
  {
    printCellAncestors(cellID);
  }
}

void MeshTopologyView::printAllEntitiesInBaseMeshTopology()
{
  if (_meshTopo != Teuchos::null)
  {
    _meshTopo->printAllEntities();
  }
  else
  {
    // this must be a MeshTopology object
    MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(this);
    meshTopo->printAllEntities();
  }
}

void MeshTopologyView::printCellAncestors(IndexType cellID)
{
  vector<IndexType> cellAncestors;
  CellPtr cell = getCell(cellID);
  while (cell->getParent() != Teuchos::null) {
    cell = cell->getParent();
    cellAncestors.push_back(cell->cellIndex());
  }
  ostringstream cellLabel;
  cellLabel << cellID;
  print(cellLabel.str(),cellAncestors);
}

Teuchos::RCP<MeshTransformationFunction> MeshTopologyView::transformationFunction()
{
  return Teuchos::null; // pure MeshTopologyViews are defined to have straight-edge geometry only.
}

// owningCellIndexForConstrainingEntity() copied from MeshTopology; once that's a subclass of MeshTopologyView, could possibly eliminate it in MeshTopology
std::pair<IndexType,IndexType> MeshTopologyView::owningCellIndexForConstrainingEntity(unsigned d, IndexType constrainingEntityIndex)
{
  // sorta like the old leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity, but now prefers larger cells
  // -- the first level of the entity refinement hierarchy that has an active cell containing an entity in that level is the one from
  // which we choose the owning cell (and we do take the least such cellIndex)
  unsigned leastActiveCellIndex = (unsigned)-1; // unsigned cast of -1 makes maximal unsigned #
  set<IndexType> constrainedEntities;
  constrainedEntities.insert(constrainingEntityIndex);
  
  IndexType leastActiveCellConstrainedEntityIndex;
  while (true)
  {
    set<IndexType> nextTierConstrainedEntities;
    
    for (set<IndexType>::iterator constrainedEntityIt = constrainedEntities.begin(); constrainedEntityIt != constrainedEntities.end(); constrainedEntityIt++)
    {
      IndexType constrainedEntityIndex = *constrainedEntityIt;
      
      // get this entity's immediate children, in case we don't find an active cell on this tier
      vector<IndexType> immediateChildren = _meshTopo->getChildEntities(d,constrainedEntityIndex);
      nextTierConstrainedEntities.insert(immediateChildren.begin(), immediateChildren.end());
      
      vector<IndexType> sideEntityIndices = getSidesContainingEntity(d, constrainedEntityIndex);
      if (sideEntityIndices.size() == 0)
      {
        cout << "ERROR: no side containing entityIndex " << constrainingEntityIndex << " of dimension " << d << " found." << endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: no side containing entityIndex found.");
      }

      for (IndexType sideEntityIndex : sideEntityIndices)
      {
        vector<IndexType> activeCellsForSide = this->getActiveCellsForSide(sideEntityIndex);
        
        for (IndexType cellIndex : activeCellsForSide)
        {
          if (cellIndex < leastActiveCellIndex)
          {
            leastActiveCellConstrainedEntityIndex = constrainedEntityIndex;
            leastActiveCellIndex = cellIndex;
          }
        }
      }
    }
    if (leastActiveCellIndex == -1)
    {
      // try the next refinement level down
      if (nextTierConstrainedEntities.size() == 0)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No active cell found containing entity constrained by constraining entity");
      }
      constrainedEntities = nextTierConstrainedEntities;
    }
    else
    {
      return {leastActiveCellIndex, leastActiveCellConstrainedEntityIndex};
    }
  }
  
  return {leastActiveCellIndex, leastActiveCellConstrainedEntityIndex};
}

void MeshTopologyView::setGlobalDofAssignment(GlobalDofAssignment* gda)
{ // for cubature degree lookups
  _gda = gda;
}

void MeshTopologyView::verticesForCell(Intrepid::FieldContainer<double>& vertices, IndexType cellID)
{
  _meshTopo->verticesForCell(vertices, cellID);
}

MeshTopologyViewPtr MeshTopologyView::getView(const set<IndexType> &activeCells)
{
  return Teuchos::rcp( new MeshTopologyView(_meshTopo, activeCells) );
}