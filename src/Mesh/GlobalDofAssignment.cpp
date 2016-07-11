//
//  GlobalDofAssignment.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/20/14.
//
//

#include "GlobalDofAssignment.h"

#include "Teuchos_GlobalMPISession.hpp"


// subclasses:
#include "GDAMinimumRule.h"
#include "GDAMaximumRule2D.h"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CondensedDofInterpreter.h"
#include "ElementType.h"
#include "Solution.h"

using namespace Intrepid;
using namespace Camellia;
using namespace std;

GlobalDofAssignment::GlobalDofAssignment(MeshPtr mesh, VarFactoryPtr varFactory,
    DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
    vector<int> initialH1OrderTrial, int testOrderEnhancement, bool enforceConformityLocally) : DofInterpreter(mesh)
{

  _mesh = mesh;
  _meshTopology = mesh->getTopology();
  _varFactory = varFactory;
  _dofOrderingFactory = dofOrderingFactory;
  _partitionPolicy = partitionPolicy;
  _initialH1OrderTrial = initialH1OrderTrial;
  _testOrderEnhancement = testOrderEnhancement;
  _enforceConformityLocally = enforceConformityLocally;
  
  Epetra_CommPtr Comm = _mesh->getTopology()->Comm();
  
  if (_mesh->getTopology()->Comm() != Teuchos::null)
  {
    set<GlobalIndexType> localActiveCellIDs = _mesh->getTopology()->getLocallyKnownActiveCellIndices();
    
    for (IndexType cellID : localActiveCellIDs)
    {
      assignParities(cellID);
    }
    
    _numPartitions = Comm->NumProc();
    
    int globalActiveCellCount = _mesh->getTopology()->activeCellCount();
    set<GlobalIndexType> myCellIDs = _mesh->getTopology()->getMyActiveCellIndices();
    int numMyCellIDs = myCellIDs.size();

    int myPartitionOffset;
    Comm->ScanSum(&numMyCellIDs, &myPartitionOffset, 1);
    myPartitionOffset -= numMyCellIDs;
    
    vector<int> gatheredPartitionOffsets(_numPartitions,0);
    Comm->GatherAll(&myPartitionOffset, &gatheredPartitionOffsets[0], 1);
    
    vector<GlobalIndexTypeToCast> partitionOrderedCells(globalActiveCellCount,0);
    vector<GlobalIndexTypeToCast> gatheredPartitionOrderedCells(globalActiveCellCount,0);
    
    int i=0;
    for (GlobalIndexType cellID : myCellIDs)
    {
      partitionOrderedCells[i + myPartitionOffset] = cellID;
      i++;
    }
    // since this does not contain the same number of entries on each rank, can't use GatherAll
    Comm->SumAll(&partitionOrderedCells[0], &gatheredPartitionOrderedCells[0], globalActiveCellCount);
    
    _partitions = vector<set<GlobalIndexType> >(_numPartitions);
    for (int partitionOrdinal=0; partitionOrdinal<_numPartitions; partitionOrdinal++)
    {
      set<GlobalIndexType>* partitionCellIDs = &_partitions[partitionOrdinal];
      int startOffset = gatheredPartitionOffsets[partitionOrdinal];
      int endOffset = (partitionOrdinal < _numPartitions-1) ? gatheredPartitionOffsets[partitionOrdinal+1] : globalActiveCellCount;
      for (int i=startOffset; i<endOffset; i++)
      {
        partitionCellIDs->insert(gatheredPartitionOrderedCells[i]);
      }
    }
  }
  else
  {
    // MeshTopology not already distributed
    set<GlobalIndexType> activeCellIDs = _mesh->getActiveCellIDsGlobal();
    
    for (IndexType cellID : activeCellIDs)
    {
      assignParities(cellID);
    }
    
    _numPartitions = _partitionPolicy->Comm()->NumProc();
    _partitions = vector<set<GlobalIndexType> >(_numPartitions);
    
    // before repartitioning (which should happen immediately), put all active cells on rank 0
    _partitions[0] = activeCellIDs;
  }

  constructActiveCellMap();
  determineMinimumSubcellDimensionForContinuityEnforcement();
}

GlobalDofAssignment::GlobalDofAssignment( GlobalDofAssignment &otherGDA ) : DofInterpreter(Teuchos::null)    // subclass deepCopy() is responsible for filling this in post-construction
{
  _activeCellOffset = otherGDA._activeCellOffset;
  _cellSideParitiesForCellID = otherGDA._cellSideParitiesForCellID;

  _elementTypeFactory = otherGDA._elementTypeFactory;
  _enforceConformityLocally = otherGDA._enforceConformityLocally;

  _mesh = Teuchos::null;         // subclass deepCopy() is responsible for filling this in post-construction
  _meshTopology = Teuchos::null; // subclass deepCopy() is responsible for filling this in post-construction
  _varFactory = otherGDA._varFactory;
  _dofOrderingFactory = otherGDA._dofOrderingFactory;
  _partitionPolicy = otherGDA._partitionPolicy;;
  _initialH1OrderTrial = otherGDA._initialH1OrderTrial;
  _testOrderEnhancement = otherGDA._testOrderEnhancement;

  _cellPRefinements = otherGDA._cellPRefinements;
  _elementTypesForCellTopology = otherGDA._elementTypesForCellTopology;

  _partitions = otherGDA._partitions;
  _partitionForCellID = otherGDA._partitionForCellID;

  _activeCellMap = Teuchos::rcp( new Epetra_Map(*otherGDA._activeCellMap) );
//  _activeCellMap2 = Teuchos::rcp( new Tpetra::Map<IndexType,GlobalIndexType>(*otherGDA._activeCellMap2) );

  _numPartitions = otherGDA._numPartitions;

  // we leave _registeredSolutions empty
  ///_registeredSolutions;
}

GlobalIndexType GlobalDofAssignment::activeCellOffset()
{
  return _activeCellOffset;
}

void GlobalDofAssignment::assignParities( GlobalIndexType cellID )
{
  CellPtr cell = _mesh->getTopology()->getCell(cellID);

  unsigned sideCount = cell->getSideCount();

  vector<int> cellParities(sideCount);
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, _meshTopology);
    GlobalIndexType neighborCellID = neighborInfo.first;
    if (neighborCellID == -1)   // boundary --> parity is 1
    {
      cellParities[sideOrdinal] = 1;
    }
    else
    {
      CellPtr neighbor = _meshTopology->getCell(neighborCellID);
      pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(neighborInfo.second, _meshTopology);
      bool cellParitySet = false;
      
      // if the neighbor sees us as its neighbor, then we're certainly peers
      if (neighborNeighborInfo.first == cellID)
      {
        // then the lower cellID gets the positive parity
        cellParities[sideOrdinal] = (cellID < neighborCellID) ? 1 : -1;
        cellParitySet = true;
      }
      else
      {
        // If not, then it's still possible that we have a descendant that is peers with neighbor,
        // in the case of anisotropic refinements.
        
        // Find the most refined descendant that shares the side:
        unsigned sideDim = _meshTopology->getDimension() - 1;
        IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
        vector<IndexType> cellIndicesForSide = _meshTopology->getCellsForSide(sideEntityIndex); // max 2 entries
        for (IndexType cellIndexForSide : cellIndicesForSide)
        {
          if (cellIndexForSide == neighborCellID) continue; // skip the neighbor
          if (cellIndexForSide == neighborNeighborInfo.first)
          {
            // a descendant that is a peer: we "inherit" parity of the descendant
            // then the lower cellID gets the positive parity
            cellParities[sideOrdinal] = (cellIndexForSide < neighborCellID) ? 1 : -1;
            cellParitySet = true;
          }
        }
      }
      if (!cellParitySet)
      {
        CellPtr parent = cell->getParent();
        if (parent.get() == NULL)
        {
          cout << "ERROR: in assignParities(), encountered cell with non-peer neighbor but without parent.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "in assignParities(), encountered cell with non-peer neighbor but without parent");
        }
        // inherit parent's parity along the shared side:
        unsigned childOrdinal = parent->childOrdinal(cellID);
        unsigned parentSideOrdinal = parent->refinementPattern()->parentSideLookupForChild(childOrdinal)[sideOrdinal];
        if (_cellSideParitiesForCellID.find(parent->cellIndex()) == _cellSideParitiesForCellID.end())
        {
          assignParities(parent->cellIndex());
        }
        cellParities[sideOrdinal] = _cellSideParitiesForCellID[parent->cellIndex()][parentSideOrdinal];
      }
    }
  }
  // track which sides flipped parities, so that solution's fluxes can be corrected accordingly
  vector<int> sidesWithChangedParities;
  if (_cellSideParitiesForCellID.find(cellID) != _cellSideParitiesForCellID.end())
  {
    vector<int> oldSideParities = _cellSideParitiesForCellID[cellID];
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      if (oldSideParities[sideOrdinal] != cellParities[sideOrdinal])
      {
        sidesWithChangedParities.push_back(sideOrdinal);
      }
    }
  }
  _cellSideParitiesForCellID[cellID] = cellParities;

  for (TSolutionPtr<double> soln : _registeredSolutions)
  {
    if (soln->cellHasCoefficientsAssigned(cellID))
    {
      soln->reverseParitiesForLocalCoefficients(cellID, sidesWithChangedParities);
//      cout << "Reversing parities for cellID " << cellID << " on " << sidesWithChangedParities.size() << " sides." << endl;
    }
  }
  
  // if this cell is a parent, then we should treat its children as well (children without peer neighbors will inherit any parity flips)
  if (cell->isParent(_meshTopology))
  {
    vector<GlobalIndexType> childIndices = cell->getChildIndices(_meshTopology);
    for (GlobalIndexType childCellIndex : childIndices)
    {
      if (_meshTopology->isValidCellIndex(childCellIndex))
      {
        // if childCellIndex isn't a valid cell index, the cell for cellID must be a ghost cell, and we don't actually need parities
        // to be correct on that side...
        assignParities(childCellIndex);
      }
    }
  }
}

//GlobalIndexType GlobalDofAssignment::cellID(Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber)
//{
//  if (partitionNumber == -1)
//  {
//    // determine the partition number for the cellIndex
//    int partitionCellOffset = 0;
//    for (PartitionIndexType i=0; i<partitionNumber; i++)
//    {
//      int numCellIDsForPartition = _cellIDsForElementType[i][elemTypePtr.get()].size();
//      if (partitionCellOffset + numCellIDsForPartition > cellIndex)
//      {
//        partitionNumber = i;
//        cellIndex -= partitionCellOffset; // rewrite as a local cellIndex
//        break;
//      }
//      partitionCellOffset += numCellIDsForPartition;
//    }
//    if (partitionNumber == -1)
//    {
//      cout << "cellIndex is out of bounds.\n";
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIndex is out of bounds.");
//    }
//  }
//  if ( ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) != _cellIDsForElementType[partitionNumber].end() )
//       &&
//       (_cellIDsForElementType[partitionNumber][elemTypePtr.get()].size() > cellIndex ) )
//  {
//    return _cellIDsForElementType[partitionNumber][elemTypePtr.get()][cellIndex];
//  }
//  else return -1;
//}

vector<GlobalIndexType> GlobalDofAssignment::cellIDsOfElementType(PartitionIndexType partitionNumber, ElementTypePtr elemType)
{
  if (partitionNumber == -1)
  {
    cout << "cellIDsOfElementType called with partitionNumber==-1.  Returning empty vector.\n";
    return vector<GlobalIndexType>();
  }
  // first, let's find the key that corresponds to elemType
  pair<CellTopologyKey,int> key;
  for (auto entry : _elementTypesForCellTopology)
  {
    if (entry.second->equals(*elemType))
    {
      key = entry.first;
      break;
    }
  }
  vector<GlobalIndexType> cellsForElementType;
  const set<GlobalIndexType>* cellsInPartition = &_partitions[partitionNumber];
  for (GlobalIndexType cellID : *cellsInPartition)
  {
    if (getElementTypeLookupKey(cellID) == key)
    {
      cellsForElementType.push_back(cellID);
    }
  }
  return cellsForElementType;
}

const set< GlobalIndexType > & GlobalDofAssignment::cellsInPartition(PartitionIndexType partitionNumber) const
{
  int rank = _partitionPolicy->Comm()->MyPID();
  if (partitionNumber == -1)
  {
    partitionNumber = rank;
  }
  return _partitions[partitionNumber];
}

FieldContainer<double> GlobalDofAssignment::cellSideParitiesForCell( GlobalIndexType cellID )
{
  if (_cellSideParitiesForCellID.find(cellID) == _cellSideParitiesForCellID.end())
  {
    assignParities(cellID);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(_cellSideParitiesForCellID.find(cellID) == _cellSideParitiesForCellID.end(),
                             std::invalid_argument, "_cellSideParities is not set for the provided cell!");
  vector<int> parities = _cellSideParitiesForCellID[cellID];
  int numSides = parities.size();
  FieldContainer<double> cellSideParities(1,numSides);
  for (int sideIndex=0; sideIndex<numSides; sideIndex++)
  {
    cellSideParities(0,sideIndex) = parities[sideIndex];
  }
  return cellSideParities;
}

void GlobalDofAssignment::constructActiveCellMap()
{
  const set<GlobalIndexType>* myCellIDs = &cellsInPartition(-1);
  vector<GlobalIndexTypeToCast> myCellIDsVector(myCellIDs->begin(), myCellIDs->end());

  int indexBase = 0;
  if (myCellIDsVector.size()==0)
    _activeCellMap = Teuchos::rcp( new Epetra_Map(-1, myCellIDsVector.size(), NULL, indexBase, *_partitionPolicy->Comm()) );
  else
    _activeCellMap = Teuchos::rcp( new Epetra_Map(-1, myCellIDsVector.size(), &myCellIDsVector[0], indexBase, *_partitionPolicy->Comm()) );
}

void GlobalDofAssignment::constructActiveCellMap2()
{
  const set<GlobalIndexType>* cellIDs = &cellsInPartition(-1);
  const vector<GlobalIndexType> myCellIDsVector(cellIDs->begin(), cellIDs->end());
  Teuchos::ArrayView< const GlobalIndexType > myCellIDsAV(myCellIDsVector);

  int indexBase = 0;
  if (myCellIDsVector.size()==0)
    _activeCellMap2 = Teuchos::rcp( new Tpetra::Map<IndexType,GlobalIndexType>(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                    myCellIDsVector.size(), indexBase, _partitionPolicy->TeuchosComm()) );
  else
    _activeCellMap2 = Teuchos::rcp( new Tpetra::Map<IndexType,GlobalIndexType>(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                    myCellIDsAV, indexBase, _partitionPolicy->TeuchosComm()) );
}

void GlobalDofAssignment::determineMinimumSubcellDimensionForContinuityEnforcement()
{
  int myMinimumSubcellDimension = _meshTopology->getDimension();
  set<pair<CellTopologyKey,int>> myCellKeys;
  const set<GlobalIndexType>* myCellIDs = &cellsInPartition(-1);
  for (GlobalIndexType myCellID : *myCellIDs)
  {
    myCellKeys.insert(this->getElementTypeLookupKey(myCellID));
  }
  for (auto myCellKey : myCellKeys)
  {
    ElementTypePtr elemType = this->getElementTypeForKey(myCellKey);
    int elemMin = elemType->trialOrderPtr->minimumSubcellDimensionForContinuity();
    myMinimumSubcellDimension = min(myMinimumSubcellDimension,elemMin);
    if (myMinimumSubcellDimension == 0) break; // can't go lower than 0
  }
  int globalMinimumSubcellDimension;
  _partitionPolicy->Comm()->MinAll(&myMinimumSubcellDimension, &globalMinimumSubcellDimension, 1);
  _minContinuityDimension = globalMinimumSubcellDimension;
}

ElementTypePtr GlobalDofAssignment::elementType(GlobalIndexType cellID)
{
  pair<CellTopologyKey, int> key = getElementTypeLookupKey(cellID);
  return getElementTypeForKey(key);
}

const map<GlobalIndexType,int>& GlobalDofAssignment::getCellPRefinements() const
{
  return _cellPRefinements;
}

void GlobalDofAssignment::setCellPRefinements(const map<GlobalIndexType,int>& pRefinements)
{
  _cellPRefinements = pRefinements;
}

void GlobalDofAssignment::repartitionAndMigrate()
{
  _partitionPolicy->partitionMesh(_mesh.get(),_numPartitions);
  // if our MeshTopology is the base, then we can prune
  
  if (_allowMeshTopologyPruning)
  {
    // then clear cell parities, as these may now be outdated
    // (there may be a more granular approach, but parity determination is relatively cheap)
    _cellSideParitiesForCellID.clear();
  }
  
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(_meshTopology.get());
  if (meshTopo != NULL)
  {
    if (_allowMeshTopologyPruning)
    {
      int dimForNeighborRelation = minimumSubcellDimensionForContinuityEnforcement(); // collective operation
      const set<GlobalIndexType>* myCells = &cellsInPartition(-1);
      meshTopo->pruneToInclude(_partitionPolicy->Comm(), *myCells, dimForNeighborRelation);
    }
  }
  for (vector< TSolutionPtr<double> >::iterator solutionIt = _registeredSolutions.begin();
       solutionIt != _registeredSolutions.end(); solutionIt++)
  {
    // if solution has a condensed dof interpreter, we should reinitialize the mapping from interpreted to global dofs
    Teuchos::RCP<DofInterpreter> dofInterpreter = (*solutionIt)->getDofInterpreter();
    CondensedDofInterpreter<double>* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter<double>*>(dofInterpreter.get());
    if (condensedDofInterpreter != NULL)
    {
      condensedDofInterpreter->reinitialize();
    }

    (*solutionIt)->initializeLHSVector(); // rebuild LHS vector; global dofs will have changed. (important for addSolution)
  }
}

void GlobalDofAssignment::didHRefine(const set<GlobalIndexType> &parentCellIDs)   // subclasses should call super
{
  // until we repartition, assign the new children to the parent's partition
  for (set<GlobalIndexType>::const_iterator cellIDIt=parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++)
  {
    GlobalIndexType parentID = *cellIDIt;
    PartitionIndexType partitionForParent = partitionForCellID(parentID);
    if (partitionForParent != -1) // this check allows us to be robust against getting the notification twice.
    {
      // If we don't know the parent cell, then for the moment we won't see even the child indices
      // I'm hoping that we will soon be able to relax the requirement that every partition knows about
      // every other partition's cell assignments; _partitionForCellID is one of the few containers
      // that grows with the number of global cells.
      
      if (_meshTopology->isValidCellIndex(parentID))
      {
        CellPtr parent = _meshTopology->getCell(parentID);
        vector<GlobalIndexType> childIDs = parent->getChildIndices(_meshTopology);
        _partitions[partitionForParent].insert(childIDs.begin(),childIDs.end());
        for (GlobalIndexType childID : childIDs)
        {
          _partitionForCellID[childID] = partitionForParent;
        }
      }
      _partitions[partitionForParent].erase(parentID);
      _partitionForCellID.erase(parentID);
    }
  }
  constructActiveCellMap();
}

void GlobalDofAssignment::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP)   // subclasses should call super
{
  for (GlobalIndexType cellID : cellIDs)
  {
    if (_cellPRefinements.find(cellID) == _cellPRefinements.end())
    {
      _cellPRefinements[cellID] = deltaP;
    }
    else
    {
      _cellPRefinements[cellID] += deltaP;
    }
  }
}

void GlobalDofAssignment::didHUnrefine(const set<GlobalIndexType> &parentCellIDs)   // subclasses should call super
{
  cout << "WARNING: GlobalDofAssignment::didHUnrefine unimplemented.  At minimum, should update partition to drop children, and add parent.\n";
  // TODO: address this -- of course, Mesh doesn't yet support h-unrefinements, so might want to do that first.
}

vector< ElementTypePtr > GlobalDofAssignment::elementTypes(PartitionIndexType partitionNumber)
{
  // Ultimately, it probably will be an error to call this with argument that is not equal to the current rank.
  if (partitionNumber != -1)
  {
    const set<GlobalIndexType>* cellIDsForPartition = &this->cellsInPartition(partitionNumber);
    
    set<pair<CellTopologyKey,int>> elementTypeKeys;
    for (GlobalIndexType cellID : *cellIDsForPartition)
    {
      elementTypeKeys.insert(getElementTypeLookupKey(cellID));
    }
    
    vector< ElementTypePtr > elemTypes;
    for (auto key : elementTypeKeys)
    {
      elemTypes.push_back(getElementTypeForKey(key));
    }
    return elemTypes;
  }
  else
  {
    int numRanks = _mesh->Comm()->NumProc();
    
    set<pair<CellTopologyKey,int>> elementTypeKeys;
    
    for (int rank=0; rank<numRanks; rank++)
    {
      const set<GlobalIndexType>* cellIDsForPartition = &this->cellsInPartition(rank);
      for (GlobalIndexType cellID : *cellIDsForPartition)
      {
        elementTypeKeys.insert(getElementTypeLookupKey(cellID));
      }
    }
    
    vector< ElementTypePtr > elemTypes;
    for (auto key : elementTypeKeys)
    {
      elemTypes.push_back(getElementTypeForKey(key));
    }

    return elemTypes;
  }
}

Teuchos::RCP<Epetra_Map> GlobalDofAssignment::getActiveCellMap()
{
  return _activeCellMap;
}

MapPtr GlobalDofAssignment::getActiveCellMap2()
{
  return _activeCellMap2;
}

int GlobalDofAssignment::getCubatureDegree(GlobalIndexType cellID)
{
  ElementTypePtr elemType = this->elementType(cellID);
  return elemType->trialOrderPtr->maxBasisDegree() + elemType->testOrderPtr->maxBasisDegree();
}

DofOrderingFactoryPtr GlobalDofAssignment::getDofOrderingFactory()
{
  return _dofOrderingFactory;
}

ElementTypeFactory & GlobalDofAssignment::getElementTypeFactory()
{
  return _elementTypeFactory;
}

ElementTypePtr GlobalDofAssignment::getElementTypeForKey(pair<CellTopologyKey,int> key)
{
  if (_elementTypesForCellTopology.find(key) == _elementTypesForCellTopology.end())
  {
    int deltaP = key.second;
    vector<int> H1Order = getH1OrderForPRefinement(deltaP);
    vector<int> testDegree(H1Order.size());
    for (int pComponent=0; pComponent<H1Order.size(); pComponent++)
    {
      testDegree[pComponent] = H1Order[pComponent] + _testOrderEnhancement;
    }
    
    CellTopoPtr cellTopo = CellTopology::cellTopology(key.first);
    DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(H1Order, cellTopo, _enforceConformityLocally);
    DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testDegree, cellTopo);
    ElementTypePtr elemType = _elementTypeFactory.getElementType(trialOrdering,testOrdering,cellTopo);
    _elementTypesForCellTopology[key] = elemType;
  }
  return _elementTypesForCellTopology[key];
}

pair<CellTopologyKey,int> GlobalDofAssignment::getElementTypeLookupKey(GlobalIndexType cellID)
{
  int deltaP = getPRefinementDegree(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(!_meshTopology->isValidCellIndex(cellID), std::invalid_argument, "cellID not found in MeshTopology!");
  CellTopologyKey cellTopoKey = _meshTopology->getCell(cellID)->topology()->getKey();
  
  pair<CellTopologyKey, int> key = {cellTopoKey,deltaP};
  return key;
}

MeshPartitionPolicyPtr GlobalDofAssignment::getPartitionPolicy()
{
  return _partitionPolicy;
}

int GlobalDofAssignment::getPRefinementDegree(GlobalIndexType cellID)
{
  if (_cellPRefinements.find(cellID) == _cellPRefinements.end())
  {
    return 0;
  }
  else
  {
    return _cellPRefinements[cellID];
  }
}

GlobalIndexType GlobalDofAssignment::globalCellIndex(GlobalIndexType cellID)
{
  int partitionNumber     = partitionForCellID(cellID);
  if (partitionNumber == -1)
  {
    // no partition number found -- cell is presumably inactive.  Return -1
    return -1;
  }
  GlobalIndexType cellIndex = partitionLocalCellIndex(cellID, partitionNumber);
  ElementTypePtr elemType = elementType(cellID);
  
  for (PartitionIndexType i=0; i<partitionNumber; i++)
  {
    cellIndex += cellIDsOfElementType(i, elemType).size();
  }
  return cellIndex;
}

vector<int> GlobalDofAssignment::getH1Order(GlobalIndexType cellID)
{
  int deltaP = getPRefinementDegree(cellID);
  return getH1OrderForPRefinement(deltaP);
}

vector<int> GlobalDofAssignment::getH1OrderForPRefinement(int deltaP)
{
  if (deltaP == 0) return _initialH1OrderTrial;
  
  vector<int> H1Order = _initialH1OrderTrial;
  for (int d=0; d<H1Order.size(); d++)
  {
    H1Order[d] += deltaP;
  }
  
  return H1Order;
}

vector<int> GlobalDofAssignment::getInitialH1Order()
{
  return _initialH1OrderTrial;
}

MeshPtr GlobalDofAssignment::getMesh()
{
  return _mesh;
}

MeshTopologyViewPtr GlobalDofAssignment::getMeshTopology()
{
  return _meshTopology;
}

bool GlobalDofAssignment::getPartitions(FieldContainer<GlobalIndexType> &partitions)
{
  if (_partitions.size() == 0) return false; // false: no partitions set
  int numPartitions = _partitions.size();
  int maxSize = 0;
  for (int i=0; i<numPartitions; i++)
  {
    maxSize = std::max<int>((int)_partitions[i].size(),maxSize);
  }
  partitions.resize(numPartitions,maxSize);
  partitions.initialize(-1);
  for (int i=0; i<numPartitions; i++)
  {
    int j=0;
    for (set<GlobalIndexType>::iterator cellIDIt = _partitions[i].begin();
         cellIDIt != _partitions[i].end(); cellIDIt++)
    {
      partitions(i,j) = *cellIDIt;
      j++;
    }
  }
  return true; // true: partitions container filled
}

PartitionIndexType GlobalDofAssignment::getPartitionCount()
{
  return _numPartitions;
}

int GlobalDofAssignment::getTestOrderEnrichment()
{
  return _testOrderEnhancement;
}

void GlobalDofAssignment::interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<double> &localCoefficients, Epetra_MultiVector &globalCoefficients)
{
  DofOrderingPtr trialOrder = elementType(cellID)->trialOrderPtr;
  FieldContainer<double> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (set<int>::iterator trialIDIt = trialOrder->getVarIDs().begin(); trialIDIt != trialOrder->getVarIDs().end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      FieldContainer<double> fittedGlobalCoefficients;
      FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients.ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndex, 0, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
//        cout << "global coefficient " << globalDofIndex << " = " << fittedGlobalCoefficients[i] << endl;
      }
    }
  }
}

template <typename Scalar>
void GlobalDofAssignment::interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<Scalar> &localCoefficients, TVectorPtr<Scalar> globalCoefficients)
{
  DofOrderingPtr trialOrder = elementType(cellID)->trialOrderPtr;
  FieldContainer<Scalar> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (set<int>::iterator trialIDIt = trialOrder->getVarIDs().begin(); trialIDIt != trialOrder->getVarIDs().end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      FieldContainer<Scalar> fittedGlobalCoefficients;
      FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients->replaceGlobalValue(globalDofIndex, 0, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
      }
    }
  }
}
template void GlobalDofAssignment::interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<double> &localCoefficients, TVectorPtr<double> globalCoefficients);

// ! Returns the smallest dimension along which continuity will be enforced.  GlobalDofAssignment's implementation
// ! assumes that the function spaces for the bases defined on cells determine this (e.g. H^1-conforming basis --> 0).
int GlobalDofAssignment::minimumSubcellDimensionForContinuityEnforcement() const
{
  return _minContinuityDimension;
}

void GlobalDofAssignment::projectParentCoefficientsOntoUnsetChildren()
{
  set<GlobalIndexType> rankLocalCellIDs = cellsInPartition(-1);

  for (vector< TSolutionPtr<double> >::iterator solutionIt = _registeredSolutions.begin();
       solutionIt != _registeredSolutions.end(); solutionIt++)
  {
    TSolutionPtr<double> soln = *solutionIt;
    for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCellIDs.begin(); cellIDIt != rankLocalCellIDs.end(); cellIDIt++)
    {
      GlobalIndexType cellID = *cellIDIt;
      if (soln->cellHasCoefficientsAssigned(cellID)) continue;

      CellPtr cell = _meshTopology->getCell(cellID);
      CellPtr parent = cell->getParent();
      if (parent.get()==NULL) continue;
      GlobalIndexType parentCellID = parent->cellIndex();
      if (! soln->cellHasCoefficientsAssigned(parentCellID)) continue;

      int childOrdinal = -1;
      vector<IndexType> childIndices = parent->getChildIndices(_meshTopology);
      for (int i=0; i<childIndices.size(); i++)
      {
        if (childIndices[i]==cellID) childOrdinal = i;
        else childIndices[i] = -1; // indication that Solution should not compute the projection for this child
      }
      if (childOrdinal == -1)
      {
        cout << "ERROR: child not found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "child not found!");
      }
//      cout << "determining cellID " << parent->cellIndex() << "'s child " << childOrdinal << "'s coefficients.\n";
      soln->projectOldCellOntoNewCells(parent->cellIndex(), elementType(parentCellID), childIndices);
    }
  }
}

void GlobalDofAssignment::setPartitions(FieldContainer<GlobalIndexType> &partitionedMesh, bool rebuildDofLookups)
{
//  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndicesGlobal();

  int partitionNumber     = _partitionPolicy->Comm()->MyPID();
  int partitionCount      = _partitionPolicy->Comm()->NumProc();

  TEUCHOS_TEST_FOR_EXCEPTION(partitionedMesh.dimension(0) > partitionCount, std::invalid_argument,
                             "Number of partitions exceeds the maximum MPI rank; this is unsupported");

  //  cout << "determineActiveElements(): there are "  << activeCellIDs.size() << " active elements.\n";
  _partitions.clear();
  _partitionForCellID.clear();

  _activeCellOffset = 0;
  for (PartitionIndexType i=0; i<partitionedMesh.dimension(0); i++)
  {
    set< GlobalIndexType > partition;
    for (int j=0; j<partitionedMesh.dimension(1); j++)
    {
      //      cout << "partitionedMesh(i,j) = " << partitionedMesh(i,j) << endl;
      if (partitionedMesh(i,j) == -1) break; // no more elements in this partition
      GlobalIndexType cellID = partitionedMesh(i,j);
      partition.insert( cellID );
      _partitionForCellID[cellID] = i;
    }
    _partitions.push_back( partition );
    //    if (partitionNumber==0) cout << "partition " << i << ": ";
    //    if (partitionNumber==0) print("",partition);
    if (partitionNumber > i)
    {
      _activeCellOffset += partition.size();
    }
  }
  
  constructActiveCellMap();
  if (rebuildDofLookups)
  {
    determineMinimumSubcellDimensionForContinuityEnforcement();
    projectParentCoefficientsOntoUnsetChildren();
    rebuildLookups();
  }
}

void GlobalDofAssignment::setPartitions(std::vector<std::set<GlobalIndexType> > &partitions, bool rebuildDofLookups)
{
  int thisPartitionNumber = _partitionPolicy->Comm()->MyPID();

  // not sure numProcs == partitions.size() is a great requirement to impose, but it is an assumption we make in some places,
  // so we require it here.
  int numProcs = _partitionPolicy->Comm()->NumProc();
  TEUCHOS_TEST_FOR_EXCEPTION(numProcs != partitions.size(), std::invalid_argument, "partitions.size() must be equal to numProcs!");

  _partitions = partitions;
  _partitionForCellID.clear();

  _activeCellOffset = 0;
  for (PartitionIndexType i=0; i< _partitions.size(); i++)
  {
    set< GlobalIndexType > partition;
    for (set< GlobalIndexType >::iterator cellIDIt = partitions[i].begin(); cellIDIt != partitions[i].end(); cellIDIt++)
    {
      GlobalIndexType cellID = *cellIDIt;
      _partitionForCellID[cellID] = i;
    }
    if (thisPartitionNumber > i)
    {
      _activeCellOffset += partition.size();
    }
  }
  constructActiveCellMap();
  if (rebuildDofLookups)
  {
    determineMinimumSubcellDimensionForContinuityEnforcement();
    projectParentCoefficientsOntoUnsetChildren();
    rebuildLookups();
  }
}

void GlobalDofAssignment::setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy, bool shouldRepartitionAndMigrate )
{
  _partitionPolicy = partitionPolicy;
  if (shouldRepartitionAndMigrate)
  {
    repartitionAndMigrate();
  }
}

PartitionIndexType GlobalDofAssignment::partitionForCellID( GlobalIndexType cellID )
{
  if (_partitionForCellID.find(cellID) != _partitionForCellID.end())
  {
    return _partitionForCellID[ cellID ];
  }
  else
  {
    return -1;
  }
}

IndexType GlobalDofAssignment::partitionLocalCellIndex(GlobalIndexType cellID, int partitionNumber)
{
  if (partitionNumber == -1)
  {
    partitionNumber     = _partitionPolicy->Comm()->MyPID();
  }

  ElementTypePtr elemType = elementType(cellID);
  vector<GlobalIndexType> cellIDsOfType = cellIDsOfElementType(partitionNumber, elemType);
  for (IndexType cellIndex = 0; cellIndex < cellIDsOfType.size(); cellIndex++)
  {
    if (cellIDsOfType[cellIndex] == cellID)
    {
      return cellIndex;
    }
  }
  return -1;
}

vector<TSolutionPtr<double>> GlobalDofAssignment::getRegisteredSolutions()
{
  return _registeredSolutions;
}

void GlobalDofAssignment::registerSolution(TSolutionPtr<double> solution)
{
  // Make a new, weak RCP, since Solution already has a pointer to Mesh, and Mesh has a pointer to us.
  TSolutionPtr<double> weakSolnPtr = Teuchos::rcp(solution.get(),false);
  _registeredSolutions.push_back( weakSolnPtr );
}

void GlobalDofAssignment::setMeshAndMeshTopology(MeshPtr mesh)
{
  // make copies of the RCPs that don't own memory.
  _mesh = Teuchos::rcp(mesh.get(), false);
  _meshTopology = Teuchos::rcp(mesh->getTopology().get(), false);

  this->DofInterpreter::_mesh = _mesh;
}

void GlobalDofAssignment::setPRefinementDegree(GlobalIndexType cellID, int deltaP)
{
  if (deltaP == 0)
  {
    _cellPRefinements.erase(cellID);
  }
  else
  {
    _cellPRefinements[cellID] = deltaP;
  }
}

void GlobalDofAssignment::unregisterSolution(TSolutionPtr<double> solution)
{
  for (vector< TSolutionPtr<double> >::iterator solnIt = _registeredSolutions.begin();
       solnIt != _registeredSolutions.end(); solnIt++)
  {
    if ( *solnIt == solution )
    {
      _registeredSolutions.erase(solnIt);
      return;
    }
  }
  cout << "GDAMaximumRule2D::unregisterSolution: Solution not found.\n";
}

// maximumRule2D provides support for legacy (MultiBasis) meshes
GlobalDofAssignmentPtr GlobalDofAssignment::maximumRule2D(MeshPtr mesh, VarFactoryPtr varFactory,
    DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
    unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
{
  return Teuchos::rcp( new GDAMaximumRule2D(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement) );
}

GlobalDofAssignmentPtr GlobalDofAssignment::minimumRule(MeshPtr mesh, VarFactoryPtr varFactory,
    DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
    unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
{
  return Teuchos::rcp( new GDAMinimumRule(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement) );
}
