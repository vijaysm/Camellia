//
//  MeshPartitionPolicy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 11/18/11.
//  Copyright 2011 Nathan Roberts. All rights reserved.
//

#include <iostream>

#include "MeshPartitionPolicy.h"

#include "GlobalDofAssignment.h"
#include "InducedMeshPartitionPolicy.h"
#include "MeshTools.h"
#include "MPIWrapper.h"
#include "ZoltanMeshPartitionPolicy.h"

using namespace Intrepid;
using namespace Camellia;
using namespace std;

MeshPartitionPolicy::MeshPartitionPolicy(Epetra_CommPtr Comm) : _Comm(Comm) {
  TEUCHOS_TEST_FOR_EXCEPTION(Comm == Teuchos::null, std::invalid_argument, "Comm may not be null!");
}

Epetra_CommPtr& MeshPartitionPolicy::Comm()
{
  return _Comm;
}

void MeshPartitionPolicy::partitionMesh(Mesh *mesh, PartitionIndexType numPartitions)
{
  // default simply divides the active cells into equally-sized partitions, in the order listed in activeCells…
  MeshTopologyViewPtr meshTopology = mesh->getTopology();
  int numActiveCells = meshTopology->activeCellCount(); // leaf nodes
  std::vector<std::set<GlobalIndexType> > partitionedActiveCells(numPartitions);

  int chunkSize = numActiveCells / numPartitions;
  int remainder = numActiveCells % numPartitions;
  IndexType activeCellOrdinal = 0;
  set<GlobalIndexType> activeCellIDsSet = mesh->getActiveCellIDsGlobal();
  vector<GlobalIndexType> activeCellIDs(activeCellIDsSet.begin(),activeCellIDsSet.end());
  TEUCHOS_TEST_FOR_EXCEPTION(activeCellIDs.size() != numActiveCells, std::invalid_argument, "meshTopology->getActiveCellCount() != mesh->getActiveCellIDsGlobal().size()");
  for (int i=0; i<numPartitions; i++)
  {
    int chunkSizeWithRemainder = (i < remainder) ? chunkSize + 1 : chunkSize;
    for (int j=0; j<chunkSizeWithRemainder; j++)
    {
      partitionedActiveCells[i].insert(activeCellIDs[activeCellOrdinal]);
      activeCellOrdinal++;
    }
  }
  
  mesh->globalDofAssignment()->setPartitions(partitionedActiveCells);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh)
{
  return InducedMeshPartitionPolicy::inducedMeshPartitionPolicy(thisMesh, otherMesh);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh, const map<GlobalIndexType,GlobalIndexType> &cellIDMap)
{
  return InducedMeshPartitionPolicy::inducedMeshPartitionPolicy(thisMesh, otherMesh, cellIDMap);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicyFromRefinedMesh(MeshTopologyViewPtr inducedMeshTopo, MeshPtr inducingRefinedMesh)
{
  // generate using the inducing mesh
  vector<GlobalIndexTypeToCast> myEntries;
  
  auto myCellIDs = &inducingRefinedMesh->cellIDsInPartition();
  for (GlobalIndexType myCellID : *myCellIDs)
  {
    IndexType ancestralCellIndex = myCellID;
    bool isFirstChild = true;
    while (isFirstChild && !inducedMeshTopo->isValidCellIndex(ancestralCellIndex))
    {
      CellPtr myCell = inducingRefinedMesh->getTopology()->getCell(ancestralCellIndex);
      CellPtr parent = myCell->getParent();
      TEUCHOS_TEST_FOR_EXCEPTION(parent == Teuchos::null, std::invalid_argument, "ancestor not found in inducedMeshTopo");
      int childOrdinal = parent->findChildOrdinal(myCell->cellIndex());
      isFirstChild = (childOrdinal == 0);
      ancestralCellIndex = parent->cellIndex();
    }
    if (isFirstChild)
    {
      myEntries.push_back(ancestralCellIndex);
      myEntries.push_back(myCellID);
    }
  }
  // all-gather the entries
  vector<GlobalIndexTypeToCast> allEntries;
  vector<int> offsets;
  MPIWrapper::allGatherCompact(*inducingRefinedMesh->Comm(), allEntries, myEntries, offsets);
  
  map<GlobalIndexType,GlobalIndexType> cellIDMap;
  for (int i=0; i<allEntries.size()/2; i++)
  {
    GlobalIndexType ancestralCellIndex = allEntries[2*i+0];
    GlobalIndexType inducingMeshCellID = allEntries[2*i+1];
    cellIDMap[ancestralCellIndex] = inducingMeshCellID;
  }
  
  return Teuchos::rcp( new InducedMeshPartitionPolicy(inducingRefinedMesh, cellIDMap) );
}

MeshPartitionPolicyPtr MeshPartitionPolicy::standardPartitionPolicy(Epetra_CommPtr Comm)
{
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new ZoltanMeshPartitionPolicy(Comm) );
  return partitionPolicy;
}

Teuchos_CommPtr& MeshPartitionPolicy::TeuchosComm()
{
  if (_TeuchosComm == Teuchos::null)
  {
#ifdef HAVE_MPI
    Epetra_MpiComm* mpiComm = dynamic_cast<Epetra_MpiComm*>(_Comm.get());
    if (mpiComm == NULL)
    {
      // serial communicator
      _TeuchosComm = MPIWrapper::TeuchosCommSerial();
    }
    else
    {
      if (mpiComm->GetMpiComm() == MPI_COMM_WORLD)
      {
        _TeuchosComm = MPIWrapper::TeuchosCommWorld();
      }
      else
      {
        _TeuchosComm = Teuchos::rcp( new Teuchos::MpiComm<int> (mpiComm->GetMpiComm()) );
      }
    }
#else
  // if we don't have MPI, then it can only be a serial communicator
    _TeuchosComm = MPIWrapper::TeuchosSerialComm();
#endif
  }
  return _TeuchosComm;
}