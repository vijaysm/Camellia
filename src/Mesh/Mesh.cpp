// @HEADER
//
// Original Version Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

/*
 *  Mesh.cpp
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "Mesh.h"
#include "ElementType.h"
#include "DofOrderingFactory.h"
#include "BasisFactory.h"
#include "BasisCache.h"

#include "Solution.h"

#include "MeshTransformationFunction.h"

#include "CamelliaCellTools.h"

#include "GlobalDofAssignment.h"

#include "GDAMinimumRule.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include <Teuchos_GlobalMPISession.hpp>

#ifdef HAVE_EPETRAEXT_HDF5
#include <EpetraExt_HDF5.h>
#include <Epetra_SerialComm.h>
#endif

#include "MeshFactory.h"

#include "MPIWrapper.h"

#include "ZoltanMeshPartitionPolicy.h"

#include <algorithm>

using namespace Intrepid;
using namespace Camellia;
using namespace std;

map<int,int> Mesh::_emptyIntIntMap;

Mesh::Mesh(MeshTopologyViewPtr meshTopology, VarFactoryPtr varFactory, vector<int> H1Order, int pToAddTest,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements,
           MeshPartitionPolicyPtr partitionPolicy, Epetra_CommPtr Comm) : DofInterpreter(Teuchos::rcp(this,false))
{
  _meshTopology = meshTopology;

  DofOrderingFactoryPtr dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory(varFactory, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
  initializePartitionPolicyIfNull(partitionPolicy, Comm);

  MeshPtr thisPtr = Teuchos::rcp(this, false);
  _gda = Teuchos::rcp( new GDAMinimumRule(thisPtr, varFactory, dofOrderingFactoryPtr,
                                          partitionPolicy, H1Order, pToAddTest));
  _gda->repartitionAndMigrate();

  _varFactory = varFactory;
  _boundary.setMesh(Teuchos::rcp(this,false));

  _meshTopology->setGlobalDofAssignment(_gda.get());

  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

Mesh::Mesh(MeshTopologyViewPtr meshTopology, VarFactoryPtr varFactory, int H1Order, int pToAddTest,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements,
           MeshPartitionPolicyPtr partitionPolicy, Epetra_CommPtr Comm) : DofInterpreter(Teuchos::rcp(this,false))
{

  _meshTopology = meshTopology;

  DofOrderingFactoryPtr dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory(varFactory, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
  initializePartitionPolicyIfNull(partitionPolicy, Comm);

  MeshPtr thisPtr = Teuchos::rcp(this, false);
  _gda = Teuchos::rcp( new GDAMinimumRule(thisPtr, varFactory, dofOrderingFactoryPtr,
                                          partitionPolicy, H1Order, pToAddTest));
  _gda->repartitionAndMigrate();

  _varFactory = varFactory;
  _boundary.setMesh(Teuchos::rcp(this,false));

  _meshTopology->setGlobalDofAssignment(_gda.get());

  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

// Deprecated constructor
Mesh::Mesh(MeshTopologyViewPtr meshTopology, TBFPtr<double> bilinearForm, vector<int> H1Order, int pToAddTest,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements,
           MeshPartitionPolicyPtr partitionPolicy, Epetra_CommPtr Comm) : DofInterpreter(Teuchos::rcp(this,false))
{

  _meshTopology = meshTopology;

  DofOrderingFactoryPtr dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory(bilinearForm, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
  //  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new MeshPartitionPolicy() );
  initializePartitionPolicyIfNull(partitionPolicy, Comm);

  MeshPtr thisPtr = Teuchos::rcp(this, false);
  _gda = Teuchos::rcp( new GDAMinimumRule(thisPtr, bilinearForm->varFactory(), dofOrderingFactoryPtr,
                                          partitionPolicy, H1Order, pToAddTest));
  _gda->repartitionAndMigrate();

  setBilinearForm(bilinearForm);
  _varFactory = bilinearForm->varFactory();
  _boundary.setMesh(Teuchos::rcp(this,false));

  _meshTopology->setGlobalDofAssignment(_gda.get());

  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

// Deprecated constructor
Mesh::Mesh(MeshTopologyViewPtr meshTopology, TBFPtr<double> bilinearForm, int H1Order, int pToAddTest,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements,
           MeshPartitionPolicyPtr partitionPolicy, Epetra_CommPtr Comm) : DofInterpreter(Teuchos::rcp(this,false))
{

  _meshTopology = meshTopology;

  DofOrderingFactoryPtr dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory(bilinearForm, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
//  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new MeshPartitionPolicy() );
  initializePartitionPolicyIfNull(partitionPolicy, Comm);

  MeshPtr thisPtr = Teuchos::rcp(this, false);
  _gda = Teuchos::rcp( new GDAMinimumRule(thisPtr, bilinearForm->varFactory(), dofOrderingFactoryPtr,
                                          partitionPolicy, H1Order, pToAddTest));
  _gda->repartitionAndMigrate();

  setBilinearForm(bilinearForm);
  _varFactory = bilinearForm->varFactory();
  _boundary.setMesh(Teuchos::rcp(this,false));

  _meshTopology->setGlobalDofAssignment(_gda.get());

  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

Mesh::Mesh(const vector<vector<double> > &vertices, vector< vector<unsigned> > &elementVertices,
           TBFPtr<double> bilinearForm, int H1Order, int pToAddTest, bool useConformingTraces,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements, vector<PeriodicBCPtr> periodicBCs,
           Epetra_CommPtr Comm) : DofInterpreter(Teuchos::rcp(this,false))
{

//  cout << "in legacy mesh constructor, periodicBCs size is " << periodicBCs.size() << endl;

  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices) );
  _meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry, periodicBCs) );

  DofOrderingFactoryPtr dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory(bilinearForm, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
  MeshPartitionPolicyPtr partitionPolicy;
  initializePartitionPolicyIfNull(partitionPolicy, Comm);

  MeshPtr thisPtr = Teuchos::rcp(this, false);
  _gda = Teuchos::rcp( new GDAMaximumRule2D(thisPtr, bilinearForm->varFactory(), dofOrderingFactoryPtr,
                       partitionPolicy, H1Order, pToAddTest, _enforceMBFluxContinuity) );
  _gda->repartitionAndMigrate();

  _meshTopology->setGlobalDofAssignment(_gda.get());

  setBilinearForm(bilinearForm);
  _varFactory = bilinearForm->varFactory();

  _useConformingTraces = useConformingTraces;
  _usePatchBasis = false;

  // DEBUGGING: check how we did:
  int numVertices = vertices.size();
  for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++ )
  {
    vector<double> vertex = _meshTopology->getVertex(vertexIndex);

    unsigned assignedVertexIndex;
    bool vertexFound = _meshTopology->getVertexIndex(vertex, assignedVertexIndex);

    if (!vertexFound)
    {
      cout << "INTERNAL ERROR: vertex not found by vertex lookup.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "internal error");
    }

    if (assignedVertexIndex != vertexIndex)
    {
      cout << "INTERNAL ERROR: assigned vertex index is incorrect.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "internal error");
    }
  }

  _boundary.setMesh(Teuchos::rcp(this,false));

  _pToAddToTest = pToAddTest;

  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

// private constructor for use by deepCopy()
Mesh::Mesh(MeshTopologyViewPtr meshTopology, Teuchos::RCP<GlobalDofAssignment> gda, VarFactoryPtr varFactory,
           int pToAddToTest, bool useConformingTraces, bool usePatchBasis, bool enforceMBFluxContinuity)
: DofInterpreter(Teuchos::rcp(this,false))
{
  _meshTopology = meshTopology;
  _gda = gda;
  _varFactory = varFactory;
  _pToAddToTest = pToAddToTest;
  _useConformingTraces = useConformingTraces;
  _usePatchBasis = usePatchBasis;
  _enforceMBFluxContinuity = enforceMBFluxContinuity;

  _boundary.setMesh(Teuchos::rcp(this,false));
}

// deprecated private constructor for use by deepCopy()
Mesh::Mesh(MeshTopologyViewPtr meshTopology, Teuchos::RCP<GlobalDofAssignment> gda, TBFPtr<double> bf,
           int pToAddToTest, bool useConformingTraces, bool usePatchBasis, bool enforceMBFluxContinuity) : DofInterpreter(Teuchos::rcp(this,false))
{
  _meshTopology = meshTopology;
  _gda = gda;
  _bilinearForm = bf;
  _varFactory = bf->varFactory();
  _pToAddToTest = pToAddToTest;
  _useConformingTraces = useConformingTraces;
  _usePatchBasis = usePatchBasis;
  _enforceMBFluxContinuity = enforceMBFluxContinuity;

  _boundary.setMesh(Teuchos::rcp(this,false));
}

// ! Constructor for a single-element mesh extracted from an existing mesh
Mesh::Mesh(MeshPtr mesh, GlobalIndexType cellID, Epetra_CommPtr Comm) : DofInterpreter(Teuchos::rcp(this,false))
{
  int meshDim = mesh->getTopology()->getDimension();
  Teuchos::RCP<MeshTopology> meshTopo = Teuchos::rcp( new MeshTopology(meshDim));
  _meshTopology = meshTopo;
  
  CellPtr cell = mesh->getTopology()->getCell(cellID);
  int vertexCount = cell->vertices().size();
  vector<vector<double> > cellVertices(vertexCount);

  const vector<unsigned>* vertexIndices = &cell->vertices();
  int i=0;
  for (unsigned vertexIndex : *vertexIndices)
  {
    cellVertices[i] = mesh->getTopology()->getVertex(vertexIndex);
    i++;
  }
  
  IndexType cellIDZero = 0;
  meshTopo->addCell(cellIDZero, cell->topology(), cellVertices);

  _varFactory = mesh->varFactory();
  
  DofOrderingFactoryPtr dofOrderingFactoryPtr = mesh->globalDofAssignment()->getDofOrderingFactory();
  MeshPartitionPolicyPtr partitionPolicy;
  initializePartitionPolicyIfNull(partitionPolicy, Comm);
  
  MeshPtr thisPtr = Teuchos::rcp(this, false);
  vector<int> H1Order = mesh->globalDofAssignment()->getInitialH1Order();
  int delta_k = mesh->globalDofAssignment()->getTestOrderEnrichment();
  _gda = Teuchos::rcp( new GDAMinimumRule(thisPtr, _varFactory, dofOrderingFactoryPtr,
                                          partitionPolicy, H1Order, delta_k));
  _gda->setElementType(cellIDZero,mesh->getElementType(cellID));
  _gda->repartitionAndMigrate();
  
  _boundary.setMesh(Teuchos::rcp(this,false));
  
  _meshTopology->setGlobalDofAssignment(_gda.get());
  
  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

GlobalIndexType Mesh::numInitialElements()
{
  return _meshTopology->getRootCellIndices().size();
}

GlobalIndexType Mesh::activeCellOffset()
{
  return _gda->activeCellOffset();
}

vector< ElementPtr > Mesh::activeElements()
{
  set< IndexType > activeCellIndices = _meshTopology->getActiveCellIndices();

  vector< ElementPtr > activeElements;

  for (set< IndexType >::iterator cellIt = activeCellIndices.begin(); cellIt != activeCellIndices.end(); cellIt++)
  {
    activeElements.push_back(getElement(*cellIt));
  }

  return activeElements;
}

ElementPtr Mesh::ancestralNeighborForSide(ElementPtr elem, int sideIndex, int &elemSideIndexInNeighbor)
{
  CellPtr cell = _meshTopology->getCell(elem->cellID());
  pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideIndex, _meshTopology);
  elemSideIndexInNeighbor = neighborInfo.second;

  if (neighborInfo.first == -1) return Teuchos::rcp( (Element*) NULL );

  return getElement(neighborInfo.first);
}

TBFPtr<double> Mesh::bilinearForm()
{
  return _bilinearForm;
}

void Mesh::setBilinearForm( TBFPtr<double> bf)
{
  // must match the original in terms of variable IDs, etc...
  _bilinearForm = bf;
}

Boundary & Mesh::boundary()
{
  return _boundary;
}

GlobalIndexType Mesh::cellID(Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber)
{
  return _gda->cellID(elemTypePtr, cellIndex, partitionNumber);
}

vector< GlobalIndexType > Mesh::cellIDsOfType(ElementTypePtr elemType)
{
  int rank = Comm()->MyPID();
  return cellIDsOfType(rank,elemType);
}

vector< GlobalIndexType > Mesh::cellIDsOfType(int partitionNumber, ElementTypePtr elemTypePtr)
{
  // returns the cell IDs for a given partition and element type
  return _gda->cellIDsOfElementType(partitionNumber, elemTypePtr);
}

vector< GlobalIndexType > Mesh::cellIDsOfTypeGlobal(ElementTypePtr elemTypePtr)
{
  vector< GlobalIndexType > cellIDs;
  int partititionCount = _gda->getPartitionCount();
  for (int partitionNumber=0; partitionNumber<partititionCount; partitionNumber++)
  {
    vector< GlobalIndexType > cellIDsForType = cellIDsOfType(partitionNumber, elemTypePtr);
    cellIDs.insert(cellIDs.end(), cellIDsForType.begin(), cellIDsForType.end());
  }
  return cellIDs;
}

const set<GlobalIndexType> & Mesh::cellIDsInPartition()
{
  return _gda->cellsInPartition(-1);
}

bool Mesh::cellIsActive(GlobalIndexType cellID) const
{
  return _meshTopology->getActiveCellIndices().find(cellID) != _meshTopology->getActiveCellIndices().end();
}

int Mesh::cellPolyOrder(GlobalIndexType cellID)   // aka H1Order
{
  return _gda->getH1Order(cellID)[0];
}

vector<int> Mesh::cellTensorPolyOrder(GlobalIndexType cellID)   // aka H1Order
{
  return _gda->getH1Order(cellID);
}

vector<GlobalIndexType> Mesh::cellIDsForPoints(const FieldContainer<double> &physicalPoints, bool minusOnesIfOffRank)
{
  vector<GlobalIndexType> cellIDs = _meshTopology->cellIDsForPoints(physicalPoints);

  if (minusOnesIfOffRank)
  {
    set<GlobalIndexType> rankLocalCellIDs = cellIDsInPartition();
    for (int i=0; i<cellIDs.size(); i++)
    {
      if (rankLocalCellIDs.find(cellIDs[i]) == rankLocalCellIDs.end())
      {
        cellIDs[i] = -1;
      }
    }
  }
  return cellIDs;
}

Epetra_CommPtr& Mesh::Comm()
{
  return _gda->getPartitionPolicy()->Comm();
}

MeshPtr Mesh::deepCopy()
{
  MeshTopologyViewPtr meshTopoCopy = _meshTopology->deepCopy();
  GlobalDofAssignmentPtr gdaCopy = _gda->deepCopy();

  MeshPtr meshCopy = Teuchos::rcp( new Mesh(meshTopoCopy, gdaCopy, _bilinearForm, _pToAddToTest, _useConformingTraces, _usePatchBasis, _enforceMBFluxContinuity ));
  gdaCopy->setMeshAndMeshTopology(meshCopy);
  return meshCopy;
}

vector<ElementPtr> Mesh::elementsForPoints(const FieldContainer<double> &physicalPoints, bool nullElementsIfOffRank)
{

  vector<GlobalIndexType> cellIDs = cellIDsForPoints(physicalPoints, nullElementsIfOffRank);
  vector<ElementPtr> elemsForPoints(cellIDs.size());

  for (int i=0; i<cellIDs.size(); i++)
  {
    GlobalIndexType cellID = cellIDs[i];
    ElementPtr elem;
    if (cellID==-1)
    {
      elem = Teuchos::rcp( (Element*) NULL);
    }
    else
    {
      elem = getElement(cellID);
    }
    elemsForPoints[i] = elem;
  }
  return elemsForPoints;
}

void Mesh::enforceOneIrregularity(bool repartitionAndMigrate)
{
  int rank = Comm()->MyPID();
  bool meshIsNotRegular = true; // assume it's not regular and check elements
  bool meshChanged = false;
  
  while (meshIsNotRegular)
  {
    int spaceDim = _meshTopology->getDimension();
    if (spaceDim == 1) return;

    map< Camellia::CellTopologyKey, set<GlobalIndexType> > irregularCellIDs; // key is CellTopology key
    set< GlobalIndexType > activeCellIDs = _meshTopology->getActiveCellIndices();
    set< GlobalIndexType >::iterator cellIDIt;

    bool useSideIrregularityEnforcement = false; // this is the old way
    if (useSideIrregularityEnforcement)
    {
      for (GlobalIndexType cellID : activeCellIDs)
      {
        CellPtr cell = _meshTopology->getCell(cellID);
        int sideCount = cell->getSideCount();
        for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++)
        {
          pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, _meshTopology);

          if (neighborInfo.first != -1) // I have a neighbor
          {
            if (spaceDim > 1)
            {
              CellPtr neighbor = _meshTopology->getCell(neighborInfo.first);
              RefinementBranch myRefinementBranch = cell->refinementBranchForSide(sideOrdinal, _meshTopology);
              if (myRefinementBranch.size() > 1)
              {
                // then *neighbor* is irregular
                irregularCellIDs[neighbor->topology()->getKey()].insert(neighborInfo.first);
  //              cout << neighborInfo.first << " is irregular.\n";
                { // DEBUGGING:
                  if (activeCellIDs.find(neighborInfo.first) == activeCellIDs.end())
                  {
                    _meshTopology->printAllEntitiesInBaseMeshTopology();
                    // repeat for entering in the debugger before the exception is thrown
                    cell->getNeighborInfo(sideOrdinal, _meshTopology);
                  }
                }
                TEUCHOS_TEST_FOR_EXCEPTION(activeCellIDs.find(neighborInfo.first) == activeCellIDs.end(),
                                           std::invalid_argument, "Internal error: 'irregular' cell is not active!");
              }
            }
          }
        }
      }
    }
    else // new way: edge 1-irregularity enforcement
    {
      for (GlobalIndexType cellID : activeCellIDs)
      {
        CellPtr cell = _meshTopology->getCell(cellID);
        int edgeCount = cell->topology()->getEdgeCount();
        static int edgeDim = 1;
        for (int edgeOrdinal=0; edgeOrdinal < edgeCount; edgeOrdinal++)
        {
          int refBranchSize = cell->refinementBranchForSubcell(edgeDim, edgeOrdinal, _meshTopology).size();
          
          if (refBranchSize > 1)
          {
            // then there is at least one active 2-irregular cell constraining this edge
            IndexType edgeEntityIndex = cell->entityIndex(edgeDim, edgeOrdinal);
            pair<IndexType, unsigned> constrainingEntity = _meshTopology->getConstrainingEntity(edgeDim, edgeEntityIndex);
            IndexType constrainingEntityIndex = constrainingEntity.first;
            unsigned constrainingEntityDim = constrainingEntity.second;
            std::vector< std::pair<IndexType,unsigned> > activeCellsForConstrainingEntity = _meshTopology->getActiveCellIndices(constrainingEntityDim, constrainingEntityIndex);
            for (auto activeCellEntry : activeCellsForConstrainingEntity)
            {
              CellTopologyKey cellTopoKey = _meshTopology->getCell(activeCellEntry.first)->topology()->getKey();
              irregularCellIDs[cellTopoKey].insert(activeCellEntry.first);
            }
          }
        }
        
        // One other thing to check has to do with interior children (e.g. the middle triangle in a regular triangle refinement)
        // If the parent of an active cell is an interior child, then all of its parent's neighbors should be refined if
        // they aren't already.
        // TODO: compare this with strategies in the literature.  (Particularly Leszek's.)
        /* 
         NOTE: this logic is not perfectly general.  In particular, it assumes that if the grandparent's neighbors *are*
               refined, they are refined in a way that makes them compatible.  In the case e.g. of anisotropic refinements,
               this need not be the case.  If a null refinement has made its way into the mesh, the same thing applies.
         */
        CellPtr parent = cell->getParent();
        if ((parent != Teuchos::null) && parent->isInteriorChild())
        {
          CellPtr grandparent = parent->getParent();
          int grandparentSideCount = grandparent->topology()->getSideCount();
          for (int grandparentSideOrdinal=0; grandparentSideOrdinal < grandparentSideCount; grandparentSideOrdinal++)
          {
            CellPtr grandparentNeighbor = grandparent->getNeighbor(grandparentSideOrdinal, _meshTopology);
            if ((grandparentNeighbor != Teuchos::null) && (!grandparentNeighbor->isParent(_meshTopology)))
            {
              irregularCellIDs[grandparentNeighbor->topology()->getKey()].insert(grandparentNeighbor->cellIndex());
            }
          }
        }
      }
    }
    
    if (irregularCellIDs.size() > 0)
    {
      for (map< Camellia::CellTopologyKey, set<GlobalIndexType> >::iterator mapIt = irregularCellIDs.begin();
           mapIt != irregularCellIDs.end(); mapIt++)
      {
        Camellia::CellTopologyKey cellKey = mapIt->first;
        hRefine(mapIt->second, RefinementPattern::regularRefinementPattern(cellKey), false); // false: don't repartition and rebuild, yet.
      }
      irregularCellIDs.clear();
      meshChanged = true;
    }
    else
    {
      meshIsNotRegular = false;
    }
  }
  if (meshChanged && repartitionAndMigrate)
  {
    // then repartition and migrate now
    repartitionAndRebuild();
  }
}

FieldContainer<double> Mesh::cellSideParities( ElementTypePtr elemTypePtr )
{
  // old version (using lookup table)
  // return dynamic_cast<GDAMaximumRule2D*>(_gda.get())->cellSideParities(elemTypePtr);

  // new implementation below:
  int rank = Comm()->MyPID();
  vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(rank, elemTypePtr);

  int numCells = cellIDs.size();
  int numSides = elemTypePtr->cellTopoPtr->getSideCount();

  FieldContainer<double> sideParities(numCells, numSides);
  for (int i=0; i<numCells; i++)
  {
    FieldContainer<double> iParities = cellSideParitiesForCell(cellIDs[i]);
    for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
    {
      sideParities(i,sideOrdinal) = iParities(0,sideOrdinal);
    }
  }

  return sideParities;
}

FieldContainer<double> Mesh::cellSideParitiesForCell( GlobalIndexType cellID )
{
  return _gda->cellSideParitiesForCell(cellID);
}

vector<double> Mesh::getCellCentroid(GlobalIndexType cellID)
{
  return _meshTopology->getCellCentroid(cellID);
}

vector< ElementPtr > Mesh::elementsInPartition(PartitionIndexType partitionNumber)
{
  set< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(partitionNumber);
  vector< ElementPtr > elements;
  for (set< GlobalIndexType >::iterator cellIt = cellsInPartition.begin(); cellIt != cellsInPartition.end(); cellIt++)
  {
    GlobalIndexType cellID = *cellIt;
    ElementPtr element = getElement(cellID);
    elements.push_back(element);
  }
  return elements;
}

vector< ElementPtr > Mesh::elementsOfType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr)
{
  // returns the elements for a given partition and element type
  vector< ElementPtr > elementsOfType;
  vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(partitionNumber, elemTypePtr);
  for (vector<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++)
  {
    elementsOfType.push_back(getElement(*cellIt));
  }
  return elementsOfType;
}

vector< ElementPtr > Mesh::elementsOfTypeGlobal(ElementTypePtr elemTypePtr)
{
  vector< ElementPtr > elementsOfTypeVector;
  int partitionCount = _gda->getPartitionCount();
  for (int partitionNumber=0; partitionNumber<partitionCount; partitionNumber++)
  {
    vector< ElementPtr > elementsOfTypeForPartition = elementsOfType(partitionNumber,elemTypePtr);
    elementsOfTypeVector.insert(elementsOfTypeVector.end(),elementsOfTypeForPartition.begin(),elementsOfTypeForPartition.end());
  }
  return elementsOfTypeVector;
}

vector< ElementTypePtr > Mesh::elementTypes(PartitionIndexType partitionNumber)
{
  return _gda->elementTypes(partitionNumber);
}

set<GlobalIndexType> Mesh::getActiveCellIDs()
{
  set<IndexType> activeCellIndices = _meshTopology->getActiveCellIndices();
  set<GlobalIndexType> activeCellIDs(activeCellIndices.begin(), activeCellIndices.end());
  return activeCellIDs;
}

int Mesh::getDimension()
{
  return _meshTopology->getDimension();
}

DofOrderingFactory & Mesh::getDofOrderingFactory()
{
  return *_gda->getDofOrderingFactory().get();
}

ElementPtr Mesh::getElement(GlobalIndexType cellID)
{
  CellPtr cell = _meshTopology->getCell(cellID);

  ElementTypePtr elemType = _gda->elementType(cellID);

  IndexType cellIndex = _gda->partitionLocalCellIndex(cellID);

  GlobalIndexType globalCellIndex = _gda->globalCellIndex(cellID);

  ElementPtr element = Teuchos::rcp( new Element(this, cellID, elemType, cellIndex, globalCellIndex) );

  return element;
}

ElementTypePtr Mesh::getElementType(GlobalIndexType cellID)
{
  return _gda->elementType(cellID);
}

ElementTypeFactory & Mesh::getElementTypeFactory()
{
  return _gda->getElementTypeFactory();
}

GlobalIndexType Mesh::getVertexIndex(double x, double y, double tol)
{
  vector<double> vertex;
  vertex.push_back(x);
  vertex.push_back(y);

  IndexType vertexIndex; // distributed mesh will need to use some sort of offset...
  if (! _meshTopology->getVertexIndex(vertex, vertexIndex) )
  {
    return -1;
  }
  else
  {
    return vertexIndex;
  }
}

const map< pair<GlobalIndexType,IndexType>, GlobalIndexType>& Mesh::getLocalToGlobalMap()
{
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL)
  {
    cout << "getLocalToGlobalMap only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getLocalToGlobalMap only supported for max rule.");
  }
  return maxRule->getLocalToGlobalMap();
}

bool Mesh::meshUsesMaximumRule()
{
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  return (maxRule != NULL);
}

bool Mesh::meshUsesMinimumRule()
{
  GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule *>(_gda.get());
  return (minRule != NULL);
}

set<GlobalIndexType> Mesh::getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal)
{
  return _gda->getGlobalDofIndices(cellID,varID,sideOrdinal);
}

map<IndexType, GlobalIndexType> Mesh::getGlobalVertexIDs(const FieldContainer<double> &vertices)
{
  double tol = 1e-12; // tolerance for vertex equality

  map<IndexType, GlobalIndexType> localToGlobalVertexIndex;
  int numVertices = vertices.dimension(0);
  for (int i=0; i<numVertices; i++)
  {
    localToGlobalVertexIndex[i] = getVertexIndex(vertices(i,0), vertices(i,1),tol);
  }
  return localToGlobalVertexIndex;
}

TFunctionPtr<double> Mesh::getTransformationFunction()
{
  // will be NULL for meshes without edge curves defined -- including those built around pure MeshTopologyView instances
  
  // for now, we recompute the transformation function each time the edge curves get updated
  // we might later want to do something lazier, updating/creating it here if it's out of date

  return _meshTopology->transformationFunction();
}

GlobalDofAssignmentPtr Mesh::globalDofAssignment()
{
  return _gda;
}

GlobalIndexType Mesh::globalDofCount()
{
  return numGlobalDofs(); // TODO: eliminate numGlobalDofs in favor of globalDofCount
}

GlobalIndexType Mesh::globalDofIndex(GlobalIndexType cellID, IndexType localDofIndex)
{
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL)
  {
    cout << "globalDofIndex lookup only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofIndex lookup only supported for max rule.");
  }
  return maxRule->globalDofIndex(cellID, localDofIndex);
}

set<GlobalIndexType> Mesh::globalDofIndicesForCell(GlobalIndexType cellID)
{
  return _gda->globalDofIndicesForCell(cellID);
}

set<GlobalIndexType> Mesh::globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal)
{
  return _gda->globalDofIndicesForVarOnSubcell(varID, cellID, dim, subcellOrdinal);
}

set<GlobalIndexType> Mesh::globalDofIndicesForPartition(PartitionIndexType partitionNumber)
{
  return _gda->globalDofIndicesForPartition(partitionNumber);
}

//void Mesh::hRefine(vector<GlobalIndexType> cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
//  hRefine(cellIDs,refPattern,vector< TSolutionPtr<double> >());
//}

void Mesh::hRefine(const vector<GlobalIndexType> &cellIDs, bool repartitionAndRebuild)
{
  set<GlobalIndexType> cellSet(cellIDs.begin(),cellIDs.end());
  hRefine(cellSet, repartitionAndRebuild);
}

void Mesh::hRefine(const set<GlobalIndexType> &cellIDs, bool repartitionAndRebuild)
{
  map< CellTopologyKey, set<GlobalIndexType> > cellIDsForTopo;
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end();
       cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    CellTopoPtr cellTopo = getElementType(cellID)->cellTopoPtr;
    cellIDsForTopo[cellTopo->getKey()].insert(cellID);
  }

  for (map< CellTopologyKey, set<GlobalIndexType> >::iterator entryIt = cellIDsForTopo.begin();
       entryIt != cellIDsForTopo.end(); entryIt++)
  {
    CellTopologyKey cellTopoKey = entryIt->first;
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopoKey);
    hRefine(entryIt->second, refPattern, repartitionAndRebuild);
  }
}

void Mesh::hRefine(const vector<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern)
{
  set<GlobalIndexType> cellSet(cellIDs.begin(),cellIDs.end());
  hRefine(cellSet,refPattern);
}

void Mesh::hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern)
{
  hRefine(cellIDs, refPattern, true);
}

void Mesh::hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool repartitionAndRebuild)
{
  if (cellIDs.size() == 0) return;

  MeshTopology* meshTopologyInstance = dynamic_cast<MeshTopology*>(_meshTopology.get());
  
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopologyInstance, std::invalid_argument, "Mesh::hRefine() called when _meshTopology is not an instance of MeshTopology--likely Mesh initialized with a pure MeshTopologyView, which cannot be h-refined.");
  
  MeshTopologyPtr writableMeshTopology = Teuchos::rcp(meshTopologyInstance, false);
  
  // send h-refinement message any registered observers (may be meshes)
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++)
  {
    (*meshIt)->hRefine(writableMeshTopology,cellIDs,refPattern);
  }

  set<GlobalIndexType>::const_iterator cellIt;

  // we do something slightly different for max rule because it wants to know about each cell as it
  // gets refined.  For reasons I'm not entirely clear on.
  bool usingMaxRule = this->meshUsesMaximumRule();
  
  GlobalIndexType nextCellID = writableMeshTopology->cellCount();
  for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++)
  {
    GlobalIndexType cellID = *cellIt;

    if (writableMeshTopology->getActiveCellIndices().find(cellID) == writableMeshTopology->getActiveCellIndices().end())
    {
      cout << "cellID " << cellID << " is not active, but Mesh received request for h-refinement.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "inactive cell");
    }
    
    writableMeshTopology->refineCell(cellID, refPattern, nextCellID); // TODO: establish nextCellID through a parallel scan so that we don't have to do the refinement on every MPI rank.
    nextCellID += refPattern->numChildren();

    // TODO: figure out what it is that breaks in GDAMaximumRule when we use didHRefine to notify about all cells together outside this loop
    if (usingMaxRule)
    {
      set<GlobalIndexType> cellIDset;
      cellIDset.insert(cellID);

      _gda->didHRefine(cellIDset);
    }
  }
  
  if (!usingMaxRule)
  {
    // TODO: consider making GDA a refinementObserver, using that interface to send it the notification
    _gda->didHRefine(cellIDs);
  }

  // NVR 12/10/14 the code below moved from inside the loop above, where it was doing the below one cell at a time...
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator observerIt = _registeredObservers.begin();
       observerIt != _registeredObservers.end(); observerIt++)
  {
    (*observerIt)->didHRefine(writableMeshTopology,cellIDs,refPattern);
  }

  // TODO: consider making transformation function a refinementObserver, using that interface to send it the notification
  // let transformation function know about the refinement that just took place
  if (writableMeshTopology->transformationFunction().get())
  {
    writableMeshTopology->transformationFunction()->didHRefine(cellIDs);
  }

  if (repartitionAndRebuild)
  {
    this->repartitionAndRebuild();
  }
}

void Mesh::hUnrefine(const set<GlobalIndexType> &cellIDs, bool repartitionAndRebuild)
{
  if (cellIDs.size() == 0) return;

  // refine any registered meshes
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++)
  {
    (*meshIt)->hUnrefine(cellIDs);
  }

  MeshTopology* meshTopologyInstance = dynamic_cast<MeshTopology*>(_meshTopology.get());

  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopologyInstance, std::invalid_argument, "Mesh::hUnrefine() called when _meshTopology is not an instance of MeshTopology--likely Mesh initialized with a pure MeshTopologyView, which cannot be h-unrefined.");
  
  MeshTopologyPtr writableMeshTopology = Teuchos::rcp(meshTopologyInstance, false);

  // TODO: finish implementing this
  
//  set<GlobalIndexType>::const_iterator cellIt;
//  set< pair<GlobalIndexType, int> > affectedNeighborSides; // (cellID, sideIndex)
//  set< GlobalIndexType > deletedCellIDs;
//
//  for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
//    GlobalIndexType cellID = *cellIt;
//    ElementPtr elem = getElement(cellID);
//    elem->deleteChildrenFromMesh(affectedNeighborSides, deletedCellIDs);
//  }
//
//  set<int> affectedNeighbors;
//  // for each nullified neighbor relationship, need to figure out the correct element type...
//  for ( set< pair<int, int> >::iterator neighborIt = affectedNeighborSides.begin();
//       neighborIt != affectedNeighborSides.end(); neighborIt++) {
//    ElementPtr elem = _elements[ neighborIt->first ];
//    if (elem->isActive()) {
//      matchNeighbor( elem, neighborIt->second );
//    }
//  }
//
//  // delete any boundary entries for deleted elements
//  for (set<int>::iterator cellIt = deletedCellIDs.begin(); cellIt != deletedCellIDs.end(); cellIt++) {
//    int cellID = *cellIt;
//    ElementPtr elem = _elements[cellID];
//    for (int sideIndex=0; sideIndex<elem->numSides(); sideIndex++) {
//      // boundary allows us to delete even combinations that weren't there to begin with...
//      _boundary.deleteElement(cellID, sideIndex);
//    }
//  }

//  // add in any new boundary elements:
//  for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
//    int cellID = *cellIt;
//    ElementPtr elem = _elements[cellID];
//    if (elem->isActive()) {
//      int elemSideIndexInNeighbor;
//      for (int sideIndex=0; sideIndex<elem->numSides(); sideIndex++) {
//        if (ancestralNeighborForSide(elem, sideIndex, elemSideIndexInNeighbor)->cellID() == -1) {
//          // boundary
//          _boundary.addElement(cellID, sideIndex);
//        }
//      }
//    }
//  }

//  // added by Jesse to try to fix bug
//  for (set<int>::iterator cellIt = deletedCellIDs.begin(); cellIt != deletedCellIDs.end(); cellIt++) {
//    // erase from _elements list
//    for (int i = 0; i<_elements.size();i++){
//      if (_elements[i]->cellID()==(*cellIt)){
//        _elements.erase(_elements.begin()+i);
//        break;
//      }
//    }
//    // erase any pairs from _edgeToCellIDs having to do with deleted cellIDs
//    for (map<pair<int,int>, vector<pair<int,int> > >::iterator mapIt = _edgeToCellIDs.begin(); mapIt!=_edgeToCellIDs.end();mapIt++){
//      vector<pair<int,int> > cellIDSideIndices = mapIt->second;
//      bool eraseEntry = false;
//      for (int i = 0;i<cellIDSideIndices.size();i++){
//        int cellID = cellIDSideIndices[i].first;
//        if (cellID==(*cellIt)){
//          eraseEntry = true;
//        }
//        if (eraseEntry)
//          break;
//      }
//      if (eraseEntry){
//        _edgeToCellIDs.erase(mapIt);
////        cout << "deleting edge to cell entry " << mapIt->first.first << " --> " << mapIt->first.second << endl;
//      }
//    }
//  }

  // TODO: consider making GDA a RefinementObserver, and using that interface to send the notification of unrefinement.
  _gda->didHUnrefine(cellIDs);

  // notify observers that of the unrefinement that just happened
  for (Teuchos::RCP<RefinementObserver> refinementObserver : _registeredObservers)
  {
    refinementObserver->didHUnrefine(writableMeshTopology,cellIDs);
  }

  if (repartitionAndRebuild)
  {
    _gda->repartitionAndMigrate();
    _boundary.buildLookupTables();
  }
}

void Mesh::initializePartitionPolicyIfNull(MeshPartitionPolicyPtr &partitionPolicy, Epetra_CommPtr Comm)
{
  if ( partitionPolicy.get() == NULL )
  {
    if (Comm == Teuchos::null)
    {
#ifdef HAVE_MPI
      Comm = Teuchos::rcp( new Epetra_MpiComm(MPI_COMM_WORLD) );
#else
      Comm = Teuchos::rcp( new Epetra_SerialComm() );
#endif
    }
    partitionPolicy = Teuchos::rcp( new ZoltanMeshPartitionPolicy(Comm) );
  }
}

void Mesh::interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localCoefficients, const Epetra_MultiVector &globalCoefficients)
{
  _gda->interpretGlobalCoefficients(cellID, localCoefficients, globalCoefficients);
}

void Mesh::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<double> &basisCoefficients,
    FieldContainer<double> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices)
{
  _gda->interpretLocalBasisCoefficients(cellID, varID, sideOrdinal, basisCoefficients, globalCoefficients, globalDofIndices);
}

void Mesh::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                              FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices)
{
  _gda->interpretLocalData(cellID, localDofs, globalDofs, globalDofIndices);
}

int Mesh::irregularity()
{
  const set<GlobalIndexType>* myCells = &this->cellIDsInPartition();
  int myIrregularity = 0; // for this partition
  for (GlobalIndexType cellID : *myCells)
  {
    CellPtr cell = _meshTopology->getCell(cellID);
    int edgeCount = cell->topology()->getEdgeCount();
    int edgeDim = 1;
    for (int edgeOrdinal=0; edgeOrdinal<edgeCount; edgeOrdinal++)
    {
      int edgeIrregularity = cell->refinementBranchForSubcell(edgeDim, edgeOrdinal, _meshTopology).size();
      myIrregularity = max(edgeIrregularity, myIrregularity);
    }
  }
  int globalIrregularity = 0;
  Comm()->MaxAll(&myIrregularity, &globalIrregularity, 1);
  return globalIrregularity;
}

GlobalIndexType Mesh::numActiveElements()
{
  return _meshTopology->getActiveCellIndices().size();
}

GlobalIndexType Mesh::numElements()
{
  return _meshTopology->cellCount();
}

GlobalIndexType Mesh::numElementsOfType( Teuchos::RCP< ElementType > elemTypePtr )
{
  // returns the global total (across all MPI nodes)
  int numElements = 0;
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  for (PartitionIndexType partitionNumber=0; partitionNumber<partitionCount; partitionNumber++)
  {
    numElements += _gda->cellIDsOfElementType(partitionNumber, elemTypePtr).size();
  }
  return numElements;
}

GlobalIndexType Mesh::numFluxDofs()
{
  GlobalIndexType fluxDofsForPartition = _gda->numPartitionOwnedGlobalFluxIndices();
  GlobalIndexType traceDofsForPartition = _gda->numPartitionOwnedGlobalTraceIndices();
//  GlobalIndexType fluxDofsForPartition = _gda->partitionOwnedGlobalFluxIndices().size();
//  GlobalIndexType traceDofsForPartition = _gda->partitionOwnedGlobalTraceIndices().size();

  return MPIWrapper::sum(*Comm(), (GlobalIndexTypeToCast)(fluxDofsForPartition + traceDofsForPartition));
}

GlobalIndexType Mesh::numFieldDofs()
{
  GlobalIndexType fieldDofsForPartition = _gda->numPartitionOwnedGlobalFieldIndices();
  return MPIWrapper::sum(*Comm(),(GlobalIndexTypeToCast)fieldDofsForPartition);
}

GlobalIndexType Mesh::numGlobalDofs()
{
  return _gda->globalDofCount();
}

int Mesh::parityForSide(GlobalIndexType cellID, int sideOrdinal)
{
  int parity = _gda->cellSideParitiesForCell(cellID)[sideOrdinal];
  return parity;
}

PartitionIndexType Mesh::partitionForCellID( GlobalIndexType cellID )
{
  return _gda->partitionForCellID(cellID);
}

PartitionIndexType Mesh::partitionForGlobalDofIndex( GlobalIndexType globalDofIndex )
{
  return _gda->partitionForGlobalDofIndex(globalDofIndex);
}

GlobalIndexType Mesh::partitionLocalIndexForGlobalDofIndex( GlobalIndexType globalDofIndex )
{
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL)
  {
    cout << "partitionLocalIndexForGlobalDofIndex only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "partitionLocalIndexForGlobalDofIndex only supported for max rule.");
  }
  return maxRule->partitionLocalIndexForGlobalDofIndex(globalDofIndex);
}

FieldContainer<double> Mesh::physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr)
{
  int rank = Comm()->MyPID();
  vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(rank, elemTypePtr);

  return physicalCellNodes(elemTypePtr, cellIDs);
}

FieldContainer<double> Mesh::physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr, vector<GlobalIndexType> &cellIDs )
{
  int numCells = cellIDs.size();
  int numVertices = elemTypePtr->cellTopoPtr->getVertexCount();
  int spaceDim = _meshTopology->getDimension();

  FieldContainer<double> physicalNodes(numCells, numVertices, spaceDim);
  for (int i=0; i<numCells; i++)
  {
    FieldContainer<double> iPhysicalNodes = physicalCellNodesForCell(cellIDs[i]);
    for (int vertexOrdinal=0; vertexOrdinal<numVertices; vertexOrdinal++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        physicalNodes(i,vertexOrdinal,d) = iPhysicalNodes(0,vertexOrdinal,d);
      }
    }
  }
  return physicalNodes;
}

FieldContainer<double> Mesh::physicalCellNodesForCell( GlobalIndexType cellID )
{
  CellPtr cell = _meshTopology->getCell(cellID);
  int vertexCount = cell->topology()->getVertexCount();
  int spaceDim = _meshTopology->getDimension();
  int numCells = 1;
  FieldContainer<double> physicalCellNodes(numCells,vertexCount,spaceDim);

  FieldContainer<double> cellVertices(vertexCount,spaceDim);
  vector<unsigned> vertexIndices = _meshTopology->getCell(cellID)->vertices();
  for (int vertex=0; vertex<vertexCount; vertex++)
  {
    unsigned vertexIndex = vertexIndices[vertex];
    for (int i=0; i<spaceDim; i++)
    {
      physicalCellNodes(0,vertex,i) = _meshTopology->getVertex(vertexIndex)[i];
    }
  }
  return physicalCellNodes;
}

FieldContainer<double> Mesh::physicalCellNodesGlobal( Teuchos::RCP< ElementType > elemTypePtr )
{
//  int numRanks = Teuchos::GlobalMPISession::getNProc();

  // user should call cellIDsOfTypeGlobal() to get the corresponding cell IDs (the cell nodes are *NOT* sorted by cell ID)

  vector<GlobalIndexType> globalCellIDs = cellIDsOfTypeGlobal(elemTypePtr);
//  for (int rank=0; rank<numRanks; rank++) {
//    vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(rank, elemTypePtr);
//    globalCellIDs.insert(globalCellIDs.end(), cellIDs.begin(), cellIDs.end());
//  }

  return physicalCellNodes(elemTypePtr, globalCellIDs);
}

void Mesh::printLocalToGlobalMap()
{
  map< pair<GlobalIndexType,IndexType>, GlobalIndexType> localToGlobalMap = this->getLocalToGlobalMap();

  for (map< pair<GlobalIndexType,IndexType>, GlobalIndexType>::iterator entryIt = localToGlobalMap.begin();
       entryIt != localToGlobalMap.end(); entryIt++)
  {
    int cellID = entryIt->first.first;
    int localDofIndex = entryIt->first.second;
    int globalDofIndex = entryIt->second;
    cout << "(" << cellID << "," << localDofIndex << ") --> " << globalDofIndex << endl;
  }
}

void Mesh::registerObserver(Teuchos::RCP<RefinementObserver> observer)
{
  _registeredObservers.push_back(observer);
}

template <typename Scalar>
void Mesh::registerSolution(TSolutionPtr<Scalar> solution)
{
  _gda->registerSolution(solution);
}

void Mesh::repartitionAndRebuild()
{
  _gda->repartitionAndMigrate();
  _boundary.buildLookupTables();
  
  MeshTopology* meshTopologyInstance = dynamic_cast<MeshTopology*>(_meshTopology.get());
  
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopologyInstance, std::invalid_argument, "Mesh::hRefine() called when _meshTopology is not an instance of MeshTopology--likely Mesh initialized with a pure MeshTopologyView, which cannot be h-refined.");
  
  MeshTopologyPtr writableMeshTopology = Teuchos::rcp(meshTopologyInstance, false);
  
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator observerIt = _registeredObservers.begin();
       observerIt != _registeredObservers.end(); observerIt++)
  {
    (*observerIt)->didRepartition(writableMeshTopology);
  }
}

void Mesh::unregisterObserver(RefinementObserver* observer)
{
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++)
  {
    if ( (*meshIt).get() == observer )
    {
      _registeredObservers.erase(meshIt);
      return;
    }
  }
  cout << "WARNING: Mesh::unregisterObserver: Observer not found.\n";
}

void Mesh::unregisterObserver(Teuchos::RCP<RefinementObserver> mesh)
{
  this->unregisterObserver(mesh.get());
}

template <typename Scalar>
void Mesh::unregisterSolution(TSolutionPtr<Scalar> solution)
{
  _gda->unregisterSolution(solution);
}

void Mesh::pRefine(const vector<GlobalIndexType> &cellIDsForPRefinements)
{
  set<GlobalIndexType> cellSet;
  for (vector<GlobalIndexType>::const_iterator cellIt=cellIDsForPRefinements.begin();
       cellIt != cellIDsForPRefinements.end(); cellIt++)
  {
    cellSet.insert(*cellIt);
  }
  pRefine(cellSet);
}

void Mesh::pRefine(const set<GlobalIndexType> &cellIDsForPRefinements)
{
  pRefine(cellIDsForPRefinements,1);
}

void Mesh::pRefine(const set<GlobalIndexType> &cellIDsForPRefinements, int pToAdd, bool repartitionAndRebuild)
{
  if (cellIDsForPRefinements.size() == 0) return;

  // refine any registered meshes
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++)
  {
    (*meshIt)->pRefine(cellIDsForPRefinements); // TODO: add pToAdd argument!
  }

  _gda->didPRefine(cellIDsForPRefinements, pToAdd);

  // let transformation function know about the refinement that just took place
  if (_meshTopology->transformationFunction().get())
  {
    _meshTopology->transformationFunction()->didPRefine(cellIDsForPRefinements);
  }

  if (repartitionAndRebuild)
  {
    this->repartitionAndRebuild();
  }
}

int Mesh::condensedRowSizeUpperBound()
{
  // includes multiplicity
  vector< Teuchos::RCP< ElementType > >::iterator elemTypeIt;
  int maxRowSize = 0;
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  for (PartitionIndexType partitionNumber=0; partitionNumber < partitionCount; partitionNumber++)
  {
    vector<ElementTypePtr> elementTypes = _gda->elementTypes(partitionNumber);
    for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)
    {
      ElementTypePtr elemTypePtr = *elemTypeIt;
      int numSides = elemTypePtr->cellTopoPtr->getSideCount();
      vector< int > fluxIDs = _bilinearForm->trialBoundaryIDs();
      vector< int >::iterator fluxIDIt;
      int numFluxDofs = 0;
      for (fluxIDIt = fluxIDs.begin(); fluxIDIt != fluxIDs.end(); fluxIDIt++)
      {
        int fluxID = *fluxIDIt;
        vector<int> sidesForFlux = elemTypePtr->trialOrderPtr->getSidesForVarID(fluxID);
        for (int sideOrdinal : sidesForFlux)
        {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(fluxID,sideOrdinal);
          numFluxDofs += numDofs;
        }
      }
      int maxPossible = numFluxDofs * 2 + numSides*fluxIDs.size();  // a side can be shared by 2 elements, and vertices can be shared
      maxRowSize = std::max(maxPossible, maxRowSize);
    }
  }
  return maxRowSize;
}

void Mesh::rebuildLookups()
{
  _gda->repartitionAndMigrate();
  _boundary.buildLookupTables();
}

int Mesh::rowSizeUpperBound()
{
  // includes multiplicity
  static const int MAX_SIZE_TO_PRESCRIBE = 100; // the below is a significant over-estimate.  Eventually, we want something more precise, that will analyze the BF to determine which variables actually talk to each other, and perhaps even provide a precise per-row count to the Epetra_CrsMatrix.  For now, we just cap the estimate.  (On construction, Epetra_CrsMatrix appears to be allocating the row size provided for every row, which is also wasteful.)
  vector< Teuchos::RCP< ElementType > >::iterator elemTypeIt;
  int maxRowSize = 0;
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  for (PartitionIndexType partitionNumber=0; partitionNumber < partitionCount; partitionNumber++)
  {
    vector<ElementTypePtr> elementTypes = _gda->elementTypes(partitionNumber);
    for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)
    {
      ElementTypePtr elemTypePtr = *elemTypeIt;
      int numSides = elemTypePtr->cellTopoPtr->getSideCount();
      vector< int > fluxIDs = _bilinearForm->trialBoundaryIDs();
      vector< int >::iterator fluxIDIt;
      int numFluxDofs = 0;
      for (fluxIDIt = fluxIDs.begin(); fluxIDIt != fluxIDs.end(); fluxIDIt++)
      {
        int fluxID = *fluxIDIt;
        vector<int> sidesForFlux = elemTypePtr->trialOrderPtr->getSidesForVarID(fluxID);
        for (int sideOrdinal : sidesForFlux)
        {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(fluxID,sideOrdinal);
          numFluxDofs += numDofs;
        }
      }
      int numFieldDofs = elemTypePtr->trialOrderPtr->totalDofs() - numFluxDofs;
      int maxPossible = numFluxDofs * 2 + numSides*fluxIDs.size() + numFieldDofs;  // a side can be shared by 2 elements, and vertices can be shared
      maxRowSize = std::max(maxPossible, maxRowSize);
    }
  }
  GlobalIndexType numGlobalDofs = this->numGlobalDofs();
  maxRowSize = std::min(maxRowSize, (int) numGlobalDofs);
  return std::min(maxRowSize, MAX_SIZE_TO_PRESCRIBE);
}

vector< ParametricCurvePtr > Mesh::parametricEdgesForCell(GlobalIndexType cellID, bool neglectCurves)
{
  MeshTopology* meshTopologyInstance = dynamic_cast<MeshTopology*>(_meshTopology.get());
  
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopologyInstance, std::invalid_argument, "Mesh::parametricEdgesForCell() called when _meshTopology is not an instance of MeshTopology--likely Mesh initialized with a pure MeshTopologyView, which does not support this.");
  
  return meshTopologyInstance->parametricEdgesForCell(cellID, neglectCurves);
}

void Mesh::setEdgeToCurveMap(const map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > &edgeToCurveMap)
{
  MeshTopology* meshTopologyInstance = dynamic_cast<MeshTopology*>(_meshTopology.get());
  
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopologyInstance, std::invalid_argument, "Mesh::setEdgeToCurveMap() called when _meshTopology is not an instance of MeshTopology--likely Mesh initialized with a pure MeshTopologyView, which does not support this.");
  
  MeshPtr thisPtr = Teuchos::rcp(this, false);
  map< pair<IndexType, IndexType>, ParametricCurvePtr > localMap(edgeToCurveMap.begin(),edgeToCurveMap.end());
  meshTopologyInstance->setEdgeToCurveMap(localMap, thisPtr);
}

void Mesh::setElementType(GlobalIndexType cellID, ElementTypePtr newType, bool sideUpgradeOnly)
{
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL)
  {
    cout << "setElementType only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "setElementType only supported for max rule.");
  }
  maxRule->setElementType(cellID, newType, sideUpgradeOnly);
}

void Mesh::setEnforceMultiBasisFluxContinuity( bool value )
{
  _enforceMBFluxContinuity = value;
}

void Mesh::setPartitionPolicy(  Teuchos::RCP< MeshPartitionPolicy > partitionPolicy )
{
  _gda->setPartitionPolicy(partitionPolicy);
}

void Mesh::setUsePatchBasis( bool value )
{
  // TODO: throw an exception if we've already been refined??
  _usePatchBasis = value;
}

bool Mesh::usePatchBasis()
{
  return _usePatchBasis;
}

MeshTopologyViewPtr Mesh::getTopology()
{
  return _meshTopology;
}

VarFactoryPtr Mesh::varFactory() const
{
  return _varFactory;
}

vector<unsigned> Mesh::vertexIndicesForCell(GlobalIndexType cellID)
{
  return _meshTopology->getCell(cellID)->vertices();
}

FieldContainer<double> Mesh::vertexCoordinates(GlobalIndexType vertexIndex)
{
  int spaceDim = _meshTopology->getDimension();
  FieldContainer<double> vertex(spaceDim);
  for (int d=0; d<spaceDim; d++)
  {
    vertex(d) = _meshTopology->getVertex(vertexIndex)[d];
  }
  return vertex;
}

vector< vector<double> > Mesh::verticesForCell(GlobalIndexType cellID)
{
  CellPtr cell = _meshTopology->getCell(cellID);
  vector<unsigned> vertexIndices = cell->vertices();
  int numVertices = vertexIndices.size();

  vector< vector<double> > vertices(numVertices);
  //vertices.resize(numVertices,dimension);
  for (unsigned vertexIndex = 0; vertexIndex < numVertices; vertexIndex++)
  {
    vertices[vertexIndex] = _meshTopology->getVertex(vertexIndices[vertexIndex]);
  }
  return vertices;
}

void Mesh::verticesForCell(FieldContainer<double>& vertices, GlobalIndexType cellID)
{
  _meshTopology->verticesForCell(vertices,cellID);
}

// global across all MPI nodes:
void Mesh::verticesForElementType(FieldContainer<double>& vertices, ElementTypePtr elemTypePtr)
{
  int spaceDim = _meshTopology->getDimension();
  int numVertices = elemTypePtr->cellTopoPtr->getNodeCount();
  int numCells = numElementsOfType(elemTypePtr);
  vertices.resize(numCells,numVertices,spaceDim);

  Teuchos::Array<int> dim; // for an individual cell
  dim.push_back(numVertices);
  dim.push_back(spaceDim);

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    int cellID = this->cellID(elemTypePtr,cellIndex);
    FieldContainer<double> cellVertices(dim,&vertices(cellIndex,0,0));
    this->verticesForCell(cellVertices, cellID);
  }
}

void Mesh::verticesForCells(FieldContainer<double>& vertices, vector<GlobalIndexType> &cellIDs)
{
  // all cells represented in cellIDs must have the same topology
  int spaceDim = _meshTopology->getDimension();
  int numCells = cellIDs.size();

  if (numCells == 0)
  {
    vertices.resize(0,0,0);
    return;
  }
  unsigned firstCellID = cellIDs[0];
  int numVertices = _meshTopology->getCell(firstCellID)->vertices().size();

  vertices.resize(numCells,numVertices,spaceDim);

  Teuchos::Array<int> dim; // for an individual cell
  dim.push_back(numVertices);
  dim.push_back(spaceDim);

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    int cellID = cellIDs[cellIndex];
    FieldContainer<double> cellVertices(dim,&vertices(cellIndex,0,0));
    this->verticesForCell(cellVertices, cellID);
  }
}

void Mesh::verticesForSide(FieldContainer<double>& vertices, GlobalIndexType cellID, int sideIndex)
{
  CellPtr cell = _meshTopology->getCell(cellID);
  int spaceDim = _meshTopology->getDimension();
  int sideDim = spaceDim - 1;
  unsigned sideEntityIndex = cell->entityIndex(sideDim, sideIndex);
  vector<unsigned> vertexIndices = _meshTopology->getEntityVertexIndices(sideDim, sideEntityIndex);

  int numVertices = vertexIndices.size();
  vertices.resize(numVertices,spaceDim);

  for (unsigned vertexIndex = 0; vertexIndex < numVertices; vertexIndex++)
  {
    for (int d=0; d<spaceDim; d++)
    {
      vertices(vertexIndex,d) = _meshTopology->getVertex(vertexIndex)[d];
    }
  }
}

void Mesh::writeMeshPartitionsToFile(const string & fileName)
{
  // TODO: rewrite this code to only talk to rank-local cells.

  ofstream myFile;
  myFile.open(fileName.c_str());
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  myFile << "numPartitions="<< partitionCount <<";"<<endl;

  int maxNumVertices=0;
  int maxNumElems=0;
  int spaceDim = 2;

  //initialize verts
  for (int i=0; i<partitionCount; i++)
  {
    set< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(i);
    for (int l=0; l<spaceDim; l++)
    {
      myFile << "verts{"<< i+1 <<","<< l+1 << "} = zeros(" << maxNumVertices << ","<< maxNumElems << ");"<< endl;
      for (set< GlobalIndexType >::iterator cellIt = cellsInPartition.begin(); cellIt != cellsInPartition.end(); cellIt++)
      {
        CellPtr cell = _meshTopology->getCell(*cellIt);
        int numVertices = cell->topology()->getVertexCount();
        FieldContainer<double> verts(numVertices,spaceDim); // gets resized inside verticesForCell
        verticesForCell(verts, *cellIt);  //verts(numVertsForCell,dim)
        maxNumVertices = std::max(maxNumVertices,verts.dimension(0));
        maxNumElems = std::max(maxNumElems,(int)cellsInPartition.size());
      }
    }
  }
  cout << "max number of elems = " << maxNumElems << endl;

  for (int i=0; i<partitionCount; i++)
  {
    set< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(i);
    for (int l=0; l<spaceDim; l++)
    {
      int j=0;
      for (set< GlobalIndexType >::iterator cellIt = cellsInPartition.begin(); cellIt != cellsInPartition.end(); cellIt++)
      {
        CellPtr cell = _meshTopology->getCell(*cellIt);
        int numVertices = cell->topology()->getVertexCount();
        FieldContainer<double> vertices(numVertices,spaceDim);
        verticesForCell(vertices, *cellIt);  //vertices(numVertsForCell,dim)

        // write vertex coordinates to file
        for (int k=0; k<numVertices; k++)
        {
          myFile << "verts{"<< i+1 <<","<< l+1 <<"}("<< k+1 <<","<< j+1 <<") = "<< vertices(k,l) << ";"<<endl; // verts{numPartitions,spaceDim}
        }
        j++;
      }

    }
  }
  myFile.close();
}

double Mesh::getCellMeasure(GlobalIndexType cellID)
{
  FieldContainer<double> physicalCellNodes = physicalCellNodesForCell(cellID);
  ElementPtr elem = getElement(cellID);
  Teuchos::RCP< ElementType > elemType = elem->elementType();
  CellTopoPtr cellTopo = elemType->cellTopoPtr;
  BasisCache basisCache(physicalCellNodes, cellTopo, 1);
  return basisCache.getCellMeasures()(0);
}

double Mesh::getCellXSize(GlobalIndexType cellID)
{
  ElementPtr elem = getElement(cellID);
  int spaceDim = 2; // assuming 2D
  int numSides = elem->numSides();
  TEUCHOS_TEST_FOR_EXCEPTION(numSides!=4, std::invalid_argument, "Anisotropic cell measures only defined for quads right now.");
  FieldContainer<double> vertices(numSides,spaceDim);
  verticesForCell(vertices, cellID);
  double xDist = vertices(1,0)-vertices(0,0);
  double yDist = vertices(1,1)-vertices(0,1);
  return sqrt(xDist*xDist + yDist*yDist);
}

double Mesh::getCellYSize(GlobalIndexType cellID)
{
  ElementPtr elem = getElement(cellID);
  int spaceDim = 2; // assuming 2D
  int numSides = elem->numSides();
  TEUCHOS_TEST_FOR_EXCEPTION(numSides!=4, std::invalid_argument, "Anisotropic cell measures only defined for quads right now.");
  FieldContainer<double> vertices(numSides,spaceDim);
  verticesForCell(vertices, cellID);
  double xDist = vertices(3,0)-vertices(0,0);
  double yDist = vertices(3,1)-vertices(0,1);
  return sqrt(xDist*xDist + yDist*yDist);
}

vector<double> Mesh::getCellOrientation(GlobalIndexType cellID)
{
  ElementPtr elem = getElement(cellID);
  int spaceDim = 2; // assuming 2D
  int numSides = elem->numSides();
  TEUCHOS_TEST_FOR_EXCEPTION(numSides!=4, std::invalid_argument, "Cell orientation only defined for quads right now.");
  FieldContainer<double> vertices(numSides,spaceDim);
  verticesForCell(vertices, cellID);
  double xDist = vertices(3,0)-vertices(0,0);
  double yDist = vertices(3,1)-vertices(0,1);
  vector<double> orientation;
  orientation.push_back(xDist);
  orientation.push_back(yDist);
  return orientation;
}


#ifdef HAVE_EPETRAEXT_HDF5
void Mesh::saveToHDF5(string filename)
{
  int commRank = Comm()->MyPID();

  if (commRank == 0)
  {
    set<IndexType> rootCellIndicesSet = getTopology()->getRootCellIndices();
    vector<IndexType> rootCellIndices(rootCellIndicesSet.begin(), rootCellIndicesSet.end());
    vector<IndexType> rootVertexIndices;
    vector<Camellia::CellTopologyKey> rootKeys;
    vector<double> rootVertices;
    for (int i=0; i < rootCellIndices.size(); i++)
    {
      CellPtr cell = getTopology()->getCell(rootCellIndices[i]);
      vector< unsigned > vertexIndices = cell->vertices();
      rootKeys.push_back(cell->topology()->getKey());
      rootVertexIndices.insert(rootVertexIndices.end(), vertexIndices.begin(), vertexIndices.end());
    }
    IndexType maxVertexIndex = *max_element(rootVertexIndices.begin(), rootVertexIndices.end());
    for (int i=0; i <= maxVertexIndex; i++)
    {
      vector< double > vertex = getTopology()->getVertex(i);
      rootVertices.insert(rootVertices.end(), vertex.begin(), vertex.end());
//      cout << "vertex " << i << ":\n";
//      Camellia::print("vertex coords", vertex);
    }
    map<int, int> trialOrderEnhancements = getDofOrderingFactory().getTrialOrderEnhancements();
    map<int, int> testOrderEnhancements = getDofOrderingFactory().getTestOrderEnhancements();
    vector<int> trialOrderEnhancementsVec;
    vector<int> testOrderEnhancementsVec;
    for (map<int,int>::iterator it=trialOrderEnhancements.begin(); it!=trialOrderEnhancements.end(); ++it)
    {
      trialOrderEnhancementsVec.push_back(it->first);
      trialOrderEnhancementsVec.push_back(it->second);
    }
    for (map<int,int>::iterator it=testOrderEnhancements.begin(); it!=testOrderEnhancements.end(); ++it)
    {
      testOrderEnhancementsVec.push_back(it->first);
      testOrderEnhancementsVec.push_back(it->second);
    }
    int vertexIndicesSize = rootVertexIndices.size();
    int topoKeysIntSize = rootKeys.size() * sizeof(Camellia::CellTopologyKey) / sizeof(int);
    int topoKeysSize = rootKeys.size();
    int verticesSize = rootVertices.size();
    int trialOrderEnhancementsSize = trialOrderEnhancementsVec.size();
    int testOrderEnhancementsSize = testOrderEnhancementsVec.size();


    FieldContainer<GlobalIndexType> partitions;
    bool partitionsSet = globalDofAssignment()->getPartitions(partitions);
    int numPartitions, maxPartitionSize;
    if (partitionsSet)
    {
      numPartitions = partitions.dimension(0);
      maxPartitionSize = partitions.dimension(1);
    }
    else
    {
      numPartitions = 0;
      maxPartitionSize = 0;
    }
    FieldContainer<int> partitionsCastToInt;
    if (partitions.size() > 0)
    {
      partitionsCastToInt.resize(numPartitions, maxPartitionSize);
      for (int i=0; i<numPartitions; i++)
      {
        for (int j=0; j<maxPartitionSize; j++)
        {
          partitionsCastToInt(i,j) = (int) partitions(i,j);
        }
      }
    }

    vector<int> initialH1Order = globalDofAssignment()->getInitialH1Order();

    Epetra_SerialComm Comm;
    EpetraExt::HDF5 hdf5(Comm);
    hdf5.Create(filename);
    hdf5.Write("Mesh", "vertexIndicesSize", vertexIndicesSize);
    hdf5.Write("Mesh", "topoKeysSize", topoKeysSize);
    hdf5.Write("Mesh", "verticesSize", verticesSize);
    hdf5.Write("Mesh", "trialOrderEnhancementsSize", trialOrderEnhancementsSize);
    hdf5.Write("Mesh", "testOrderEnhancementsSize", testOrderEnhancementsSize);
    hdf5.Write("Mesh", "dimension", getDimension());
    hdf5.Write("Mesh", "vertexIndices", H5T_NATIVE_INT, rootVertexIndices.size(), &rootVertexIndices[0]);
    hdf5.Write("Mesh", "topoKeys", H5T_NATIVE_INT, topoKeysIntSize, &rootKeys[0]);
    hdf5.Write("Mesh", "vertices", H5T_NATIVE_DOUBLE, rootVertices.size(), &rootVertices[0]);
    hdf5.Write("Mesh", "H1OrderSize", (int)initialH1Order.size());
    hdf5.Write("Mesh", "deltaP", globalDofAssignment()->getTestOrderEnrichment());
    hdf5.Write("Mesh", "numPartitions", numPartitions);
    hdf5.Write("Mesh", "maxPartitionSize", maxPartitionSize);
    if (numPartitions > 0)
    {
      hdf5.Write("Mesh", "partitions", H5T_NATIVE_INT, partitionsCastToInt.size(), &partitionsCastToInt[0]);
    }
    else
    {
      hdf5.Write("Mesh", "partitions", H5T_NATIVE_INT, partitionsCastToInt.size(), NULL);
    }
    if (meshUsesMaximumRule())
      hdf5.Write("Mesh", "GDARule", "max");
    else if(meshUsesMinimumRule())
      hdf5.Write("Mesh", "GDARule", "min");
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid GDA");
    hdf5.Write("Mesh", "trialOrderEnhancements", H5T_NATIVE_INT, trialOrderEnhancementsVec.size(), &trialOrderEnhancementsVec[0]);
    hdf5.Write("Mesh", "testOrderEnhancements", H5T_NATIVE_INT, testOrderEnhancementsVec.size(), &testOrderEnhancementsVec[0]);
    hdf5.Write("Mesh", "H1Order", H5T_NATIVE_INT, initialH1Order.size(), &initialH1Order[0]);
    _refinementHistory.saveToHDF5(hdf5);
    hdf5.Close();
  }
}
// end HAVE_EPETRAEXT_HDF5 include guard
#endif

Teuchos_CommPtr& Mesh::TeuchosComm()
{
  return _gda->getPartitionPolicy()->TeuchosComm();
}

MeshPtr Mesh::readMsh(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::readMsh() deprecated.  Use MeshFactory::readMesh() instead.\n";

  return MeshFactory::readMesh(filePath, bilinearForm, H1Order, pToAdd);
}

MeshPtr Mesh::readTriangle(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::readTriangle() deprecated.  Use MeshFactory::readTriangle() instead.\n";

  return MeshFactory::readTriangle(filePath, bilinearForm, H1Order, pToAdd);
}

MeshPtr Mesh::buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints,
                            int horizontalElements, int verticalElements,
                            TBFPtr<double> bilinearForm,
                            int H1Order, int pTest, bool triangulate, bool useConformingTraces,
                            map<int,int> trialOrderEnhancements,
                            map<int,int> testOrderEnhancements)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::buildQuadMesh() deprecated.  Use MeshFactory::buildQuadMesh() instead.\n";

  return MeshFactory::buildQuadMesh(quadBoundaryPoints, horizontalElements, verticalElements, bilinearForm, H1Order, pTest, triangulate, useConformingTraces, trialOrderEnhancements, testOrderEnhancements);
}

MeshPtr Mesh::buildQuadMeshHybrid(const FieldContainer<double> &quadBoundaryPoints,
                                  int horizontalElements, int verticalElements,
                                  TBFPtr<double> bilinearForm,
                                  int H1Order, int pTest, bool useConformingTraces)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::buildQuadMeshHybrid() deprecated.  Use MeshFactory::buildQuadMeshHybrid() instead.\n";

  return MeshFactory::buildQuadMeshHybrid(quadBoundaryPoints, horizontalElements, verticalElements, bilinearForm, \
                                          H1Order, pTest, useConformingTraces);
}

void Mesh::quadMeshCellIDs(FieldContainer<int> &cellIDs,
                           int horizontalElements, int verticalElements,
                           bool useTriangles)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::quadMeshCellIDs() deprecated.  Use MeshFactory::quadMeshCellIDs() instead.\n";

  MeshFactory::quadMeshCellIDs(cellIDs, horizontalElements, verticalElements, useTriangles);
}

namespace Camellia
{
template void Mesh::registerSolution(TSolutionPtr<double> solution);
template void Mesh::unregisterSolution(TSolutionPtr<double> solution);
}
