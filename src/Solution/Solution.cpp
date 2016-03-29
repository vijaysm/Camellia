
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
 *  Solution.cpp
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */


// Intrepid includes
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_Basis.hpp"

#include "Amesos_Klu.h"
#include "Amesos.h"
#include "Amesos_Utils.h"

// only use MUMPS when we have MPI
#ifdef HAVE_MPI
#ifdef HAVE_AMESOS_MUMPS
#include "Amesos_Mumps.h"
#endif
#endif

// Epetra includes
#ifdef HAVE_MPI
#include "Epetra_MpiDistributor.h"
#endif
#include "Epetra_SerialDistributor.h"
#include "Epetra_Time.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Epetra_SerialDenseSolver.h"
#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"
#include "Epetra_DataAccess.h"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "ml_epetra_utils.h"
//#include "ml_common.h"
#include "ml_epetra_preconditioner.h"

#include <stdlib.h>

#include "Solution.h"

// Camellia includes:
#include "BasisEvaluation.h"
#include "BasisCache.h"
#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"
#include "CondensedDofInterpreter.h"
#include "CubatureFactory.h"
#include "Function.h"
#include "IP.h"
#include "GlobalDofAssignment.h"
#include "LagrangeConstraints.h"
#include "Mesh.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "PreviousSolutionFunction.h"
#include "Projector.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "Solver.h"
#include "Var.h"

#include "AztecOO_ConditionNumber.h"

#ifdef HAVE_EPETRAEXT_HDF5
#include <EpetraExt_HDF5.h>
#include <Epetra_SerialComm.h>
#endif

using namespace Camellia;
using namespace Intrepid;

template <typename Scalar>
double TSolution<Scalar>::conditionNumberEstimate( Epetra_LinearProblem & problem )
{
  // estimates the 2-norm condition number
  AztecOOConditionNumber conditionEstimator;
  conditionEstimator.initialize(*problem.GetOperator());

  int maxIters = 40000;
  double tol = 1e-10;
  int status = conditionEstimator.computeConditionNumber(maxIters, tol);
  if (status!=0)
    cout << "status result from computeConditionNumber(): " << status << endl;
  double condest = conditionEstimator.getConditionNumber();

  return condest;
}

template <typename Scalar>
int TSolution<Scalar>::cubatureEnrichmentDegree() const
{
  return _cubatureEnrichmentDegree;
}

template <typename Scalar>
void TSolution<Scalar>::setCubatureEnrichmentDegree(int value)
{
  _cubatureEnrichmentDegree = value;
}

static const int MAX_BATCH_SIZE_IN_BYTES = 3*1024*1024; // 3 MB
static const int MIN_BATCH_SIZE_IN_CELLS = 1; // overrides the above, if it results in too-small batches

// copy constructor:
template <typename Scalar>
TSolution<Scalar>::TSolution(const TSolution<Scalar> &soln) : Narrator("Solution")
{
  _mesh = soln.mesh();
  _dofInterpreter = Teuchos::rcp( _mesh.get(), false ); // false: doesn't own memory
  _bf = soln.bf();
  _bc = soln.bc();
  _rhs = soln.rhs();
  _ip = soln.ip();
  _solutionForCellIDGlobal = soln.solutionForCellIDGlobal();
  _filter = soln.filter();
  _lagrangeConstraints = soln.lagrangeConstraints();
  _reportConditionNumber = false;
  _reportTimingResults = false;
  _writeMatrixToMatlabFile = false;
  _writeMatrixToMatrixMarketFile = false;
  _writeRHSToMatrixMarketFile = false;
  _cubatureEnrichmentDegree = soln.cubatureEnrichmentDegree();
  _zmcsAsLagrangeMultipliers = soln.getZMCsAsGlobalLagrange();
}

template <typename Scalar>
TSolution<Scalar>::TSolution(TBFPtr<Scalar> bf, MeshPtr mesh, TBCPtr<Scalar> bc, TRHSPtr<Scalar> rhs, TIPPtr<Scalar> ip) : Narrator("Solution")
{
  _bf = bf;
  _mesh = mesh;
  _dofInterpreter = Teuchos::rcp( _mesh.get(), false ); // false: doesn't own memory
  _bc = bc;
  _rhs = rhs;
  _ip = ip;
  _lagrangeConstraints = Teuchos::rcp( new LagrangeConstraints ); // empty

  initialize();
}

// Deprecated constructor, use the one which explicitly passes in BF
// Will eventually be removing BF reference from Mesh
template <typename Scalar>
TSolution<Scalar>::TSolution(MeshPtr mesh, TBCPtr<Scalar> bc, TRHSPtr<Scalar> rhs, TIPPtr<Scalar> ip) : Narrator("Solution")
{
  _bf = Teuchos::null;
  _mesh = mesh;
  _dofInterpreter = Teuchos::rcp( _mesh.get(), false ); // false: doesn't own memory
  _bc = bc;
  _rhs = rhs;
  _ip = ip;
  _lagrangeConstraints = Teuchos::rcp( new LagrangeConstraints ); // empty

  initialize();
}

template <typename Scalar>
void TSolution<Scalar>::clear()
{
  // clears all solution values.  Leaves everything else intact.
  _solutionForCellIDGlobal.clear();
}

template <typename Scalar>
void TSolution<Scalar>::initialize()
{
  // clear the data structure in case it already stores some stuff
  _solutionForCellIDGlobal.clear();

  _writeMatrixToMatlabFile = false;
  _writeMatrixToMatrixMarketFile = false;
  _writeRHSToMatrixMarketFile = false;
  _residualsComputed = false;
  _energyErrorComputed = false;
  _rankLocalEnergyErrorComputed = false;
  _reportConditionNumber = false;
  _reportTimingResults = false;
  _globalSystemConditionEstimate = -1;
  _cubatureEnrichmentDegree = 0;
  
  _zmcsAsLagrangeMultipliers = true; // default -- when false, it's user's / Solver's responsibility to enforce ZMCs
  _zmcsAsRankOneUpdate = false; // I believe this works, but it's slow!
  _zmcRho = -1; // default value: stabilization parameter for zero-mean constraints
}

template <typename Scalar>
void TSolution<Scalar>::addSolution(Teuchos::RCP< TSolution<Scalar> > otherSoln, double weight, bool allowEmptyCells, bool replaceBoundaryTerms)
{
  // In many situations, we can't legitimately add two condensed solution _lhsVectors together and back out the other (field) dofs.
  // E.g., consider a nonlinear problem in which the bilinear form (and therefore stiffness matrix) depends on background data.
  // Even a linear problem with two solutions with different RHS data would require us to accumulate the local load vectors.
  // For this reason, we don't attempt to add the two _lhsVectors together.  Instead, we add their respective cell-local
  // (expanded, basically) coefficients together, and then glean the condensed representation from that using the private
  // setGlobalSolutionFromCellLocalCoefficients() method.

  set<GlobalIndexType> myCellIDs = _mesh->cellIDsInPartition();

  // in case otherSoln has a distinct mesh partitioning, import data for this's cells that is off-rank in otherSoln
  otherSoln->importSolutionForOffRankCells(myCellIDs);

  for (set<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    Intrepid::FieldContainer<Scalar> myCoefficients;
    if (_solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end())
    {
      myCoefficients = _solutionForCellIDGlobal[cellID];
    }
    else
    {
      myCoefficients.resize(_mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
    }

    bool warnAboutOffRank = false;
    Intrepid::FieldContainer<Scalar> otherCoefficients = otherSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);

    SerialDenseWrapper::addFCs(myCoefficients, otherCoefficients, weight);

    if (replaceBoundaryTerms)
    {
      // then copy the flux/field terms from otherCoefficients, without weighting with weight (used to weight with weight; changed 2/5/15)
      DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
      set<int> traceDofIndices = trialOrder->getTraceDofIndices();
      for (set<int>::iterator traceDofIndexIt = traceDofIndices.begin(); traceDofIndexIt != traceDofIndices.end(); traceDofIndexIt++)
      {
        int traceDofIndex = *traceDofIndexIt;
        myCoefficients[traceDofIndex] = otherCoefficients[traceDofIndex];
      }
    }
    _solutionForCellIDGlobal[cellID] = myCoefficients;
  }

  setGlobalSolutionFromCellLocalCoefficients();

  clearComputedResiduals();

  return;
}

template <typename Scalar>
void TSolution<Scalar>::addSolution(Teuchos::RCP< TSolution<Scalar> > otherSoln, double weight, set<int> varsToAdd, bool allowEmptyCells)
{
  // In many situations, we can't legitimately add two condensed solution _lhsVectors together and back out the other (field) dofs.
  // E.g., consider a nonlinear problem in which the bilinear form (and therefore stiffness matrix) depends on background data.
  // Even a linear problem with two solutions with different RHS data would require us to accumulate the local load vectors.
  // For this reason, we don't attempt to add the two _lhsVectors together.  Instead, we add their respective cell-local
  // (expanded, basically) coefficients together, and then glean the condensed representation from that using the private
  // setGlobalSolutionFromCellLocalCoefficients() method.

  set<GlobalIndexType> myCellIDs = _mesh->cellIDsInPartition();

  for (set<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    Intrepid::FieldContainer<Scalar> myCoefficients;
    if (_solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end())
    {
      myCoefficients = _solutionForCellIDGlobal[cellID];
    }
    else
    {
      myCoefficients.resize(_mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
    }

    Intrepid::FieldContainer<Scalar> otherCoefficients = otherSoln->allCoefficientsForCellID(cellID);

    DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
    for (set<int>::iterator varIDIt = varsToAdd.begin(); varIDIt != varsToAdd.end(); varIDIt++)
    {
      int varID = *varIDIt;
      const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(varID);
      for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
      {
        int sideOrdinal = *sideIt;
        vector<int> dofIndices = trialOrder->getDofIndices(varID, sideOrdinal);
        for (vector<int>::iterator dofIndexIt = dofIndices.begin(); dofIndexIt != dofIndices.end(); dofIndexIt++)
        {
          int dofIndex = *dofIndexIt;
          myCoefficients[dofIndex] += weight * otherCoefficients[dofIndex];
        }
      }
    }

    _solutionForCellIDGlobal[cellID] = myCoefficients;
  }

  setGlobalSolutionFromCellLocalCoefficients();

  clearComputedResiduals();

  // old implementation below:
  /*  set<GlobalIndexType> globalIndicesForVars = _mesh->globalDofAssignment()->partitionOwnedIndicesForVariables(varsToAdd);
    Epetra_Map partMap = getPartitionMap();

    // add the global solution vectors together.
    if (otherSoln->getLHSVector().get() != NULL) {
      if (_lhsVector.get() == NULL) {
        // then we treat this solution as 0
        _lhsVector = Teuchos::rcp(new Epetra_FEVector(partMap,1,true));
        _lhsVector->PutScalar(0); // unclear whether this is redundant with constructor or not
      }
      for (set<GlobalIndexType>::iterator gidIt = globalIndicesForVars.begin(); gidIt != globalIndicesForVars.end(); gidIt++) {
        int lid = partMap.LID((GlobalIndexTypeToCast)*gidIt);
        (*_lhsVector)[0][lid] += (*otherSoln->getLHSVector())[0][lid] * weight;
      }
      // now, interpret the global data
      importSolution();

      clearComputedResiduals();
    }*/
}

template <typename Scalar>
void TSolution<Scalar>::addReplaceSolution(Teuchos::RCP< TSolution<Scalar> > otherSoln, double weight, set<int> varsToAdd, set<int> varsToReplace, bool allowEmptyCells)
{
  // In many situations, we can't legitimately add two condensed solution _lhsVectors together and back out the other (field) dofs.
  // E.g., consider a nonlinear problem in which the bilinear form (and therefore stiffness matrix) depends on background data.
  // Even a linear problem with two solutions with different RHS data would require us to accumulate the local load vectors.
  // For this reason, we don't attempt to add the two _lhsVectors together.  Instead, we add their respective cell-local
  // (expanded, basically) coefficients together, and then glean the condensed representation from that using the private
  // setGlobalSolutionFromCellLocalCoefficients() method.

  set<GlobalIndexType> myCellIDs = _mesh->cellIDsInPartition();

  for (set<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    Intrepid::FieldContainer<Scalar> myCoefficients;
    if (_solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end())
    {
      myCoefficients = _solutionForCellIDGlobal[cellID];
    }
    else
    {
      myCoefficients.resize(_mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
    }

    Intrepid::FieldContainer<Scalar> otherCoefficients = otherSoln->allCoefficientsForCellID(cellID);

    DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
    for (set<int>::iterator varIDIt = varsToAdd.begin(); varIDIt != varsToAdd.end(); varIDIt++)
    {
      int varID = *varIDIt;
      const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(varID);
      for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
      {
        int sideOrdinal = *sideIt;
        vector<int> dofIndices = trialOrder->getDofIndices(varID, sideOrdinal);
        for (vector<int>::iterator dofIndexIt = dofIndices.begin(); dofIndexIt != dofIndices.end(); dofIndexIt++)
        {
          int dofIndex = *dofIndexIt;
          myCoefficients[dofIndex] += weight * otherCoefficients[dofIndex];
        }
      }
    }
    for (set<int>::iterator varIDIt = varsToReplace.begin(); varIDIt != varsToReplace.end(); varIDIt++)
    {
      int varID = *varIDIt;
      const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(varID);
      for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
      {
        int sideOrdinal = *sideIt;
        vector<int> dofIndices = trialOrder->getDofIndices(varID, sideOrdinal);
        for (vector<int>::iterator dofIndexIt = dofIndices.begin(); dofIndexIt != dofIndices.end(); dofIndexIt++)
        {
          int dofIndex = *dofIndexIt;
          myCoefficients[dofIndex] = otherCoefficients[dofIndex];
        }
      }
    }

    _solutionForCellIDGlobal[cellID] = myCoefficients;
  }

  setGlobalSolutionFromCellLocalCoefficients();

  clearComputedResiduals();

  // old implementation below:
  /*  set<GlobalIndexType> globalIndicesForVars = _mesh->globalDofAssignment()->partitionOwnedIndicesForVariables(varsToAdd);
    Epetra_Map partMap = getPartitionMap();

    // add the global solution vectors together.
    if (otherSoln->getLHSVector().get() != NULL) {
      if (_lhsVector.get() == NULL) {
        // then we treat this solution as 0
        _lhsVector = Teuchos::rcp(new Epetra_FEVector(partMap,1,true));
        _lhsVector->PutScalar(0); // unclear whether this is redundant with constructor or not
      }
      for (set<GlobalIndexType>::iterator gidIt = globalIndicesForVars.begin(); gidIt != globalIndicesForVars.end(); gidIt++) {
        int lid = partMap.LID((GlobalIndexTypeToCast)*gidIt);
        (*_lhsVector)[0][lid] += (*otherSoln->getLHSVector())[0][lid] * weight;
      }
      // now, interpret the global data
      importSolution();

      clearComputedResiduals();
    }*/
}

template <typename Scalar>
bool TSolution<Scalar>::cellHasCoefficientsAssigned(GlobalIndexType cellID)
{
  return _solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end();
}

template <typename Scalar>
void TSolution<Scalar>::applyDGJumpTerms()
{
  // accumulate any inter-element DG terms
  // we integrate over the interior faces.
  
  /*
   We do the integration elementwise; on each face of each element, we decide whether the
   element "owns" the face, so that the term is only integrated once, and only on the side
   with finer quadrature, in the case of a locally refined mesh.
   */
  
  Epetra_CommPtr Comm = _mesh->Comm();
  int numProcs = Comm->NumProc();
  
  int indexBase = 0;
  
  Epetra_Map timeMap(numProcs,indexBase,*Comm);
  
  Epetra_Time timer(*Comm);
  
  Epetra_FECrsMatrix* stiffnessMatrix = dynamic_cast<Epetra_FECrsMatrix*>(_globalStiffMatrix.get());
  
  MeshTopologyViewPtr meshTopo = _mesh->getTopology();
  set<GlobalIndexType> activeCellIDs = _mesh->getActiveCellIDs();
  set<GlobalIndexType> myCellIDs = _mesh->cellIDsInPartition();
  int sideDim = meshTopo->getDimension() - 1;
  
  FieldContainer<double> emptyRefPointsVolume(0,meshTopo->getDimension()); // (P,D)
  FieldContainer<double> emptyRefPointsSide(0,sideDim); // (P,D)
  FieldContainer<double> emptyCubWeights(1,0); // (C,P)
  
  map<pair<CellTopologyKey,int>, FieldContainer<double>> cubPointsForSideTopo;
  map<pair<CellTopologyKey,int>, FieldContainer<double>> cubWeightsForSideTopo;
  
  CubatureFactory cubFactory;
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForVolumeTopo; // used for "my" cells
  map<CellTopologyKey,BasisCachePtr> basisCacheForNeighborVolumeTopo; // used for neighbor cells
  map<pair<int,CellTopologyKey>,BasisCachePtr> basisCacheForSideOnVolumeTopo;
  map<pair<int,CellTopologyKey>,BasisCachePtr> basisCacheForSideOnNeighborVolumeTopo; // these can have permuted cubature points (i.e. we need to set them every time, so we can't share with basisCacheForSideOnVolumeTopo, which tries to avoid this)
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForReferenceCellTopo;

  const vector< TBilinearTerm<Scalar> >* bilinearTerms = &_mesh->bilinearForm()->getJumpTerms();
  
  for (GlobalIndexType cellID : myCellIDs)
  {
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    ElementTypePtr elemType = _mesh->getElementType(cellID);
    
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
    BasisCachePtr cellBasisCacheVolume;
    if (basisCacheForVolumeTopo.find(cellTopo->getKey()) == basisCacheForVolumeTopo.end())
    {
      basisCacheForVolumeTopo[cellTopo->getKey()] = Teuchos::rcp( new BasisCache(physicalCellNodes, cellTopo,
                                                                                 emptyRefPointsVolume, emptyCubWeights) );
    }
    
    cellBasisCacheVolume = basisCacheForVolumeTopo[cellTopo->getKey()];
    cellBasisCacheVolume->setPhysicalCellNodes(physicalCellNodes, {cellID}, false);
    
    cellBasisCacheVolume->setCellIDs({cellID});
    cellBasisCacheVolume->setMesh(_mesh);
    
    // now, determine the max cubature degree for the "me vs. me" terms
    int myMaxCubatureDegree = 0;
    
    DofOrderingPtr trialOrder = elemType->trialOrderPtr;
    DofOrderingPtr testOrder = elemType->testOrderPtr;
    
    int sideCount = cell->getSideCount();

    for (TBilinearTerm<Scalar> bilinearTerm : *bilinearTerms)
    {
      TLinearTermPtr<Scalar> trialTerm = bilinearTerm.first;
      TLinearTermPtr<Scalar> testTerm = bilinearTerm.second;
      
      // in what follows, we do assume that the test and trial terms are both scalar-valued
      // TODO: relax this assumption
      TEUCHOS_TEST_FOR_EXCEPTION(trialTerm->rank() > 0, std::invalid_argument, "Only scalar-valued DG jump terms are supported right now");
      TEUCHOS_TEST_FOR_EXCEPTION(testTerm->rank() > 0, std::invalid_argument, "Only scalar-valued DG jump terms are supported right now");
      
      const std::set<int> * trialVarIDs = &trialTerm->varIDs();
      const std::set<int> * testVarIDs = &testTerm->varIDs();
      
      for (int trialVarID : *trialVarIDs)
      {
        int trialDegree = trialOrder->getBasis(trialVarID)->getDegree();
        
        vector<GlobalIndexType> myGlobalDofs_trial_native = _mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(cellID, trialVarID);
        
        for (int testVarID : *testVarIDs)
        {
          int testDegree = testOrder->getBasis(testVarID)->getDegree();
          int myMaxCubatureDegree = max(myMaxCubatureDegree,testDegree+trialDegree);
        }
      }
    }
    
    for (TBilinearTerm<Scalar> bilinearTerm : *bilinearTerms)
    {
      TLinearTermPtr<Scalar> trialTerm = bilinearTerm.first;
      TLinearTermPtr<Scalar> testTerm = bilinearTerm.second;
      
      const std::set<int> * trialVarIDs = &trialTerm->varIDs();
      const std::set<int> * testVarIDs = &testTerm->varIDs();
      
      for (int trialVarID : *trialVarIDs)
      {
        BasisPtr trialBasis = trialOrder->getBasis(trialVarID);
        
        vector<GlobalIndexType> myGlobalDofs_trial_native = _mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(cellID, trialVarID);
        
        for (int testVarID : *testVarIDs)
        {
          BasisPtr testBasis = testOrder->getBasis(testVarID);
          
          vector<GlobalIndexType> myGlobalDofs_test_native = _mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(cellID, testVarID);
          
          for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
          {
            // we'll filter this down to match only the side dofs, so we need to do this inside the sideOrdinal loop:
            vector<GlobalIndexTypeToCast> myGlobalDofs_trial(myGlobalDofs_trial_native.begin(),myGlobalDofs_trial_native.end());
            vector<GlobalIndexTypeToCast> myGlobalDofs_test(myGlobalDofs_test_native.begin(),myGlobalDofs_test_native.end());
            
            pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
            GlobalIndexType neighborCellID = neighborInfo.first;
            unsigned mySideOrdinalInNeighbor = neighborInfo.second;
            if (activeCellIDs.find(neighborCellID) == activeCellIDs.end())
            {
              // no active neigbor on this side: either this is not an interior face (neighborCellID == -1),
              // or the neighbor is refined and therefore inactive.  If the latter, then the neighbor's
              // descendants will collectively "own" this side.
              continue;
            }
            
            // Finally, we need to check whether the neighbor is a "peer" in terms of h-refinements.
            // If so, we use the cellID to break the tie of ownership; lower cellID owns the face.
            CellPtr neighbor = meshTopo->getCell(neighborInfo.first);
            pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
            bool neighborIsPeer = neighborNeighborInfo.first == cell->cellIndex();
            if (neighborIsPeer && (cellID > neighborCellID))
            {
              // neighbor wins the tie-breaker
              continue;
            }
            
            // if we get here, we own the face and should compute its contribution.
            // determine global dof indices:
            vector<GlobalIndexType> neighborGlobalDofs_trial_native = _mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(neighborCellID, trialVarID);
            vector<GlobalIndexTypeToCast> neighborGlobalDofs_trial(neighborGlobalDofs_trial_native.begin(),neighborGlobalDofs_trial_native.end());
            
            vector<GlobalIndexType> neighborGlobalDofs_test_native = _mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(neighborCellID, testVarID);
            vector<GlobalIndexTypeToCast> neighborGlobalDofs_test(neighborGlobalDofs_test_native.begin(),neighborGlobalDofs_test_native.end());
            
            // figure out what the cubature degree should be
            DofOrderingPtr neighborTrialOrder = _mesh->getElementType(neighborCellID)->trialOrderPtr;
            DofOrderingPtr neighborTestOrder = _mesh->getElementType(neighborCellID)->testOrderPtr;
            BasisPtr trialBasisNeighbor = neighborTrialOrder->getBasis(trialVarID);
            BasisPtr testBasisNeighbor = neighborTestOrder->getBasis(testVarID);
            int neighborCubatureDegree = trialBasisNeighbor->getDegree() + testBasisNeighbor->getDegree();
            
            int cubaturePolyOrder = max(myMaxCubatureDegree, neighborCubatureDegree);
            
            // set up side basis cache
            CellTopoPtr mySideTopo = cellTopo->getSide(sideOrdinal); // for non-peers, this is the descendant cell topo
            
            pair<int,CellTopologyKey> sideCacheKey{sideOrdinal,cellTopo->getKey()};
            if (basisCacheForSideOnVolumeTopo.find(sideCacheKey) == basisCacheForSideOnVolumeTopo.end())
            {
              basisCacheForSideOnVolumeTopo[sideCacheKey] = Teuchos::rcp( new BasisCache(sideOrdinal, cellBasisCacheVolume,
                                                                                         emptyRefPointsSide, emptyCubWeights, -1));
            }
            BasisCachePtr cellBasisCacheSide = basisCacheForSideOnVolumeTopo[sideCacheKey];
            
            pair<CellTopologyKey,int> cubKey{mySideTopo->getKey(),cubaturePolyOrder};
            if (cubWeightsForSideTopo.find(cubKey) == cubWeightsForSideTopo.end())
            {
              int cubDegree = cubKey.second;
              if (sideDim > 0)
              {
                Teuchos::RCP<Cubature<double> > sideCub;
                if (cubDegree >= 0)
                  sideCub = cubFactory.create(mySideTopo, cubDegree);
                
                int numCubPointsSide;
                
                if (sideCub != Teuchos::null)
                  numCubPointsSide = sideCub->getNumPoints();
                else
                  numCubPointsSide = 0;
                
                FieldContainer<double> cubPoints(numCubPointsSide, sideDim); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
                FieldContainer<double> cubWeights(numCubPointsSide);
                if (numCubPointsSide > 0)
                  sideCub->getCubature(cubPoints, cubWeights);
                cubPointsForSideTopo[cubKey] = cubPoints;
                cubWeightsForSideTopo[cubKey] = cubWeights;
              }
              else
              {
                int numCubPointsSide = 1;
                FieldContainer<double> cubPoints(numCubPointsSide, 1); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
                FieldContainer<double> cubWeights(numCubPointsSide);
                
                cubPoints.initialize(0.0);
                cubWeights.initialize(1.0);
                cubPointsForSideTopo[cubKey] = cubPoints;
                cubWeightsForSideTopo[cubKey] = cubWeights;
              }
            }
            if (cellBasisCacheSide->cubatureDegree() != cubaturePolyOrder)
            {
              cellBasisCacheSide->setRefCellPoints(cubPointsForSideTopo[cubKey], cubWeightsForSideTopo[cubKey], cubaturePolyOrder, false);
            }
            cellBasisCacheSide->setPhysicalCellNodes(cellBasisCacheVolume->getPhysicalCellNodes(), {cellID}, false);
            
            int numCells = 1;
            int numTrialFields = trialBasis->getCardinality();
            int numPoints = cellBasisCacheSide->getRefCellPoints().dimension(0);
            Intrepid::FieldContainer<double> trial_values(numCells, numTrialFields, numPoints);
            trialTerm->values(trial_values, trialVarID, trialBasis, cellBasisCacheSide);
            
            // we integrate against the jump term beta * [v n]
            // this is simply the sum of beta * (v n) from cell and neighbor's point of view
            
            int numTestFields = testBasis->getCardinality();
            Intrepid::FieldContainer<double> test_values(numCells, numTestFields, numPoints);
            testTerm->values(test_values, testVarID, testBasis, cellBasisCacheSide);
            
            // filter to include only those members of uBasis that have support on the side
            // (it might be nice to have LinearTerm::values() support this directly, via a basisDofOrdinal container argument...)
            auto filterSideBasisValues = [] (BasisPtr basis, int sideOrdinal,
                                             FieldContainer<double> &values) -> void
            {
              set<int> mySideBasisDofOrdinals = basis->dofOrdinalsForSide(sideOrdinal);
              int numFilteredFields = mySideBasisDofOrdinals.size();
              int numCells = values.dimension(0);
              int numPoints = values.dimension(2);
              FieldContainer<double> filteredValues(numCells,numFilteredFields,numPoints);
              for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
              {
                int filteredDofOrdinal = 0;
                for (int basisDofOrdinal : mySideBasisDofOrdinals)
                {
                  for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
                  {
                    filteredValues(cellOrdinal,filteredDofOrdinal,pointOrdinal) = values(cellOrdinal,basisDofOrdinal,pointOrdinal);
                  }
                  filteredDofOrdinal++;
                }
              }
              values = filteredValues;
            };
            
            auto filterGlobalDofOrdinals = [] (BasisPtr basis, int sideOrdinal,
                                               vector<GlobalIndexTypeToCast> &globalDofOrdinals) -> void
            {
              set<int> mySideBasisDofOrdinals = basis->dofOrdinalsForSide(sideOrdinal);
              vector<GlobalIndexTypeToCast> filteredGlobalDofOrdinals;
              for (int basisDofOrdinal : mySideBasisDofOrdinals)
              {
                filteredGlobalDofOrdinals.push_back(globalDofOrdinals[basisDofOrdinal]);
              }
              globalDofOrdinals = filteredGlobalDofOrdinals;
            };
            
            filterSideBasisValues(trialBasis,sideOrdinal,trial_values);
            filterSideBasisValues(testBasis,sideOrdinal,test_values);
            filterGlobalDofOrdinals(trialBasis,sideOrdinal,myGlobalDofs_trial);
            filterGlobalDofOrdinals(testBasis,sideOrdinal,myGlobalDofs_test);
            numTrialFields = myGlobalDofs_trial.size();
            numTestFields = myGlobalDofs_test.size();
            
            // Now the geometrically challenging bit: we need to line up the physical points in
            // the cellBasisCacheSide with those in a BasisCache for the neighbor cell
            
            CellTopoPtr neighborTopo = neighbor->topology();
            CellTopoPtr sideTopo = neighborTopo->getSide(mySideOrdinalInNeighbor); // for non-peers, this is my ancestor's cell topo
            int nodeCount = sideTopo->getNodeCount();
            
            unsigned permutationFromMeToNeighbor;
            Intrepid::FieldContainer<double> myRefPoints = cellBasisCacheSide->getRefCellPoints();
            
            if (!neighborIsPeer) // then we have some refinements relative to neighbor
            {
              /*******   Map my ref points to my ancestor ******/
              pair<GlobalIndexType,unsigned> ancestorInfo = neighbor->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
              GlobalIndexType ancestorCellIndex = ancestorInfo.first;
              unsigned ancestorSideOrdinal = ancestorInfo.second;
              
              RefinementBranch refinementBranch = cell->refinementBranchForSide(sideOrdinal, meshTopo);
              RefinementBranch sideRefinementBranch = RefinementPattern::sideRefinementBranch(refinementBranch, ancestorSideOrdinal);
              FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(sideRefinementBranch);
              
              cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
              BasisCachePtr ancestralBasisCache = Teuchos::rcp(new BasisCache(cellNodes,sideTopo,cubaturePolyOrder,false)); // false: don't create side cache too
              
              ancestralBasisCache->setRefCellPoints(myRefPoints, emptyCubWeights, cubaturePolyOrder, true);
              
              // now, the "physical" points in ancestral cache are the ones we want
              myRefPoints = ancestralBasisCache->getPhysicalCubaturePoints();
              myRefPoints.resize(myRefPoints.dimension(1),myRefPoints.dimension(2)); // strip cell dimension
              
              /*******  Determine ancestor's permutation of the side relative to neighbor ******/
              CellPtr ancestor = meshTopo->getCell(ancestorCellIndex);
              vector<IndexType> ancestorSideNodes, neighborSideNodes; // this will list the indices as seen by MeshTopology
              
              CellTopoPtr sideTopo = ancestor->topology()->getSide(ancestorSideOrdinal);
              nodeCount = sideTopo->getNodeCount();
              
              for (int node=0; node<nodeCount; node++)
              {
                int nodeInCell = cellTopo->getNodeMap(sideDim, sideOrdinal, node);
                ancestorSideNodes.push_back(ancestor->vertices()[nodeInCell]);
                int nodeInNeighborCell = neighborTopo->getNodeMap(sideDim, mySideOrdinalInNeighbor, node);
                neighborSideNodes.push_back(neighbor->vertices()[nodeInNeighborCell]);
              }
              // now, we want to know what permutation of the side topology takes us from my order to neighbor's
              // TODO: make sure I'm not going the wrong direction here; it's easy to get confused.
              permutationFromMeToNeighbor = CamelliaCellTools::permutationMatchingOrder(sideTopo, ancestorSideNodes, neighborSideNodes);
            }
            else
            {
              nodeCount = cellTopo->getSide(sideOrdinal)->getNodeCount();
              
              vector<IndexType> mySideNodes, neighborSideNodes; // this will list the indices as seen by MeshTopology
              for (int node=0; node<nodeCount; node++)
              {
                int nodeInCell = cellTopo->getNodeMap(sideDim, sideOrdinal, node);
                mySideNodes.push_back(cell->vertices()[nodeInCell]);
                int nodeInNeighborCell = neighborTopo->getNodeMap(sideDim, mySideOrdinalInNeighbor, node);
                neighborSideNodes.push_back(neighbor->vertices()[nodeInNeighborCell]);
              }
              // now, we want to know what permutation of the side topology takes us from my order to neighbor's
              // TODO: make sure I'm not going the wrong direction here; it's easy to get confused.
              permutationFromMeToNeighbor = CamelliaCellTools::permutationMatchingOrder(sideTopo, mySideNodes, neighborSideNodes);
            }
            
            Intrepid::FieldContainer<double> permutedRefNodes(nodeCount,sideDim);
            CamelliaCellTools::refCellNodesForTopology(permutedRefNodes, sideTopo, permutationFromMeToNeighbor);
            permutedRefNodes.resize(1,nodeCount,sideDim); // add cell dimension to make this a "physical" node container
            if (basisCacheForReferenceCellTopo.find(sideTopo->getKey()) == basisCacheForReferenceCellTopo.end())
            {
              basisCacheForReferenceCellTopo[sideTopo->getKey()] = BasisCache::basisCacheForReferenceCell(sideTopo, -1);
            }
            BasisCachePtr referenceBasisCache = basisCacheForReferenceCellTopo[sideTopo->getKey()];
            referenceBasisCache->setRefCellPoints(myRefPoints,emptyCubWeights,cubaturePolyOrder,false);
            std::vector<GlobalIndexType> cellIDs = {0}; // unused
            referenceBasisCache->setPhysicalCellNodes(permutedRefNodes, cellIDs, false);
            // now, the "physical" points are the ones we should use as ref points for the neighbor
            Intrepid::FieldContainer<double> neighborRefCellPoints = referenceBasisCache->getPhysicalCubaturePoints();
            neighborRefCellPoints.resize(numPoints,sideDim); // strip cell dimension to convert to a "reference" point container
            
            FieldContainer<double> neighborCellNodes = _mesh->physicalCellNodesForCell(neighborCellID);
            if (basisCacheForNeighborVolumeTopo.find(neighborTopo->getKey()) == basisCacheForNeighborVolumeTopo.end())
            {
              basisCacheForNeighborVolumeTopo[neighborTopo->getKey()] = Teuchos::rcp( new BasisCache(neighborCellNodes, neighborTopo,
                                                                                                     emptyRefPointsVolume, emptyCubWeights) );
            }
            BasisCachePtr neighborVolumeCache = basisCacheForNeighborVolumeTopo[neighborTopo->getKey()];
            neighborVolumeCache->setPhysicalCellNodes(neighborCellNodes, {neighborCellID}, false);
            
            pair<int,CellTopologyKey> neighborSideCacheKey{mySideOrdinalInNeighbor,neighborTopo->getKey()};
            if (basisCacheForSideOnNeighborVolumeTopo.find(neighborSideCacheKey) == basisCacheForSideOnNeighborVolumeTopo.end())
            {
              basisCacheForSideOnNeighborVolumeTopo[neighborSideCacheKey]
              = Teuchos::rcp( new BasisCache(mySideOrdinalInNeighbor, neighborVolumeCache, emptyRefPointsSide, emptyCubWeights, -1));
            }
            BasisCachePtr neighborSideCache = basisCacheForSideOnNeighborVolumeTopo[neighborSideCacheKey];
            neighborSideCache->setRefCellPoints(neighborRefCellPoints, emptyCubWeights, cubaturePolyOrder, false);
            neighborSideCache->setPhysicalCellNodes(neighborCellNodes, {neighborCellID}, false);
            {
              // Sanity check that the physical points agree:
              double tol = 1e-15;
              Intrepid::FieldContainer<double> myPhysicalPoints = cellBasisCacheSide->getPhysicalCubaturePoints();
              Intrepid::FieldContainer<double> neighborPhysicalPoints = neighborSideCache->getPhysicalCubaturePoints();
              
              bool pointsMatch = (myPhysicalPoints.size() == neighborPhysicalPoints.size()); // true unless we find a point that doesn't match
              if (pointsMatch)
              {
                for (int i=0; i<myPhysicalPoints.size(); i++)
                {
                  double diff = abs(myPhysicalPoints[i]-neighborPhysicalPoints[i]);
                  if (diff > tol)
                  {
                    pointsMatch = false;
                    break;
                  }
                }
              }
              
              if (!pointsMatch)
              {
                cout << "ERROR: pointsMatch is false.\n";
                cout << "myPhysicalPoints:\n" << myPhysicalPoints;
                cout << "neighborPhysicalPoints:\n" << neighborPhysicalPoints;
              }
            }
            
            int numNeighborTrialFields = trialBasisNeighbor->getCardinality();
            int numNeighborTestFields = testBasisNeighbor->getCardinality();
            
            Intrepid::FieldContainer<double> neighbor_trial_values(numCells, numNeighborTrialFields, numPoints);
            trialTerm->values(neighbor_trial_values, trialVarID, trialBasisNeighbor, neighborSideCache);
            
            Intrepid::FieldContainer<double> neighbor_test_values(numCells, numNeighborTestFields, numPoints);
            testTerm->values(neighbor_test_values, testVarID, testBasisNeighbor, neighborSideCache);
            
            int neighborSideOrdinal = neighborSideCache->getSideIndex();
            filterSideBasisValues(trialBasisNeighbor,neighborSideOrdinal,neighbor_trial_values);
            filterSideBasisValues(testBasisNeighbor,neighborSideOrdinal,neighbor_test_values);
            filterGlobalDofOrdinals(trialBasisNeighbor,neighborSideOrdinal,neighborGlobalDofs_trial);
            filterGlobalDofOrdinals(testBasisNeighbor,neighborSideOrdinal,neighborGlobalDofs_test);
            numNeighborTrialFields = neighborGlobalDofs_trial.size();
            numNeighborTestFields = neighborGlobalDofs_test.size();
            
            // weight u_values containers using cubature weights defined in cellBasisCacheSide:
            Intrepid::FunctionSpaceTools::multiplyMeasure<double>(trial_values, cellBasisCacheSide->getWeightedMeasures(), trial_values);
            Intrepid::FunctionSpaceTools::multiplyMeasure<double>(neighbor_trial_values, cellBasisCacheSide->getWeightedMeasures(), neighbor_trial_values);
            
            // now, we compute four integrals (test, trial) and insert into global stiffness
            // define a lambda function for insertion
            auto insertValues = [stiffnessMatrix] (vector<int> &rowDofOrdinals, vector<int> &colDofOrdinals,
                                                   Intrepid::FieldContainer<double> &values) -> void
            {
              // values container is (cell, test, trial)
              int rowCount = rowDofOrdinals.size();
              int colCount = colDofOrdinals.size();
              
              //        stiffnessMatrix->InsertGlobalValues(rowCount,&rowDofOrdinals[0],colCount,&colDofOrdinals[0],&values(0,0,0),
              //                                            Epetra_FECrsMatrix::ROW_MAJOR); // COL_MAJOR is the right thing, actually, but for some reason does not work...
              
              // because I don't trust Epetra in terms of the format, let's insert values one at a time
              for (int i=0; i<rowCount; i++)
              {
                for (int j=0; j<colCount; j++)
                {
                  // rows in FieldContainer correspond to trial variables, cols to test
                  // in stiffness matrix, it's the opposite.
                  if (values(0,i,j) != 0) // skip 0's, which I believe Epetra should ignore anyway
                  {
                    //              cout << "Inserting (" << rowDofOrdinals[i] << "," << colDofOrdinals[j] << ") = ";
                    //              cout << values(0,i,j) << endl;
                    stiffnessMatrix->InsertGlobalValues(1,&rowDofOrdinals[i],1,&colDofOrdinals[j],&values(0,i,j));
                  }
                }
              }
            };
            
            auto hasNonzeros = [] (Intrepid::FieldContainer<double> &values, double tol) -> bool
            {
              for (int i=0; i<values.size(); i++)
              {
                if (abs(values[i]) > tol) return true;
              }
              return false;
            };
            
            if (hasNonzeros(test_values,0) && hasNonzeros(trial_values,0))
            {
              Intrepid::FieldContainer<double> integralValues_me_me(numCells,numTestFields,numTrialFields);
              Intrepid::FunctionSpaceTools::integrate<double>(integralValues_me_me,test_values,trial_values,Intrepid::COMP_BLAS);
              insertValues(myGlobalDofs_trial,myGlobalDofs_test,integralValues_me_me);
            }
            
            if (hasNonzeros(neighbor_test_values,0) && hasNonzeros(trial_values,0))
            {
              Intrepid::FieldContainer<double> integralValues_neighbor_me(numCells,numNeighborTestFields,numTrialFields);
              Intrepid::FunctionSpaceTools::integrate<double>(integralValues_neighbor_me,neighbor_test_values,trial_values,Intrepid::COMP_BLAS);
              insertValues(neighborGlobalDofs_test,myGlobalDofs_trial,integralValues_neighbor_me);
            }
            
            if (hasNonzeros(test_values,0) && hasNonzeros(neighbor_trial_values,0))
            {
              Intrepid::FieldContainer<double> integralValues_me_neighbor(numCells,numTestFields,numNeighborTrialFields);
              Intrepid::FunctionSpaceTools::integrate<double>(integralValues_me_neighbor,test_values,neighbor_trial_values,Intrepid::COMP_BLAS);
              insertValues(myGlobalDofs_test,neighborGlobalDofs_trial,integralValues_me_neighbor);
            }
            
            if (hasNonzeros(neighbor_test_values,0) && hasNonzeros(neighbor_trial_values,0))
            {
              Intrepid::FieldContainer<double> integralValues_neighbor_neighbor(numCells,numNeighborTestFields,numNeighborTrialFields);
              Intrepid::FunctionSpaceTools::integrate<double>(integralValues_neighbor_neighbor,neighbor_test_values,neighbor_trial_values,Intrepid::COMP_BLAS);
              insertValues(neighborGlobalDofs_test,neighborGlobalDofs_trial,integralValues_neighbor_neighbor);
            }
          }
        }
      }
    }
  }
  
  Epetra_Vector timeApplyJumpVector(timeMap);
 
  double timeApplyDGJumpTerms = timer.ElapsedTime();
  //  cout << "Done computing local matrices" << endl;
  Epetra_Vector timeApplyJumpTermsVector(timeMap);
  timeApplyJumpTermsVector[0] = timeApplyDGJumpTerms;
  
  int err = timeApplyJumpTermsVector.Norm1( &_totalTimeApplyJumpTerms );
  err = timeApplyJumpTermsVector.MeanValue( &_meanTimeApplyJumpTerms );
  err = timeApplyJumpTermsVector.MinValue( &_minTimeApplyJumpTerms );
  err = timeApplyJumpTermsVector.MaxValue( &_maxTimeApplyJumpTerms );
}

template <typename Scalar>
int TSolution<Scalar>::solve()
{
  return solve(TSolver<Scalar>::getDirectSolver());
}

template <typename Scalar>
int TSolution<Scalar>::solve(bool useMumps)
{
  TSolverPtr<Scalar> solver;
#ifdef HAVE_AMESOS_MUMPS
  if (useMumps)
  {
    solver = Teuchos::rcp(new MumpsSolver());
  }
  else
  {
    solver = Teuchos::rcp(new TAmesos2Solver<Scalar>(false, "klu"));
  }
#else
  solver = Teuchos::rcp(new TAmesos2Solver<Scalar>(false, "klu"));
#endif
  return solve(solver);
}

template <typename Scalar>
void TSolution<Scalar>::setSolution(Teuchos::RCP< TSolution<Scalar> > otherSoln)
{
  _solutionForCellIDGlobal = otherSoln->solutionForCellIDGlobal();
  _lhsVector = Teuchos::rcp( new Epetra_FEVector(*otherSoln->getLHSVector()) );
  clearComputedResiduals();
}

template <typename Scalar>
void TSolution<Scalar>::initializeLHSVector()
{
//  _lhsVector = Teuchos::rcp( (Epetra_FEVector*) NULL); // force a delete
  Epetra_Map partMap = getPartitionMap();
  _lhsVector = Teuchos::rcp(new Epetra_FEVector(partMap,1,true));

  setGlobalSolutionFromCellLocalCoefficients();
  clearComputedResiduals();
}

template <typename Scalar>
void TSolution<Scalar>::initializeLoad()
{
  narrate("initializeLoad");
  Epetra_Map partMap = getPartitionMap();

  _rhsVector = Teuchos::rcp(new Epetra_FEVector(partMap));
}

template <typename Scalar>
void TSolution<Scalar>::initializeStiffnessAndLoad()
{
  narrate("initializeStiffnessAndLoad");
  Epetra_Map partMap = getPartitionMap();
  
//  int maxRowSize = _mesh->rowSizeUpperBound();
  int maxRowSize = 0; // will cause more mallocs during insertion into the CrsMatrix, but will minimize the amount of memory allocated now.
  
  _globalStiffMatrix = Teuchos::rcp(new Epetra_FECrsMatrix(::Copy, partMap, maxRowSize));
  _rhsVector = Teuchos::rcp(new Epetra_FEVector(partMap));
}

template <typename Scalar>
void TSolution<Scalar>::populateStiffnessAndLoad()
{
  narrate("populateStiffnessAndLoad()");
  Epetra_CommPtr Comm = _mesh->Comm();
  int rank = Comm->MyPID();
  int numProcs = Comm->NumProc();

  Epetra_FECrsMatrix* globalStiffness = dynamic_cast<Epetra_FECrsMatrix*>(_globalStiffMatrix.get());

  if (globalStiffness == NULL)
  {
    cout << "Error: Solutio::populateStiffnessAndLoad() requires that _globalStiffMatrix be an Epetra_FECrsMatrix\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "populateStiffnessAndLoad() requires that _globalStiffMatrix be an Epetra_FECrsMatrix");
  }

  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
  Epetra_Map partMap = getPartitionMap();

  vector< ElementTypePtr > elementTypes = _mesh->elementTypes(rank);
  vector< ElementTypePtr >::iterator elemTypeIt;

  //cout << "process " << rank << " about to loop over elementTypes.\n";
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,*Comm);
  Epetra_Time timer(*Comm);
  Epetra_Time subTimer(*Comm);

  double testMatrixAssemblyTime = 0, testMatrixInversionTime = 0, localStiffnessDeterminationFromTestsTime = 0;
  double localStiffnessInterpretationTime = 0, rhsIntegrationAgainstOptimalTestsTime = 0, filterApplicationTime = 0;

  //  cout << "Computing local matrices" << endl;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)
  {
    //cout << "Solution: elementType loop, iteration: " << elemTypeNumber++ << endl;
    ElementTypePtr elemTypePtr = *(elemTypeIt);

    Intrepid::FieldContainer<double> myPhysicalCellNodesForType = _mesh->physicalCellNodes(elemTypePtr);
    Intrepid::FieldContainer<double> myCellSideParitiesForType = _mesh->cellSideParities(elemTypePtr);
    int totalCellsForType = myPhysicalCellNodesForType.dimension(0);
    int startCellIndexForBatch = 0;

    if (totalCellsForType == 0) continue;
    // if we get here, there is at least one, so we find a sample cellID to help us set up prototype BasisCaches:
    GlobalIndexType sampleCellID = _mesh->cellID(elemTypePtr, 0, rank);
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,sampleCellID,false,_cubatureEnrichmentDegree);
    BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(_mesh,sampleCellID,true,_cubatureEnrichmentDegree);

    DofOrderingPtr trialOrderingPtr = elemTypePtr->trialOrderPtr;
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTrialDofs = trialOrderingPtr->totalDofs();
    int numTestDofs = testOrderingPtr->totalDofs();
    int maxCellBatch = MAX_BATCH_SIZE_IN_BYTES / 8 / (numTestDofs*numTestDofs + numTestDofs*numTrialDofs + numTrialDofs*numTrialDofs);
    maxCellBatch = max( maxCellBatch, MIN_BATCH_SIZE_IN_CELLS );
    //cout << "numTestDofs^2:" << numTestDofs*numTestDofs << endl;
    //cout << "maxCellBatch: " << maxCellBatch << endl;

    Teuchos::Array<int> nodeDimensions, parityDimensions;
    myPhysicalCellNodesForType.dimensions(nodeDimensions);
    myCellSideParitiesForType.dimensions(parityDimensions);
    while (startCellIndexForBatch < totalCellsForType)
    {
      int cellsLeft = totalCellsForType - startCellIndexForBatch;
      int numCells = min(maxCellBatch,cellsLeft);

      // determine cellIDs
      vector<GlobalIndexType> cellIDs;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++)
      {
        GlobalIndexType cellID = _mesh->cellID(elemTypePtr, cellIndex+startCellIndexForBatch, rank);
        cellIDs.push_back(cellID);
      }

      //cout << "testDofOrdering: " << *testOrderingPtr;
      //cout << "trialDofOrdering: " << *trialOrderingPtr;
      nodeDimensions[0] = numCells;
      parityDimensions[0] = numCells;
      Intrepid::FieldContainer<double> physicalCellNodes(nodeDimensions,&myPhysicalCellNodesForType(startCellIndexForBatch,0,0));
      Intrepid::FieldContainer<double> cellSideParities(parityDimensions,&myCellSideParitiesForType(startCellIndexForBatch,0));

      bool createSideCacheToo = true;
      basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
      basisCache->setCellSideParities(cellSideParities);

      // hard-coding creating side cache for IP for now, since _ip->hasBoundaryTerms() only recognizes terms explicitly passed in as boundary terms:
      ipBasisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true);//_ip->hasBoundaryTerms()); // create side cache if ip has boundary values
      ipBasisCache->setCellSideParities(cellSideParities); // I don't anticipate these being needed, though

      Intrepid::FieldContainer<Scalar> localStiffness(numCells,numTrialDofs,numTrialDofs);
      Intrepid::FieldContainer<Scalar> localRHSVector(numCells,numTrialDofs);

      if (_bf != Teuchos::null)
        _bf->localStiffnessMatrixAndRHS(localStiffness, localRHSVector, _ip, ipBasisCache, _rhs, basisCache);
      else
        _mesh->bilinearForm()->localStiffnessMatrixAndRHS(localStiffness, localRHSVector, _ip, ipBasisCache, _rhs, basisCache);

      // apply filter(s) (e.g. penalty method, preconditioners, etc.)
      if (_filter.get())
      {
        subTimer.ResetStartTime();
        _filter->filter(localStiffness,localRHSVector,basisCache,_mesh,_bc);
        filterApplicationTime += subTimer.ElapsedTime();
        //        _filter->filter(localRHSVector,physicalCellNodes,cellIDs,_mesh,_bc);
      }

//      cout << "local stiffness matrices:\n" << localStiffness;
//      cout << "local loads:\n" << localRHSVector;

      subTimer.ResetStartTime();

      Intrepid::FieldContainer<GlobalIndexType> globalDofIndices;

      Intrepid::FieldContainer<GlobalIndexTypeToCast> globalDofIndicesCast;

      Teuchos::Array<int> localStiffnessDim(2,numTrialDofs);
      Teuchos::Array<int> localRHSDim(1,numTrialDofs);

      Intrepid::FieldContainer<Scalar> interpretedStiffness;
      Intrepid::FieldContainer<Scalar> interpretedRHS;

      Teuchos::Array<int> dim;

      for (int cellIndex=0; cellIndex<numCells; cellIndex++)
      {
        GlobalIndexType cellID = _mesh->cellID(elemTypePtr,cellIndex+startCellIndexForBatch,rank);
        Intrepid::FieldContainer<Scalar> cellStiffness(localStiffnessDim,&localStiffness(cellIndex,0,0)); // shallow copy
        Intrepid::FieldContainer<Scalar> cellRHS(localRHSDim,&localRHSVector(cellIndex,0)); // shallow copy

        _dofInterpreter->interpretLocalData(cellID, cellStiffness, cellRHS, interpretedStiffness, interpretedRHS, globalDofIndices);

        // cast whatever the global index type is to a type that Epetra supports
        globalDofIndices.dimensions(dim);
        globalDofIndicesCast.resize(dim);

        for (int dofOrdinal = 0; dofOrdinal < globalDofIndices.size(); dofOrdinal++)
        {
          globalDofIndicesCast[dofOrdinal] = globalDofIndices[dofOrdinal];
        }

        globalStiffness->InsertGlobalValues(globalDofIndices.size(),&globalDofIndicesCast(0),
                                            globalDofIndices.size(),&globalDofIndicesCast(0),&interpretedStiffness[0]);
        _rhsVector->SumIntoGlobalValues(globalDofIndices.size(),&globalDofIndicesCast(0),&interpretedRHS[0]);
      }
      localStiffnessInterpretationTime += subTimer.ElapsedTime();

      startCellIndexForBatch += numCells;
    }
  }
  {
    /*    cout << "testMatrixAssemblyTime: " << testMatrixAssemblyTime << " seconds.\n";
        cout << "testMatrixInversionTime: " << testMatrixInversionTime << " seconds.\n";
        cout << "localStiffnessDeterminationFromTestsTime: " << localStiffnessDeterminationFromTestsTime << " seconds.\n";
        cout << "localStiffnessInterpretationTime: " << localStiffnessInterpretationTime << " seconds.\n";
        cout << "rhsIntegrationAgainstOptimalTestsTime: " << rhsIntegrationAgainstOptimalTestsTime << " seconds.\n";
        cout << "filterApplicationTime: " << filterApplicationTime << " seconds.\n";*/
  }

  double timeLocalStiffness = timer.ElapsedTime();
  //  cout << "Done computing local matrices" << endl;
  Epetra_Vector timeLocalStiffnessVector(timeMap);
  timeLocalStiffnessVector[0] = timeLocalStiffness;

  int localRowIndex = myGlobalIndicesSet.size(); // starts where the dofs left off

  // order is: element-lagrange, then (on rank 0) global lagrange and ZMC
  for (int elementConstraintIndex = 0; elementConstraintIndex < _lagrangeConstraints->numElementConstraints();
       elementConstraintIndex++)
  {
    for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)
    {
      ElementTypePtr elemTypePtr = *(elemTypeIt);
      BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh));

      // get cellIDs for basisCache
      vector< ElementPtr > cells = _mesh->elementsOfType(rank,elemTypePtr);
      int numCells = cells.size();
      vector<GlobalIndexType> cellIDs;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++)
      {
        int cellID = cells[cellIndex]->cellID();
        cellIDs.push_back(cellID);
      }
      // set physical cell nodes:
      Intrepid::FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
      bool createSideCacheToo = true;
      basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
      basisCache->setCellSideParities(_mesh->cellSideParities(elemTypePtr));

      int numTrialDofs = elemTypePtr->trialOrderPtr->totalDofs();
      Intrepid::FieldContainer<Scalar> lhs(numCells,numTrialDofs);
      Intrepid::FieldContainer<Scalar> rhs(numCells);
      _lagrangeConstraints->getCoefficients(lhs,rhs,elementConstraintIndex,
                                            elemTypePtr->trialOrderPtr,basisCache);

      Intrepid::FieldContainer<GlobalIndexTypeToCast> globalDofIndices(numTrialDofs+1); // max # of nonzeros
      Intrepid::FieldContainer<Scalar> nonzeroValues(numTrialDofs+1);
      Teuchos::Array<int> localLHSDim(1, numTrialDofs); // changed from (numTrialDofs) by NVR, 8/27/14
      Intrepid::FieldContainer<Scalar> interpretedLHS;

      Intrepid::FieldContainer<GlobalIndexType> interpretedGlobalDofIndices;

      // need to ask for local stiffness, too, for condensed dof interpreter, even though this is not used.
      Intrepid::FieldContainer<Scalar> dummyLocalStiffness(numTrialDofs, numTrialDofs);
      Intrepid::FieldContainer<Scalar> dummyInterpretedStiffness;

      for (int cellIndex=0; cellIndex<numCells; cellIndex++)
      {
        GlobalIndexTypeToCast globalRowIndex = partMap.GID(localRowIndex);
        int nnz = 0;
        Intrepid::FieldContainer<Scalar> localLHS(localLHSDim,&lhs(cellIndex,0)); // shallow copy
        _dofInterpreter->interpretLocalData(cellIDs[cellIndex], dummyLocalStiffness, localLHS, dummyInterpretedStiffness,
                                            interpretedLHS, interpretedGlobalDofIndices);

        for (int i=0; i<interpretedLHS.size(); i++)
        {
          if (interpretedLHS(i) != 0.0)
          {
            globalDofIndices(nnz) = interpretedGlobalDofIndices(i);
            nonzeroValues(nnz) = interpretedLHS(i);
            nnz++;
          }
        }
        // rhs:
        globalDofIndices(nnz) = globalRowIndex;
        if (nnz!=0)
        {
          nonzeroValues(nnz) = 0.0;
        }
        else     // no nonzero weights
        {
          nonzeroValues(nnz) = 1.0; // just put a 1 in the diagonal to avoid singular matrix
        }
        // insert row:
        globalStiffness->InsertGlobalValues(1,&globalRowIndex,nnz+1,&globalDofIndices(0),
                                            &nonzeroValues(0));
        // insert column:
        globalStiffness->InsertGlobalValues(nnz+1,&globalDofIndices(0),1,&globalRowIndex,
                                            &nonzeroValues(0));
        _rhsVector->ReplaceGlobalValues(1,&globalRowIndex,&rhs(cellIndex));

        localRowIndex++;
      }
    }
  }

  //  // compute max, min h
  //  // TODO: get rid of the Global calls below (MPI-enable this code)
  //  double maxCellMeasure = 0;
  //  double minCellMeasure = 1e300;
  //  vector< ElementTypePtr > elemTypes = _mesh->elementTypes(); // global element types
  //  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
  //    ElementTypePtr elemType = *elemTypeIt;
  //    vector< ElementPtr > elems = _mesh->elementsOfTypeGlobal(elemType);
  //    vector<GlobalIndexType> cellIDs;
  //    for (int i=0; i<elems.size(); i++) {
  //      cellIDs.push_back(elems[i]->cellID());
  //    }
  //    Intrepid::FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemType);
  //    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,_mesh) );
  //    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
  //    Intrepid::FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
  //
  //    for (int i=0; i<elems.size(); i++) {
  //      maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
  //      minCellMeasure = min(minCellMeasure,cellMeasures(i));
  //    }
  //  }
  //  double min_h = sqrt(minCellMeasure);
  //  double max_h = sqrt(maxCellMeasure);


  // TODO: change ZMC imposition to be a distributed computation, instead of doing it all on rank 0
  //       (It's both the code below and the integrateBasisFunctions() methods that will need revision.)
  vector<int> zeroMeanConstraints = getZeroMeanConstraints();
  int numGlobalConstraints = _lagrangeConstraints->numGlobalConstraints();
  TEUCHOS_TEST_FOR_EXCEPTION(numGlobalConstraints != 0, std::invalid_argument, "global constraints not yet supported in Solution.");
  for (int lagrangeIndex = 0; lagrangeIndex < numGlobalConstraints; lagrangeIndex++)
  {
    int globalRowIndex = partMap.GID(localRowIndex);

    localRowIndex++;
  }

  // impose zero mean constraints:
  if (!_zmcsAsRankOneUpdate)
  {
    // if neither doing ZMCs as rank one update nor imposing as Lagrange, we nevertheless set up one global row per ZMC
    // on rank 0.  The rationale is that this makes iterative solves using CG easier; it means that we don't need to have
    // a different matrix shape for the case where we rely on the coarse grid solve to impose the ZMC via Lagrange constraints
    // or when we have a rank-one update in an iterative solve to handle that.
    // (We put 1's in the diagonals of the new rows, but otherwise leave them unpopulated.)

    imposeZMCsUsingLagrange();
  }
  else
  {
    // NOTE: this code remains here as reference only; it's quite inefficient because it creates a lot of fill-in for A
    // we may want to implement the same idea, but with a separate Epetra_Operator that simply stores the vector and the weight
    for (vector< int >::iterator trialIt = zeroMeanConstraints.begin(); trialIt != zeroMeanConstraints.end(); trialIt++)
    {
      int trialID = *trialIt;

      // sample an element to make sure that the basis used for trialID is nodal
      // (this is assumed in our imposition mechanism)
      GlobalIndexType firstActiveCellID = *_mesh->getActiveCellIDs().begin();
      ElementTypePtr elemTypePtr = _mesh->getElementType(firstActiveCellID);
      BasisPtr trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);
      if (!trialBasis->isNodal())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Zero-mean constraint imposition assumes a nodal basis, and this basis isn't nodal.");
      }

      GlobalIndexTypeToCast zmcIndex;
      if (rank==0)
        zmcIndex = partMap.GID(localRowIndex);
      else
        zmcIndex = 0;

      zmcIndex = MPIWrapper::sum(*_mesh->Comm(), zmcIndex);

      Intrepid::FieldContainer<Scalar> basisIntegrals;
      Intrepid::FieldContainer<GlobalIndexTypeToCast> globalIndices;
      integrateBasisFunctions(globalIndices,basisIntegrals, trialID);
      int numValues = globalIndices.size();

      Intrepid::FieldContainer<Scalar> product(numValues,numValues);
      Scalar denominator = 0.0;
      for (int i=0; i<numValues; i++)
      {
        denominator += basisIntegrals(i);
      }
      denominator *= denominator;

      for (int i=0; i<numValues; i++)
      {
        for (int j=0; j<numValues; j++)
        {
          product(i,j) = _zmcRho * basisIntegrals(i) * basisIntegrals(j) / denominator;
        }
      }
      globalStiffness->SumIntoGlobalValues(numValues, &globalIndices(0), numValues, &globalIndices(0), &product(0,0));
    }
  }
  // end of ZMC imposition

  Comm->Barrier();  // for cleaner time measurements, let everyone else catch up before calling ResetStartTime() and GlobalAssemble()
  timer.ResetStartTime();

  _rhsVector->GlobalAssemble();

  //  EpetraExt::MultiVectorToMatrixMarketFile("rhs_vector_before_bcs.dat",rhsVector,0,0,false);

  globalStiffness->GlobalAssemble(); // will call globalStiffMatrix.FillComplete();

  double timeGlobalAssembly = timer.ElapsedTime();
  Epetra_Vector timeGlobalAssemblyVector(timeMap);
  timeGlobalAssemblyVector[0] = timeGlobalAssembly;

//  cout << "debugging: outputting stiffness matrix before BC imposition to /tmp/stiffness_noBCs.dat\n";
//  EpetraExt::RowMatrixToMatlabFile("/tmp/stiffness_noBCs.dat",*_globalStiffMatrix);

  // determine and impose BCs

  timer.ResetStartTime();

  imposeBCs();

  double timeBCImposition = timer.ElapsedTime();
  Epetra_Vector timeBCImpositionVector(timeMap);
  timeBCImpositionVector[0] = timeBCImposition;

  _rhsVector->GlobalAssemble();

  Epetra_FEVector lhsVector(partMap, true);

  if (_writeRHSToMatrixMarketFile)
  {
    if (rank==0)
    {
      cout << "Solution: writing rhs to file: " << _rhsFilePath << endl;
    }
    EpetraExt::MultiVectorToMatrixMarketFile(_rhsFilePath.c_str(),*_rhsVector,0,0,false);
  }

  // Dump matrices to disk
  if (_writeMatrixToMatlabFile)
  {
    //    EpetraExt::MultiVectorToMatrixMarketFile("rhs_vector.dat",rhsVector,0,0,false);
    EpetraExt::RowMatrixToMatlabFile(_matrixFilePath.c_str(),*_globalStiffMatrix);
    //    EpetraExt::MultiVectorToMatrixMarketFile("lhs_vector.dat",lhsVector,0,0,false);
  }
  if (_writeMatrixToMatrixMarketFile)
  {
    EpetraExt::RowMatrixToMatrixMarketFile(_matrixFilePath.c_str(),*_globalStiffMatrix,NULL,NULL,false);
  }

  int err = timeLocalStiffnessVector.Norm1( &_totalTimeLocalStiffness );
  err = timeGlobalAssemblyVector.Norm1( &_totalTimeGlobalAssembly );
  err = timeBCImpositionVector.Norm1( &_totalTimeBCImposition );

  err = timeLocalStiffnessVector.MeanValue( &_meanTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MeanValue( &_meanTimeGlobalAssembly );
  err = timeBCImpositionVector.MeanValue( &_meanTimeBCImposition );

  err = timeLocalStiffnessVector.MinValue( &_minTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MinValue( &_minTimeGlobalAssembly );
  err = timeBCImpositionVector.MinValue( &_minTimeBCImposition );

  err = timeLocalStiffnessVector.MaxValue( &_maxTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MaxValue( &_maxTimeGlobalAssembly );
  err = timeBCImpositionVector.MaxValue( &_maxTimeBCImposition );
}

template <typename Scalar>
void TSolution<Scalar>::setProblem(TSolverPtr<Scalar> solver)
{
  // Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp( new Epetra_LinearProblem(&*_globalStiffMatrix, &*_lhsVector, &*_rhsVector));
  solver->setProblem(_globalStiffMatrix, _lhsVector, _rhsVector);
}

template <typename Scalar>
int TSolution<Scalar>::solveWithPrepopulatedStiffnessAndLoad(TSolverPtr<Scalar> solver, bool callResolveInsteadOfSolve)
{
  narrate("solveWithPrepopulatedStiffnessAndLoad()");
  
  Epetra_CommPtr Comm = _mesh->Comm();
  int rank = Comm->MyPID();
  int numProcs = Comm->NumProc();

  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,*Comm);
  Epetra_Time timer(*Comm);

  if (_reportConditionNumber)
  {
    //    double oneNorm = globalStiffMatrix.NormOne();
    Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp( new Epetra_LinearProblem(&*_globalStiffMatrix, &*_lhsVector, &*_rhsVector));
    double condest = conditionNumberEstimate(*problem);
    if (rank == 0)
    {
      // cout << "(one-norm) of global stiffness matrix: " << oneNorm << endl;
      cout << "condition # estimate for global stiffness matrix: " << condest << endl;
    }
    _globalSystemConditionEstimate = condest;
  }

  timer.ResetStartTime();

  int solveSuccess;
  if (!callResolveInsteadOfSolve)
  {
    solveSuccess = solver->solve();
  }
  else
  {
    solveSuccess = solver->resolve();
  }

  if (solveSuccess != 0 )
  {
    if (rank==0) cout << "**** WARNING: in Solution.solve(), solver->solve() failed with error code " << solveSuccess << ". ****\n";
    if (_saveMeshOnSolveError)
    {
      string savePrefix = "solveFailure";
      if (rank==0) cout << "**** Outputting solution and mesh with prefix " << savePrefix << ". ****\n";
      save(savePrefix);
    }
  }

  double timeSolve = timer.ElapsedTime();
  Epetra_Vector timeSolveVector(timeMap);
  timeSolveVector[0] = timeSolve;

  int err = timeSolveVector.Norm1( &_totalTimeSolve );
  err = timeSolveVector.MeanValue( &_meanTimeSolve );
  err = timeSolveVector.MinValue( &_minTimeSolve );
  err = timeSolveVector.MaxValue( &_maxTimeSolve );

  return solveSuccess;
}

template <typename Scalar>
int TSolution<Scalar>::solve(TSolverPtr<Scalar> solver)
{
  if (_oldDofInterpreter.get() != NULL)   // proxy for having a condensation interpreter
  {
    CondensedDofInterpreter<Scalar>* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter<Scalar>*>(_dofInterpreter.get());
    if (condensedDofInterpreter != NULL)
    {
      condensedDofInterpreter->reinitialize();
    }
  }

  initializeLHSVector();
  initializeStiffnessAndLoad();
  setProblem(solver);
  applyDGJumpTerms();
  populateStiffnessAndLoad();
  int solveSuccess = solveWithPrepopulatedStiffnessAndLoad(solver);
//  cout << "about to call importSolution on rank " << rank << endl;
  importSolution();
//  cout << "calling importGlobalSolution (this doesn't scale well, especially in its current form).\n";
//  importGlobalSolution();
//  cout << "about to call clearComputedResiduals on rank " << rank << endl;

  clearComputedResiduals(); // now that we've solved, will need to recompute residuals...

  if (_reportTimingResults )
  {
    reportTimings();
  }

  return solveSuccess;
}

template <typename Scalar>
void TSolution<Scalar>::reportTimings()
{
  int rank = Teuchos::GlobalMPISession::getRank();

  if (rank == 0)
  {
    cout << "****** SUM OF TIMING REPORTS ******\n";
    cout << "localStiffness: " << _totalTimeLocalStiffness << " sec." << endl;
    cout << "globalAssembly: " << _totalTimeGlobalAssembly << " sec." << endl;
    cout << "impose BCs:     " << _totalTimeBCImposition << " sec." << endl;
    cout << "solve:          " << _totalTimeSolve << " sec." << endl;
    cout << "dist. solution: " << _totalTimeDistributeSolution << " sec." << endl << endl;

    cout << "****** MEAN OF TIMING REPORTS ******\n";
    cout << "localStiffness: " << _meanTimeLocalStiffness << " sec." << endl;
    cout << "globalAssembly: " << _meanTimeGlobalAssembly << " sec." << endl;
    cout << "impose BCs:     " << _meanTimeBCImposition << " sec." << endl;
    cout << "solve:          " << _meanTimeSolve << " sec." << endl;
    cout << "dist. solution: " << _meanTimeDistributeSolution << " sec." << endl << endl;

    cout << "****** MAX OF TIMING REPORTS ******\n";
    cout << "localStiffness: " << _maxTimeLocalStiffness << " sec." << endl;
    cout << "globalAssembly: " << _maxTimeGlobalAssembly << " sec." << endl;
    cout << "impose BCs:     " << _maxTimeBCImposition << " sec." << endl;
    cout << "solve:          " << _maxTimeSolve << " sec." << endl;
    cout << "dist. solution: " << _maxTimeDistributeSolution << " sec." << endl << endl;

    cout << "****** MIN OF TIMING REPORTS ******\n";
    cout << "localStiffness: " << _minTimeLocalStiffness << " sec." << endl;
    cout << "globalAssembly: " << _minTimeGlobalAssembly << " sec." << endl;
    cout << "impose BCs:     " << _minTimeBCImposition << " sec." << endl;
    cout << "solve:          " << _minTimeSolve << " sec." << endl;
    cout << "dist. solution: " << _minTimeDistributeSolution << " sec." << endl;
  }
}

template <typename Scalar>
void TSolution<Scalar>::clearComputedResiduals()
{
  _residualsComputed = false;
  _energyErrorComputed = false;
  _rankLocalEnergyErrorComputed = false;
  _energyErrorForCell.clear(); // rank local values
  _energyErrorForCellGlobal.clear();
  _residualForCell.clear();
}

template <typename Scalar>
Teuchos::RCP<Mesh> TSolution<Scalar>::mesh() const
{
  return _mesh;
}

template <typename Scalar>
TBCPtr<Scalar> TSolution<Scalar>::bc() const
{
  return _bc;
}
template <typename Scalar>
TRHSPtr<Scalar> TSolution<Scalar>::rhs() const
{
  return _rhs;
}

template <typename Scalar>
void TSolution<Scalar>::importSolution()
{
  Epetra_CommPtr Comm = _mesh->Comm();
  int rank = Comm->MyPID();

  Epetra_Time timer(*Comm);

//  cout << "on rank " << rank << ", about to determine globalDofIndicesForPartition\n";

  set<GlobalIndexType> globalDofIndicesForMyCells;
  const set<GlobalIndexType>* myCellIDs = &_mesh->globalDofAssignment()->cellsInPartition(-1);
  for (GlobalIndexType cellID : *myCellIDs)
  {
    set<GlobalIndexType> globalDofsForCell = _dofInterpreter->globalDofIndicesForCell(cellID);
//    cout << "globalDofs for cell " << cellID << ":\n";
//    Camellia::print("globalDofIndices", globalDofsForCell);

    globalDofIndicesForMyCells.insert(globalDofsForCell.begin(),globalDofsForCell.end());
  }

//  cout << "on rank " << rank << ", about to create myDofs container of size "<< globalDofIndicesForMyCells.size() << "\n";
  GlobalIndexTypeToCast myDofs[globalDofIndicesForMyCells.size()];
  GlobalIndexTypeToCast* myDof = (globalDofIndicesForMyCells.size() > 0) ? &myDofs[0] : NULL;
  for (set<GlobalIndexType>::iterator dofIndexIt = globalDofIndicesForMyCells.begin();
       dofIndexIt != globalDofIndicesForMyCells.end(); dofIndexIt++)
  {
    *myDof = *dofIndexIt;
    myDof++;
  }
//  cout << "on rank " << rank << ", about to create myCellsMap\n";
  Epetra_Map     myCellsMap(-1, globalDofIndicesForMyCells.size(), myDofs, 0, *Comm);

  // Import solution onto current processor
  Epetra_Map partMap = getPartitionMap();
  Epetra_Import  solnImporter(myCellsMap, partMap);
  Epetra_Vector  solnCoeff(myCellsMap);
//  cout << "on rank " << rank << ", about to Import\n";
  solnCoeff.Import(*_lhsVector, solnImporter, Insert);
//  cout << "on rank " << rank << ", returned from Import\n";

  // copy the dof coefficients into our data structure
  for (GlobalIndexType cellID : *myCellIDs)
  {
//    cout << "on rank " << rank << ", about to interpret data for cell " << cellID << "\n";
    Intrepid::FieldContainer<Scalar> cellDofs(_mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
    _dofInterpreter->interpretGlobalCoefficients(cellID,cellDofs,solnCoeff);
    _solutionForCellIDGlobal[cellID] = cellDofs;
  }
//  cout << "on rank " << rank << ", finished interpretation\n";
  double timeDistributeSolution = timer.ElapsedTime();

  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,*Comm);
  Epetra_Vector timeDistributeSolutionVector(timeMap);
  timeDistributeSolutionVector[0] = timeDistributeSolution;

  int err = timeDistributeSolutionVector.Norm1( &_totalTimeDistributeSolution );
  err = timeDistributeSolutionVector.MeanValue( &_meanTimeDistributeSolution );
  err = timeDistributeSolutionVector.MinValue( &_minTimeDistributeSolution );
  err = timeDistributeSolutionVector.MaxValue( &_maxTimeDistributeSolution );
}

template <typename Scalar>
void TSolution<Scalar>::importSolutionForOffRankCells(std::set<GlobalIndexType> cellIDs)
{
  Epetra_CommPtr Comm = _mesh->Comm();
  int rank = Comm->MyPID();

  // it appears to be important that the requests be sorted by MPI rank number
  // the requestMap below accomplishes that.
  
  map<int, vector<GlobalIndexTypeToCast>> requestMap;
  
  for (GlobalIndexType cellID : cellIDs)
  {
    int partitionForCell = _mesh->globalDofAssignment()->partitionForCellID(cellID);
    if (partitionForCell != rank)
    {
      requestMap[partitionForCell].push_back(cellID);
    }
  }
  
  vector<int> myRequestOwners;
  vector<GlobalIndexTypeToCast> myRequest;

  for (auto entry : requestMap)
  {
    int partition = entry.first;
    for (auto cellIDInPartition : entry.second)
    {
      myRequest.push_back(cellIDInPartition);
      myRequestOwners.push_back(partition);
    }
  }

  int myRequestCount = myRequest.size();

  Teuchos::RCP<Epetra_Distributor> distributor;
#ifdef HAVE_MPI
  Epetra_MpiComm* mpiComm = dynamic_cast<Epetra_MpiComm*>(Comm.get());
  if (mpiComm != NULL)
    distributor = Teuchos::rcp( new Epetra_MpiDistributor(*mpiComm) );
  else
  {
    Epetra_SerialComm* serialComm = dynamic_cast<Epetra_SerialComm*>(Comm.get());
    distributor = Teuchos::rcp( new Epetra_SerialDistributor(*serialComm) );
  }
#else
  Epetra_SerialComm* serialComm = dynamic_cast<Epetra_SerialComm*>(Comm.get());
  distributor = Teuchos::rcp( new Epetra_SerialDistributor(*serialComm) );
#endif

  GlobalIndexTypeToCast* myRequestPtr = NULL;
  int *myRequestOwnersPtr = NULL;
  if (myRequest.size() > 0)
  {
    myRequestPtr = &myRequest[0];
    myRequestOwnersPtr = &myRequestOwners[0];
  }
  int numCellsToExport = 0;
  GlobalIndexTypeToCast* cellIDsToExport = NULL;  // we are responsible for deleting the allocated arrays
  int* exportRecipients = NULL;

  distributor->CreateFromRecvs(myRequestCount, myRequestPtr, myRequestOwnersPtr, true, numCellsToExport, cellIDsToExport, exportRecipients);

  const std::set<GlobalIndexType>* myCells = &_mesh->globalDofAssignment()->cellsInPartition(-1);
  
  vector<int> sizes(numCellsToExport);
  vector<Scalar> dataToExport;
  for (int cellOrdinal=0; cellOrdinal<numCellsToExport; cellOrdinal++)
  {
    GlobalIndexType cellID = cellIDsToExport[cellOrdinal];
    if (myCells->find(cellID) == myCells->end())
    {
      cout << "cellID " << cellID << " does not belong to rank " << rank << endl;
      ostringstream myRankDescriptor;
      myRankDescriptor << "rank " << rank << ", cellID ownership";
      Camellia::print(myRankDescriptor.str().c_str(), *myCells);
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested cellID does not belong to this rank!");
    }

    Intrepid::FieldContainer<Scalar>* solnCoeffs = &_solutionForCellIDGlobal[cellID];
    sizes[cellOrdinal] = solnCoeffs->size();
    for (int dofOrdinal=0; dofOrdinal < solnCoeffs->size(); dofOrdinal++)
    {
      dataToExport.push_back((*solnCoeffs)[dofOrdinal]);
    }
  }

  int objSize = sizeof(Scalar) / sizeof(char);

  int importLength = 0;
  char* importedData = NULL;
  int* sizePtr = NULL;
  char* dataToExportPtr = NULL;
  if (numCellsToExport > 0)
  {
    sizePtr = &sizes[0];
    dataToExportPtr = (char *) &dataToExport[0];
  }
  distributor->Do(dataToExportPtr, objSize, sizePtr, importLength, importedData);
  const char* copyFromLocation = importedData;
  int numDofsImport = importLength / objSize;
  int dofsImported = 0;
  for (vector<GlobalIndexTypeToCast>::iterator cellIDIt = myRequest.begin(); cellIDIt != myRequest.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    Intrepid::FieldContainer<Scalar> cellDofs(_mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
    if (cellDofs.size() + dofsImported > numDofsImport)
    {
      cout << "ERROR: not enough dofs provided to this rank!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Attempt to go beyond array bounds because not enough dofs were imported.");
    }

    Scalar* copyToLocation = &cellDofs[0];
    memcpy(copyToLocation, copyFromLocation, objSize * cellDofs.size());
    copyFromLocation += objSize * cellDofs.size();
    copyToLocation += cellDofs.size(); // copyToLocation has type Scalar*, so this moves the pointer the same # of bytes
    dofsImported += cellDofs.size();
    _solutionForCellIDGlobal[cellID] = cellDofs;
  }

  if( cellIDsToExport != 0 ) delete [] cellIDsToExport;
  if( exportRecipients != 0 ) delete [] exportRecipients;
  if (importedData != 0 ) delete [] importedData;
}

template <typename Scalar>
void TSolution<Scalar>::importGlobalSolution()
{
  Epetra_CommPtr Comm = _mesh->Comm();
  Epetra_Time timer(*Comm);

  GlobalIndexType globalDofCount = _mesh->globalDofAssignment()->globalDofCount();

  GlobalIndexTypeToCast myDofs[globalDofCount];
  GlobalIndexTypeToCast* myDof = (globalDofCount > 0) ? &myDofs[0] : NULL;
  for (GlobalIndexType dofIndex = 0; dofIndex < globalDofCount; dofIndex++)
  {
    *myDof = dofIndex;
    myDof++;
  }

  Epetra_Map     myCellsMap(-1, globalDofCount, myDofs, 0, *Comm);

  // Import global solution onto each processor
  Epetra_Map partMap = getPartitionMap();
  Epetra_Import  solnImporter(myCellsMap, partMap);
  Epetra_Vector  solnCoeff(myCellsMap);
  solnCoeff.Import(*_lhsVector, solnImporter, Insert);

  set<GlobalIndexType> globalActiveCellIDs = _mesh->getActiveCellIDs();
  // copy the dof coefficients into our data structure
  for (set<GlobalIndexType>::iterator cellIDIt = globalActiveCellIDs.begin(); cellIDIt != globalActiveCellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    Intrepid::FieldContainer<Scalar> cellDofs(_mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
    _dofInterpreter->interpretGlobalCoefficients(cellID,cellDofs,solnCoeff);
    _solutionForCellIDGlobal[cellID] = cellDofs;
  }
  double timeDistributeSolution = timer.ElapsedTime();

  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,*Comm);
  Epetra_Vector timeDistributeSolutionVector(timeMap);
  timeDistributeSolutionVector[0] = timeDistributeSolution;

  int err = timeDistributeSolutionVector.Norm1( &_totalTimeDistributeSolution );
  err = timeDistributeSolutionVector.MeanValue( &_meanTimeDistributeSolution );
  err = timeDistributeSolutionVector.MinValue( &_minTimeDistributeSolution );
  err = timeDistributeSolutionVector.MaxValue( &_maxTimeDistributeSolution );
}

template <typename Scalar>
TIPPtr<Scalar> TSolution<Scalar>::ip() const
{
  return _ip;
}

template <typename Scalar>
TBFPtr<Scalar> TSolution<Scalar>::bf() const
{
  return _bf;
}

template <typename Scalar>
void TSolution<Scalar>::imposeBCs()
{
  narrate("imposeBCs()");
  int rank     = Teuchos::GlobalMPISession::getRank();

  Intrepid::FieldContainer<GlobalIndexType> bcGlobalIndices;
  Intrepid::FieldContainer<Scalar> bcGlobalValues;

  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
  //  cout << "rank " << rank << " has " << myGlobalIndicesSet.size() << " locally-owned dof indices.\n";
  Epetra_Map partMap = getPartitionMap();

  _mesh->boundary().bcsToImpose(bcGlobalIndices,bcGlobalValues,*(_bc.get()), myGlobalIndicesSet, _dofInterpreter.get());
  int numBCs = bcGlobalIndices.size();

  Intrepid::FieldContainer<GlobalIndexTypeToCast> bcGlobalIndicesCast;
  // cast whatever the global index type is to a type that Epetra supports
  Teuchos::Array<int> dim;
  bcGlobalIndices.dimensions(dim);
  bcGlobalIndicesCast.resize(dim);
  for (int dofOrdinal = 0; dofOrdinal < bcGlobalIndices.size(); dofOrdinal++)
  {
    bcGlobalIndicesCast[dofOrdinal] = bcGlobalIndices[dofOrdinal];
  }
//  cout << "bcGlobalIndices:" << endl << bcGlobalIndices;
  //  cout << "bcGlobalValues:" << endl << bcGlobalValues;

  Epetra_MultiVector v(partMap,1);
  v.PutScalar(0.0);
  for (int i = 0; i < numBCs; i++)
  {
    v.ReplaceGlobalValue(bcGlobalIndicesCast(i), 0, bcGlobalValues(i));
  }

  Epetra_MultiVector rhsDirichlet(partMap,1);
  _globalStiffMatrix->Apply(v,rhsDirichlet);

  // Update right-hand side
  _rhsVector->Update(-1.0,rhsDirichlet,1.0);

  if (numBCs == 0)
  {
    //cout << "Solution: Warning: Imposing no BCs." << endl;
  }
  else
  {
    int err = _rhsVector->ReplaceGlobalValues(numBCs,&bcGlobalIndicesCast(0),&bcGlobalValues(0));
    if (err != 0)
    {
      cout << "ERROR: rhsVector.ReplaceGlobalValues(): some indices non-local...\n";
    }
    err = _lhsVector->ReplaceGlobalValues(numBCs,&bcGlobalIndicesCast(0),&bcGlobalValues(0));
    if (err != 0)
    {
      cout << "ERROR: rhsVector.ReplaceGlobalValues(): some indices non-local...\n";
    }
  }
  // Zero out rows and columns of stiffness matrix corresponding to Dirichlet edges
  //  and add one to diagonal.
  Intrepid::FieldContainer<int> bcLocalIndices(bcGlobalIndices.dimension(0));
  for (int i=0; i<bcGlobalIndices.dimension(0); i++)
  {
    bcLocalIndices(i) = _globalStiffMatrix->LRID(bcGlobalIndicesCast(i));
  }
  if (numBCs == 0)
  {
    ML_Epetra::Apply_OAZToMatrix(NULL, 0, *_globalStiffMatrix);
  }
  else
  {
    ML_Epetra::Apply_OAZToMatrix(&bcLocalIndices(0), numBCs, *_globalStiffMatrix);
  }
}


template <typename Scalar>
void TSolution<Scalar>::imposeZMCsUsingLagrange()
{
  narrate("imposeZMCsUsingLagrange()");
  int rank = Teuchos::GlobalMPISession::getRank();

  if (_zmcsAsRankOneUpdate)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "imposeZMCsUsingLagrange called when _zmcsAsRankOneUpdate is true!");
  }

  Epetra_Map partMap = getPartitionMap();

  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
  int localRowIndex = myGlobalIndicesSet.size();
  int numLocalActiveElements = _mesh->globalDofAssignment()->cellsInPartition(rank).size();
  localRowIndex += numLocalActiveElements * _lagrangeConstraints->numElementConstraints() + _lagrangeConstraints->numGlobalConstraints();

//  Epetra_FECrsMatrix* globalStiffness = dynamic_cast<Epetra_FECrsMatrix*>(_globalStiffMatrix.get());
//  if (globalStiffness==NULL) {
//
//  }

  // order is: element-lagrange, then (on rank 0) global lagrange and ZMC
  vector<int> zeroMeanConstraints = getZeroMeanConstraints();
  for (vector< int >::iterator trialIt = zeroMeanConstraints.begin(); trialIt != zeroMeanConstraints.end(); trialIt++)
  {
    int trialID = *trialIt;

    // sample an element to make sure that the basis used for trialID is nodal
    // (this is assumed in our imposition mechanism)
    GlobalIndexType firstActiveCellID = *_mesh->getActiveCellIDs().begin();
    ElementTypePtr elemTypePtr = _mesh->getElementType(firstActiveCellID);
    BasisPtr trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);
    if (!trialBasis->isNodal())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Zero-mean constraint imposition assumes a nodal basis, and this basis isn't nodal.");
    }

    GlobalIndexTypeToCast zmcIndex;
    if (rank==0)
      zmcIndex = partMap.GID(localRowIndex);
    else
      zmcIndex = 0;

    zmcIndex = MPIWrapper::sum(*_mesh->Comm(),zmcIndex);

    if (_zmcsAsLagrangeMultipliers)
    {
      //    cout << "Imposing zero-mean constraint for variable " << _mesh->bilinearForm()->trialName(trialID) << endl;
      Intrepid::FieldContainer<Scalar> basisIntegrals;
      Intrepid::FieldContainer<GlobalIndexTypeToCast> globalIndices;
      integrateBasisFunctions(globalIndices,basisIntegrals, trialID);

      Intrepid::FieldContainer<int> offsets;
      Intrepid::FieldContainer<Scalar> allBasisIntegrals;
      MPIWrapper::allGatherCompact(*_mesh->Comm(),allBasisIntegrals,basisIntegrals,offsets);

//      if (rank==0) cout << "allBasisIntegrals:\n" << allBasisIntegrals;

      Intrepid::FieldContainer<GlobalIndexTypeToCast> allGlobalIndices;
      MPIWrapper::allGatherCompact(*_mesh->Comm(),allGlobalIndices,globalIndices,offsets);

//      if (rank==0) cout << "allGlobalIndices:\n" << allGlobalIndices;

      if ((rank == 0) && (allBasisIntegrals.size() > 0))
      {
        // insert the row at zmcIndex with the gathered basis integrals
        _globalStiffMatrix->InsertGlobalValues(zmcIndex,allBasisIntegrals.size(),&allBasisIntegrals(0),&allGlobalIndices(0));
//        cout << "Inserted globalValues for row " << zmcIndex << "; values:\n" << allBasisIntegrals << "indices:\n" << allGlobalIndices;
      }

      // here, we increase the size of the system to accommodate the zmc...
      if (basisIntegrals.size() > 0)
      {
        // insert column:
        for (int valueOrdinal=0; valueOrdinal<basisIntegrals.size(); valueOrdinal++)
        {
//          cout << "Inserting globalValues for (" << globalIndices(valueOrdinal)  << "," << zmcIndex << ") = " << basisIntegrals(valueOrdinal) << endl;
          _globalStiffMatrix->InsertGlobalValues(globalIndices(valueOrdinal),1,&basisIntegrals(valueOrdinal),&zmcIndex);
        }

        // old, FECrsMatrix version below:
//        globalStiffness->InsertGlobalValues(1,&zmcIndex,numValues,&globalIndices(0),&basisIntegrals(0));
//        // insert column:
//        globalStiffness->InsertGlobalValues(numValues,&globalIndices(0),1,&zmcIndex,&basisIntegrals(0));
      }

      //      cout << "in zmc, diagonal entry: " << rho << endl;
      //rho /= numValues;
      if (rank==0)   // insert the diagonal entry on rank 0; other ranks insert basis integrals according to which cells they own
      {
        Scalar rho_entry = - 1.0 / _zmcRho;
        _globalStiffMatrix->InsertGlobalValues(zmcIndex,1,&rho_entry,&zmcIndex);
      }
    }
    else
    {
      // put ones in the diagonal on rank 0
      if (rank==0)   // insert the diagonal entry on rank 0; other ranks insert basis integrals according to which cells they own
      {
        Scalar one = 1.0;
        _globalStiffMatrix->InsertGlobalValues(zmcIndex,1,&one,&zmcIndex);
      }
    }
    if (rank==0) localRowIndex++;
  }
  // end of ZMC imposition
}

template <typename Scalar>
Teuchos::RCP<LocalStiffnessMatrixFilter> TSolution<Scalar>::filter() const
{
  return _filter;
}

template <typename Scalar>
Teuchos::RCP<DofInterpreter> TSolution<Scalar>::getDofInterpreter() const
{
  return _dofInterpreter;
}

template <typename Scalar>
void TSolution<Scalar>::setDofInterpreter(Teuchos::RCP<DofInterpreter> dofInterpreter)
{
  _dofInterpreter = dofInterpreter;
  Epetra_Map map = getPartitionMap();
  Teuchos::RCP<Epetra_Map> mapPtr = Teuchos::rcp( new Epetra_Map(map) ); // copy map to RCP
//  _mesh->boundary().setDofInterpreter(_dofInterpreter.get(), mapPtr);
  // TODO: notice that the above call to Boundary::setDofInterpreter() will cause incompatibilities if two solutions share
  //       a mesh but not a dof interpreter.  This basically only would come up in standard cases if one solution has
  //       had setUseCondensedSolve(true) called, and the other has not.  Not too likely to arise in production code, but
  //       this did come up in the tests in SolutionTests.  In any case, it indicates a poor design; the BC enforcement code
  //       (i.e. what Boundary now controls) really belongs to Solution, not to Mesh.  I.e. each Solution should have a BC
  //       enforcer, not each mesh.  One simple, immediate fix would be to add arguments for dofInterpreter to each BC enforcement
  //       method in Boundary (i.e. don't let Boundary own either the partition map or the dof interpreter reference).
}

template <typename Scalar>
Epetra_MultiVector* TSolution<Scalar>::getGlobalCoefficients()
{
  return (*_lhsVector)(0);
}

template <typename Scalar>
TVectorPtr<Scalar> TSolution<Scalar>::getGlobalCoefficients2()
{
  return _lhsVector2;
}

template <typename Scalar>
double TSolution<Scalar>::globalCondEstLastSolve()
{
  // the condition # estimate for the last system matrix used in a solve, if _reportConditionNumber is true.
  return _globalSystemConditionEstimate;
}

template <typename Scalar>
void TSolution<Scalar>::integrateBasisFunctions(Intrepid::FieldContainer<GlobalIndexTypeToCast> &globalIndices, Intrepid::FieldContainer<Scalar> &values, int trialID)
{
  int rank = Teuchos::GlobalMPISession::getRank();

  // only supports scalar-valued field bases right now...
  int sideIndex = VOLUME_INTERIOR_SIDE_ORDINAL; // field variables only
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  vector<GlobalIndexType> globalIndicesVector;
  vector<Scalar> valuesVector;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++)
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    vector<GlobalIndexType> cellIDs = _mesh->globalDofAssignment()->cellIDsOfElementType(rank,elemTypePtr);
    int numCellsOfType = cellIDs.size();
    int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
    Intrepid::FieldContainer<Scalar> valuesForType(numCellsOfType, basisCardinality);
    integrateBasisFunctions(valuesForType,elemTypePtr,trialID);

    int numTrialDofs = elemTypePtr->trialOrderPtr->totalDofs();
    Intrepid::FieldContainer<Scalar> localDiscreteValues(numTrialDofs);
    Intrepid::FieldContainer<Scalar> interpretedDiscreteValues;
    Intrepid::FieldContainer<GlobalIndexType> globalDofIndices;

    // need to ask for local stiffness, too, for condensed dof interpreter, even though this is not used.
    Intrepid::FieldContainer<Scalar> dummyLocalStiffness(numTrialDofs, numTrialDofs);
    Intrepid::FieldContainer<Scalar> dummyInterpretedStiffness;

    CondensedDofInterpreter<Scalar>* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter<Scalar>*>(_dofInterpreter.get());

    // copy into values:
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++)
    {
      GlobalIndexType cellID = cellIDs[cellIndex];

      for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++)
      {
        IndexType dofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID, dofOrdinal);
        localDiscreteValues(dofIndex) = valuesForType(cellIndex,dofOrdinal);
      }
      Intrepid::FieldContainer<Scalar> storedLoad;
      if (condensedDofInterpreter != NULL)
      {
        // condensedDofInterpreter requires the *true* local stiffness, because it will invert part of it...
        // we assume that the condensedDofInterpreter already has the local stiffness stored:
        // (CondensedDofInterpreter will throw an exception if not)
        dummyLocalStiffness = condensedDofInterpreter->storedLocalStiffnessForCell(cellID);
        // condensedDofInterpreter also requires that we restore the previous load vector for the cell once we're done
        // (otherwise it would store interpretedDiscreteValues as the load, causing errors)
        storedLoad = condensedDofInterpreter->storedLocalLoadForCell(cellID);
      }
      _dofInterpreter->interpretLocalData(cellID, dummyLocalStiffness, localDiscreteValues, dummyInterpretedStiffness,
                                          interpretedDiscreteValues, globalDofIndices);
      if (condensedDofInterpreter != NULL)
      {
        condensedDofInterpreter->storeLoadForCell(cellID, storedLoad);
      }

      for (int dofIndex=0; dofIndex<globalDofIndices.size(); dofIndex++)
      {
        if (interpretedDiscreteValues(dofIndex) != 0)
        {
          globalIndicesVector.push_back(globalDofIndices(dofIndex));
          valuesVector.push_back(interpretedDiscreteValues(dofIndex));
        }
      }
    }
  }
  int numValues = globalIndicesVector.size();
  globalIndices.resize(numValues);
  values.resize(numValues);
  for (int i=0; i<numValues; i++)
  {
    globalIndices[i] = globalIndicesVector[i];
    values[i] = valuesVector[i];
  }
}

template <typename Scalar>
void TSolution<Scalar>::integrateBasisFunctions(Intrepid::FieldContainer<Scalar> &values, ElementTypePtr elemTypePtr, int trialID)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  vector<GlobalIndexType> cellIDs = _mesh->globalDofAssignment()->cellIDsOfElementType(rank,elemTypePtr);

  int numCellsOfType = cellIDs.size();
  if (numCellsOfType==0)
  {
    return;
  }

  int sideIndex = VOLUME_INTERIOR_SIDE_ORDINAL;
  int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(0) != numCellsOfType,
                             std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr), trialBasisCardinality)");
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(1) != basisCardinality,
                             std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr), trialBasisCardinality)");
  BasisPtr trialBasis;
  trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);

  int cubDegree = trialBasis->getDegree();

  BasisCache basisCache(_mesh->physicalCellNodes(elemTypePtr), elemTypePtr->cellTopoPtr, cubDegree);

  Teuchos::RCP < const Intrepid::FieldContainer<Scalar> > trialValuesTransformedWeighted;

  trialValuesTransformedWeighted = basisCache.getTransformedWeightedValues(trialBasis, OP_VALUE);

  if (trialValuesTransformedWeighted->rank() != 3)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "integrateBasisFunctions only supports scalar-valued field variables at present.");
  }
  // integrate:
  int numPoints = trialValuesTransformedWeighted->dimension(2);
  for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++)
  {
    for (int basisIndex=0; basisIndex < basisCardinality; basisIndex++)
    {
      for (int pointIndex=0; pointIndex < numPoints; pointIndex++)
      {
        values(cellIndex,basisIndex) += (*trialValuesTransformedWeighted)(cellIndex,basisIndex,pointIndex);
      }
    }
  }
  //FunctionSpaceTools::integrate<double>(values,*trialValuesTransformedWeighted,ones,COMP_BLAS);
}

template <typename Scalar>
Scalar TSolution<Scalar>::meanValue(int trialID)
{
  return integrateSolution(trialID) / meshMeasure();
}

template <typename Scalar>
double TSolution<Scalar>::meshMeasure()
{
  double value = 0.0;
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes();
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++)
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    int cubDegree = 1;
    BasisCache basisCache(_mesh->physicalCellNodesGlobal(elemTypePtr), elemTypePtr->cellTopoPtr, cubDegree);
    Intrepid::FieldContainer<double> cellMeasures = basisCache.getCellMeasures();
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++)
    {
      value += cellMeasures(cellIndex);
    }
  }
  return value;
}

template <typename Scalar>
double TSolution<Scalar>::InfNormOfSolutionGlobal(int trialID)
{
  Epetra_CommPtr Comm = _mesh->Comm();
  int rank = Comm->MyPID();
  int numProcs = Comm->NumProc();

  int indexBase = 0;
  Epetra_Map procMap(numProcs,indexBase,*Comm);
  double localInfNorm = InfNormOfSolution(trialID);
  Epetra_Vector infNormVector(procMap);
  infNormVector[0] = localInfNorm;
  double globalInfNorm;
  int errCode = infNormVector.NormInf( &globalInfNorm );
  if (errCode!=0)
  {
    cout << "Error in infNormOfSolutionGlobal, errCode = " << errCode << endl;
  }
  return globalInfNorm;
}

template <typename Scalar>
double TSolution<Scalar>::InfNormOfSolution(int trialID)
{
  Epetra_CommPtr Comm = _mesh->Comm();
  int rank = Comm->MyPID();
  int numProcs = Comm->NumProc();

  double value = 0.0;
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++)
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    vector< ElementPtr > cells = _mesh->elementsOfType(rank,elemTypePtr);
    int numCells = cells.size();
    // note: basisCache below will use a greater cubature degree than strictly necessary
    //       (it'll use maxTrialDegree + maxTestDegree, when it only needs maxTrialDegree * 2)
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh));

    // get cellIDs for basisCache
    vector<GlobalIndexType> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      GlobalIndexType cellID = cells[cellIndex]->cellID();
      cellIDs.push_back(cellID);
    }
    Intrepid::FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
    bool createSideCacheToo = false;
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);

    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    Intrepid::FieldContainer<Scalar> values(numCells,numPoints);
    bool weightForCubature = false;
    solutionValues(values, trialID, basisCache, weightForCubature);

    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        value = max(abs(values(cellIndex,ptIndex)),value);
      }
    }
  }
  return value;
}

template <typename Scalar>
double TSolution<Scalar>::L2NormOfSolutionGlobal(int trialID)
{
  VarPtr var = _mesh->varFactory()->trial(trialID);
  SolutionPtr thisPtr = Teuchos::rcp(this,false);
  FunctionPtr solnFxn = Function::solution(var, thisPtr);
  return solnFxn->l2norm(_mesh);
}

template <typename Scalar>
double TSolution<Scalar>::L2NormOfSolution(int trialID)
{
  // at this point, this does precisely what L2NormOfSolutionGlobal does.
  // probably the two should be merged.
  VarPtr var = _mesh->varFactory()->trial(trialID);
  SolutionPtr thisPtr = Teuchos::rcp(this,false);
  FunctionPtr solnFxn = Function::solution(var, thisPtr);
  return solnFxn->l2norm(_mesh);
}

template <typename Scalar>
Teuchos::RCP<LagrangeConstraints> TSolution<Scalar>::lagrangeConstraints() const
{
  return _lagrangeConstraints;
}

template <typename Scalar>
Teuchos::RCP<Epetra_FEVector> TSolution<Scalar>::getLHSVector()
{
  return _lhsVector;
}

template <typename Scalar>
TVectorPtr<Scalar> TSolution<Scalar>::getLHSVector2()
{
  return _lhsVector2;
}

template <typename Scalar>
Scalar TSolution<Scalar>::integrateSolution(int trialID)
{
  Scalar value = 0.0;
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes();
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++)
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    Intrepid::FieldContainer<Scalar> valuesForType(numCellsOfType);
    integrateSolution(valuesForType,elemTypePtr,trialID);
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++)
    {
      value += valuesForType(cellIndex);
    }
  }
  return value;
}

template <typename Scalar>
void TSolution<Scalar>::integrateSolution(Intrepid::FieldContainer<Scalar> &values, ElementTypePtr elemTypePtr, int trialID)
{
  int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
  int sideIndex = VOLUME_INTERIOR_SIDE_ORDINAL;
  int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(0) != numCellsOfType,
                             std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr))");
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 1,
                             std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr))");
  BasisPtr trialBasis;
  trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);

  int cubDegree = trialBasis->getDegree();

  BasisCache basisCache(_mesh->physicalCellNodesGlobal(elemTypePtr), elemTypePtr->cellTopoPtr, cubDegree);

  Teuchos::RCP < const Intrepid::FieldContainer<Scalar> > trialValuesTransformedWeighted;

  trialValuesTransformedWeighted = basisCache.getTransformedWeightedValues(trialBasis, OP_VALUE);

  if (trialValuesTransformedWeighted->rank() != 3)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "integrateSolution only supports scalar-valued field variables at present.");
  }

  // integrate:
  Intrepid::FieldContainer<double> physicalCubaturePoints = basisCache.getPhysicalCubaturePoints();

  Intrepid::FieldContainer<Scalar> solnCoeffs(basisCardinality);

  int numPoints = trialValuesTransformedWeighted->dimension(2);
  for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++)
  {
    int cellID = _mesh->cellID(elemTypePtr,cellIndex);
    solnCoeffsForCellID(solnCoeffs, cellID, trialID, sideIndex);
    for (int basisIndex=0; basisIndex < basisCardinality; basisIndex++)
    {
      for (int pointIndex=0; pointIndex < numPoints; pointIndex++)
      {
        values(cellIndex) += solnCoeffs(basisIndex) * (*trialValuesTransformedWeighted)(cellIndex,basisIndex,pointIndex);
      }
    }
  }
}

template <typename Scalar>
double TSolution<Scalar>::energyErrorTotal()
{
  narrate("energyErrorTotal()");
  double energyErrorSquared = 0.0;
  const map<GlobalIndexType,double>* energyErrorPerCell = &(rankLocalEnergyError());

  for (map<GlobalIndexType,double>::const_iterator cellEnergyIt = energyErrorPerCell->begin();
       cellEnergyIt != energyErrorPerCell->end(); cellEnergyIt++)
  {
    energyErrorSquared += (cellEnergyIt->second) * (cellEnergyIt->second);
  }
  energyErrorSquared = MPIWrapper::sum(energyErrorSquared);
  return sqrt(energyErrorSquared);
}

template <typename Scalar>
const map<GlobalIndexType,double> & TSolution<Scalar>::globalEnergyError()
{
  if ( _energyErrorComputed )
  {
    return _energyErrorForCellGlobal;
  }

  const map<GlobalIndexType,double>* rankLocalEnergy = &rankLocalEnergyError();

  Teuchos::RCP<Epetra_Map> cellMap = _mesh->globalDofAssignment()->getActiveCellMap();

  int cellCount = cellMap->NumGlobalElements();
  Intrepid::FieldContainer<double> globalCellEnergyErrors(cellCount);
  Intrepid::FieldContainer<GlobalIndexTypeToCast> globalCellIDs(cellCount);

  set<GlobalIndexType> rankLocalCells = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++)
  {
    GlobalIndexTypeToCast cellID = *cellIDIt;
    int lid = cellMap->LID(cellID);
    lid += _mesh->globalDofAssignment()->activeCellOffset();
    globalCellEnergyErrors[lid] = rankLocalEnergy->find(cellID)->second;
    globalCellIDs[lid] = cellID;
  }
  MPIWrapper::entryWiseSum(globalCellIDs);
  MPIWrapper::entryWiseSum(globalCellEnergyErrors);

  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    GlobalIndexTypeToCast cellID = globalCellIDs[cellOrdinal];
    _energyErrorForCellGlobal[cellID] = globalCellEnergyErrors[cellOrdinal];
//    if (Teuchos::GlobalMPISession::getRank()==0) {
//      cout << "energy error for cell " << cellID << ": " << _energyErrorForCellGlobal[cellID] << endl;
//    }
  }

//  if (Teuchos::GlobalMPISession::getRank()==0) {
//    cout << "globalCellIDs:\n" << globalCellIDs;
//    cout << "globalCellEnergyErrors:\n" << globalCellEnergyErrors;
//  }

  _energyErrorComputed = true;

  return _energyErrorForCellGlobal;
}

template <typename Scalar>
const map<GlobalIndexType,double> & TSolution<Scalar>::rankLocalEnergyError()
{
  if ( _rankLocalEnergyErrorComputed )
  {
    return _energyErrorForCell;
  }

  computeErrorRepresentation();

  set<GlobalIndexType> rankLocalCells = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    // for error rep v_e, residual res, energyError = sqrt ( ve_^T * res)
    Intrepid::FieldContainer<double> residual = _residualForCell[cellID];
    Intrepid::FieldContainer<double> errorRep = _errorRepresentationForCell[cellID];
    int numTestDofs = residual.dimension(1);
    int numCells = residual.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION( numCells!=1, std::invalid_argument, "In energyError::numCells != 1.");

    double errorSquared = 0.0;
    for (int i=0; i<numTestDofs; i++)
    {
      errorSquared += residual(0,i) * errorRep(0,i);
    }
    _energyErrorForCell[cellID] = sqrt(errorSquared);
  } // end of loop thru element types

  _rankLocalEnergyErrorComputed = true;

  return _energyErrorForCell;
}

template <typename Scalar>
void TSolution<Scalar>::computeErrorRepresentation()
{
  narrate("computeErrorRepresentation()");
  if (!_residualsComputed)
  {
    computeResiduals();
  }
  set<GlobalIndexType> rankLocalCells = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(_mesh, cellID, true);

    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);

    Teuchos::RCP<DofOrdering> testOrdering = elemTypePtr->testOrderPtr;
    CellTopoPtr cellTopo = elemTypePtr->cellTopoPtr;

    int numCells = 1;
    int numTestDofs = testOrdering->totalDofs();

    Intrepid::FieldContainer<double> representationMatrix(numTestDofs, 1);
    Intrepid::FieldContainer<double> errorRepresentation(numCells,numTestDofs);

    Intrepid::FieldContainer<Scalar> ipMatrix(1,numTestDofs,numTestDofs);
    _ip->computeInnerProductMatrix(ipMatrix,testOrdering, ipBasisCache);
    Intrepid::FieldContainer<Scalar> rhsMatrix = _residualForCell[cellID];
    // transpose residual :
    rhsMatrix.resize(numTestDofs, 1);

    // strip cell dimension:
    ipMatrix.resize(ipMatrix.dimension(1),ipMatrix.dimension(2));

    {
      // DEBUGGING
//      cout << "\nIn computeErrorRepresentation, ipMatrix 2-norm condition number: " << SerialDenseWrapper::getMatrixConditionNumber2Norm(ipMatrix);

//      cout << "In computeErrorRepresentation, ipMatrix:\n" << ipMatrix;
//      cout << "In computeErrorRepresentation, rhsMatrix:\n" << rhsMatrix;
    }

    int result = SerialDenseWrapper::solveSystemUsingQR(representationMatrix, ipMatrix, rhsMatrix);
    if (result != 0)
    {
      cout << "WARNING: computeErrorRepresentation: call to solveSystemUsingQR failed with error code " << result << endl;
    }
    for (int i=0; i<numTestDofs; i++)
    {
      errorRepresentation(0,i) = representationMatrix(i,0);
    }
    _errorRepresentationForCell[cellID] = errorRepresentation;
  }
}

template <typename Scalar>
void TSolution<Scalar>::computeResiduals()
{
  narrate("computeResiduals()");
  set<GlobalIndexType> rankLocalCells = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);

    Teuchos::RCP<DofOrdering> trialOrdering = elemTypePtr->trialOrderPtr;
    Teuchos::RCP<DofOrdering> testOrdering = elemTypePtr->testOrderPtr;

    int numCells = 1;
    int numTrialDofs = trialOrdering->totalDofs();
    int numTestDofs  = testOrdering->totalDofs();

    // compute l(v) and store in residuals:
    Intrepid::FieldContainer<double> residual(1,numTestDofs);

    Teuchos::Array<int> oneCellDim(2);
    oneCellDim[0] = 1;
    oneCellDim[1] = numTestDofs;

    Intrepid::FieldContainer<Scalar> localCoefficients;
    if (_solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end())
    {
      localCoefficients = _solutionForCellIDGlobal[cellID];
    }
    else
    {
      localCoefficients.resize(numTrialDofs);
    }

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID, false, _cubatureEnrichmentDegree);
    _rhs->integrateAgainstStandardBasis(residual, testOrdering, basisCache);

//    cout << "computeResiduals(): testOrdering:\n" << *testOrdering;

//    cout << "computeResiduals(): RHS values:\n" << residual;

    // compute b(u, v):
    Intrepid::FieldContainer<Scalar> preStiffness(1,numTestDofs,numTrialDofs );
    Intrepid::FieldContainer<double> cellSideParitiesForCell = _mesh->cellSideParitiesForCell(cellID);
    if (_bf != Teuchos::null)
      _bf->stiffnessMatrix(preStiffness, elemTypePtr, cellSideParitiesForCell, basisCache);
    else
      _mesh->bilinearForm()->stiffnessMatrix(preStiffness, elemTypePtr, cellSideParitiesForCell, basisCache);

    for (int i=0; i<numTestDofs; i++)
    {
      for (int j=0; j<numTrialDofs; j++)
      {
        residual(0,i) -= localCoefficients(j) * preStiffness(0,i,j);
      }
    }

//    cout << "computeResiduals(): residual values:\n" << residual;

    _residualForCell[cellID] = residual;
//    cout << "computed residual vector for cell " << cellID << "; nonzeros:\n";
//    double tol = 1e-15;
//    for (int i=0; i< _residualForCell[cellID].size(); i++) {
//      if (abs(_residualForCell[cellID][i]) > tol) {
//        cout << setw(10) << i << setw(25) << _residualForCell[cellID][i] << endl;
//      }
//    }
  }
  _residualsComputed = true;
}

template <typename Scalar>
void TSolution<Scalar>::discardInactiveCellCoefficients()
{
  set< GlobalIndexType > activeCellIDs = _mesh->getActiveCellIDs();
  vector<GlobalIndexType> cellIDsToErase;
  for (typename map< GlobalIndexType, Intrepid::FieldContainer<Scalar> >::iterator solnIt = _solutionForCellIDGlobal.begin();
       solnIt != _solutionForCellIDGlobal.end(); solnIt++)
  {
    GlobalIndexType cellID = solnIt->first;
    if ( activeCellIDs.find(cellID) == activeCellIDs.end() )
    {
      cellIDsToErase.push_back(cellID);
    }
  }
  for (vector<GlobalIndexType>::iterator it = cellIDsToErase.begin(); it !=cellIDsToErase.end(); it++)
  {
    _solutionForCellIDGlobal.erase(*it);
  }
}

template <typename Scalar>
Teuchos::RCP<Epetra_FEVector> TSolution<Scalar>::getRHSVector()
{
  return _rhsVector;
}

template <typename Scalar>
TVectorPtr<Scalar> TSolution<Scalar>::getRHSVector2()
{
  return _rhsVector2;
}

template <typename Scalar>
Teuchos::RCP<Epetra_CrsMatrix> TSolution<Scalar>::getStiffnessMatrix()
{
  return _globalStiffMatrix;
}

template <typename Scalar>
TMatrixPtr<Scalar> TSolution<Scalar>::getStiffnessMatrix2()
{
  return _globalStiffMatrix2;
}

template <typename Scalar>
void TSolution<Scalar>::setStiffnessMatrix(Teuchos::RCP<Epetra_CrsMatrix> stiffness)
{
  narrate("setStiffnessMatrix()");
//  Epetra_FECrsMatrix* stiffnessFEMatrix = dynamic_cast<Epetra_FECrsMatrix*>(_globalStiffMatrix.get());
  _globalStiffMatrix = stiffness;
}

template <typename Scalar>
void TSolution<Scalar>::setStiffnessMatrix2(TMatrixPtr<Scalar> stiffness)
{
  narrate("setStiffnessMatrix2()");
//  Epetra_FECrsMatrix* stiffnessFEMatrix = dynamic_cast<Epetra_FECrsMatrix*>(_globalStiffMatrix.get());
  _globalStiffMatrix2 = stiffness;
}

template <typename Scalar>
void TSolution<Scalar>::solutionValues(Intrepid::FieldContainer<Scalar> &values, int trialID, BasisCachePtr basisCache,
                                       bool weightForCubature, Camellia::EOperator op)
{
  values.initialize(0.0);
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  int sideIndex = basisCache->getSideIndex();
  bool forceVolumeCoords = false; // used for evaluating fields on sides...
  bool fluxOrTrace;
  if (_bf != Teuchos::null)
    fluxOrTrace = _bf->isFluxOrTrace(trialID);
  else
    fluxOrTrace = _mesh->bilinearForm()->isFluxOrTrace(trialID);
  if ( ( sideIndex != -1 ) && !fluxOrTrace)
  {
    forceVolumeCoords = true;
    //    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    //                       "solutionValues doesn't support evaluation of field variables along sides (not yet anyway).");
  }
  if ( (sideIndex == -1 ) && _mesh->bilinearForm()->isFluxOrTrace(trialID) )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                               "solutionValues doesn't support evaluation of trace or flux variables on element interiors.");
  }

  int numCells = cellIDs.size();
  if (numCells != values.dimension(0))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "first dimension of values should == numCells.");
  }
  int spaceDim = basisCache->getSpaceDim();
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  for (int cellIndex = 0; cellIndex < numCells; cellIndex++)
  {
    GlobalIndexType cellID = cellIDs[cellIndex];


    if ( _solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() )
    {
      // cellID not known -- default values for that cell to 0
//      int rank = Teuchos::GlobalMPISession::getRank();
//      cout << "In TSolution<Scalar>::solutionValues() on rank " << rank << ", data for cellID " << cellID << " not found; defaulting to 0.\n" ;
      continue;
    }
    else
    {
      int rank = Teuchos::GlobalMPISession::getRank();
//      cout << "In TSolution<Scalar>::solutionValues() on rank " << rank << ", data for cellID " << cellID << " found; container size is " << _solutionForCellIDGlobal[cellID].size() << endl;
    }

    Intrepid::FieldContainer<Scalar>* solnCoeffs = &_solutionForCellIDGlobal[cellID];

    DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;

    BasisPtr basis;
    if (fluxOrTrace)
    {
      if (! trialOrder->hasBasisEntry(trialID, sideIndex)) continue;
      basis = trialOrder->getBasis(trialID, sideIndex);
    }
    else
    {
      basis = trialOrder->getBasis(trialID);
    }

    int basisCardinality = basis->getCardinality();

    Teuchos::RCP<const Intrepid::FieldContainer<Scalar> > transformedValues;
    if (weightForCubature)
    {
      if (forceVolumeCoords)
      {
        transformedValues = basisCache->getVolumeBasisCache()->getTransformedWeightedValues(basis,op,sideIndex,true);
      }
      else
      {
        transformedValues = basisCache->getTransformedWeightedValues(basis, op);
      }
    }
    else
    {
      if (forceVolumeCoords)
      {
        transformedValues = basisCache->getVolumeBasisCache()->getTransformedValues(basis, op, sideIndex, true);
      }
      else
      {
        transformedValues = basisCache->getTransformedValues(basis, op);
      }
    }

//    cout << "solnCoeffs:\n" << *solnCoeffs;

    const vector<int> *dofIndices = fluxOrTrace ? &(trialOrder->getDofIndices(trialID,sideIndex))
                                    : &(trialOrder->getDofIndices(trialID));

    int rank = transformedValues->rank() - 3; // 3 ==> scalar valued, 4 ==> vector, etc.

    // now, apply coefficient weights:
    for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++)
    {
      int localDofIndex = (*dofIndices)[dofOrdinal];
      //      cout << "localDofIndex " << localDofIndex << " solnCoeffs(localDofIndex): " << solnCoeffs(localDofIndex) << endl;
      for (int ptIndex=0; ptIndex < numPoints; ptIndex++)
      {
        if (rank == 0)
        {
          values(cellIndex,ptIndex) += (*transformedValues)(cellIndex,dofOrdinal,ptIndex) * (*solnCoeffs)(localDofIndex);
        }
        else if (rank == 1)
        {
          for (int i=0; i<spaceDim; i++)
          {
            values(cellIndex,ptIndex,i) += (*transformedValues)(cellIndex,dofOrdinal,ptIndex,i) * (*solnCoeffs)(localDofIndex);
          }
        }
        else if (rank == 2)
        {
          for (int i=0; i<spaceDim; i++)
          {
            for (int j=0; j<spaceDim; j++)
            {
              values(cellIndex,ptIndex,i,j) += (*transformedValues)(cellIndex,dofOrdinal,ptIndex,i,j) * (*solnCoeffs)(localDofIndex);
            }
          }
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "solutionValues doesn't support values with rank > 2.");
        }
      }
    }
  }
}

template <typename Scalar>
void TSolution<Scalar>::solutionValues(Intrepid::FieldContainer<Scalar> &values, int trialID, const Intrepid::FieldContainer<double> &physicalPoints)
{
  // physicalPoints may have dimensions (C,P,D) or (P,D)
  // either way, this method requires searching the mesh for the points provided
  if (physicalPoints.rank()==3)   // dimensions (C,P,D)
  {
    int numTotalCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    if (values.dimension(0) != numTotalCells)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values.dimension(0) != physicalPoints.dimension(0)");
    }
    if (values.dimension(1) != numPoints)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values.dimension(1) != physicalPoints.dimension(1)");
    }

    values.initialize(0.0);
    for (int cellIndex=0; cellIndex<numTotalCells; cellIndex++)
    {
      Intrepid::FieldContainer<double> cellPoint(1,spaceDim); // a single point to find elem we're in
      for (int i=0; i<spaceDim; i++)
      {
        cellPoint(0,i) = physicalPoints(cellIndex,0,i);
      }
      vector< ElementPtr > elements = _mesh->elementsForPoints(cellPoint); // operate under assumption that all points for a given cell index are in that cell
      ElementPtr elem = elements[0];
      if (elem.get() == NULL) continue;
      ElementTypePtr elemTypePtr = elem->elementType();
      int cellID = elem->cellID();

      Intrepid::FieldContainer<Scalar> solnCoeffs = allCoefficientsForCellID(cellID);
      if (solnCoeffs.size()==0) continue; // cell ID not known: default to zero
      int numCells = 1; // do one cell at a time

      Intrepid::FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);

      // store points in local container
      Intrepid::FieldContainer<double> physicalPointsForCell(numCells,numPoints,spaceDim);
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        for (int dim=0; dim<spaceDim; dim++)
        {
          physicalPointsForCell(0,ptIndex,dim) = physicalPoints(cellIndex,ptIndex,dim);
        }
      }

      // 1. compute refElemPoints, the evaluation points mapped to reference cell:
      Intrepid::FieldContainer<double> refElemPoints(numCells,numPoints, spaceDim);
      CamelliaCellTools::mapToReferenceFrame(refElemPoints,physicalPointsForCell,_mesh->getTopology(),cellID,_mesh->globalDofAssignment()->getCubatureDegree(cellID));
      refElemPoints.resize(numPoints,spaceDim);

      BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID);
      basisCache->setRefCellPoints(refElemPoints);
      std::vector<int> dim;
      values.dimensions(dim);
      dim[0] = 1; // one cell
      std::vector<int> cellOffset = dim;
      cellOffset[0] = cellIndex;
      for (int containerRank=1; containerRank<cellOffset.size(); containerRank++)
      {
        cellOffset[containerRank] = 0;
      }
      Intrepid::FieldContainer<double> cellValues(dim,&values[SerialDenseWrapper::getEnumeration(cellOffset,values)]);
      this->solutionValues(cellValues, trialID, basisCache);
    }
    //  when the cell containing the point is off-rank, we have 0s.
    // We sum entrywise to get the missing values.
    MPIWrapper::entryWiseSum(*_mesh->Comm(),values);
  }
  else     // (P,D) physicalPoints
  {
    // the following is due to the fact that we *do not* transform basis values.
    Camellia::EFunctionSpace fs;
    if (_bf != Teuchos::null)
      fs = _bf->functionSpaceForTrial(trialID);
    else
      fs = _mesh->bilinearForm()->functionSpaceForTrial(trialID);
    TEUCHOS_TEST_FOR_EXCEPTION( (fs != Camellia::FUNCTION_SPACE_HVOL) && (fs != Camellia::FUNCTION_SPACE_HGRAD),
                                std::invalid_argument,
                                "This version of solutionValues only supports HVOL and HGRAD bases.");

    TEUCHOS_TEST_FOR_EXCEPTION( values.dimension(0) != physicalPoints.dimension(0),
                                std::invalid_argument,
                                "values.dimension(0) != physicalPoints.dimension(0).");

    // physicalPoints dimensions: (P,D)
    // values dimensions: (P) or (P,D)
    //int numPoints = physicalPoints.dimension(0);
    int spaceDim = physicalPoints.dimension(1);
    int valueRank = values.rank();
    Teuchos::Array<int> oneValueDimensions;
    oneValueDimensions.push_back(1);
    Teuchos::Array<int> onePointDimensions;
    onePointDimensions.push_back(1); // C (cell)
    onePointDimensions.push_back(1); // P (point)
    onePointDimensions.push_back(spaceDim); // D (space)
    if (valueRank >= 1) oneValueDimensions.push_back(spaceDim);
    Intrepid::FieldContainer<Scalar> oneValue(oneValueDimensions);
    Teuchos::Array<int> oneCellDofsDimensions;
    oneCellDofsDimensions.push_back(0); // initialize according to elementType
    bool nullElementsOffRank = true;
    vector< ElementPtr > elements = _mesh->elementsForPoints(physicalPoints, nullElementsOffRank);
    vector< ElementPtr >::iterator elemIt;
    int physicalPointIndex = -1;
    values.initialize(0.0);
    for (elemIt = elements.begin(); elemIt != elements.end(); elemIt++)
    {
      physicalPointIndex++;
      ElementPtr elem = *elemIt;
      if (elem.get() == NULL)
      {
        // values for this point will already have been initialized to 0, the best we can do...
        continue;
      }
      ElementTypePtr elemTypePtr = elem->elementType();

      int cellID = elem->cellID();
      Intrepid::FieldContainer<Scalar> solnCoeffs = allCoefficientsForCellID(cellID);
      if (solnCoeffs.size()==0) continue; // cell ID not known: default to zero

      int numCells = 1;
      int numPoints = 1;

      Intrepid::FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);

      Intrepid::FieldContainer<double> physicalPoint(onePointDimensions);
      for (int dim=0; dim<spaceDim; dim++)
      {
        physicalPoint[dim] = physicalPoints(physicalPointIndex,dim);
      }

      // 1. Map the physicalPoints from the element specified in physicalCellNodes into reference element
      // 2. Compute each basis on those points
      // 3. Transform those basis evaluations back into the physical space
      // 4. Multiply by the solnCoeffs

      // 1. compute refElemPoints, the evaluation points mapped to reference cell:
      Intrepid::FieldContainer<double> refElemPoint(numCells, numPoints, spaceDim);
      CellTopoPtr cellTopo = elemTypePtr->cellTopoPtr;
      CamelliaCellTools::mapToReferenceFrame(refElemPoint,physicalPoint,_mesh->getTopology(),cellID,_mesh->globalDofAssignment()->getCubatureDegree(cellID));
      refElemPoint.resize(numPoints,spaceDim);

      Teuchos::RCP<DofOrdering> trialOrder = elemTypePtr->trialOrderPtr;
      int basisRank = trialOrder->getBasisRank(trialID);

      BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID);
      basisCache->setRefCellPoints(refElemPoint);
      Teuchos::Array<int> dim;
      dim.push_back(1); // one cell
      dim.push_back(1); // one point

      for (int rank=0; rank<basisRank; rank++)
      {
        dim.push_back(spaceDim);
      }

      Teuchos::Array<int> cellOffset = dim;
      Intrepid::FieldContainer<double> cellValues(dim);
      this->solutionValues(cellValues, trialID, basisCache);

      if (basisRank == 0)
      {
        values(physicalPointIndex) = cellValues(0,0);
      }
      else if (basisRank == 1)
      {
        for (int d=0; d<spaceDim; d++)
        {
          values(physicalPointIndex,d) = cellValues(0,0,d);
        }
      }
      else if (basisRank == 2)
      {
        for (int d0=0; d0<spaceDim; d0++)
        {
          for (int d1=0; d1<spaceDim; d1++)
          {
            values(physicalPointIndex,d0,d1) = cellValues(0,0,d0,d1);
          }
        }
      }
      else
      {
        cout << "unhandled basis rank.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled basis rank.");
      }
    }
    // for the (P,D) version of this method, when the cell containing the point is off-rank, we have 0s.
    // We sum entrywise to get the missing values.
    MPIWrapper::entryWiseSum(values);
  } // end (P,D)
}

void determineQuadEdgeWeights(double weights[], int edgeVertexNumber, int numDivisionsPerEdge, bool xEdge)
{
  if (xEdge)
  {
    weights[0] = ((double)(numDivisionsPerEdge - edgeVertexNumber)) / (double)numDivisionsPerEdge;
    weights[1] = ((double)edgeVertexNumber) / (double)numDivisionsPerEdge;
    weights[2] = ((double)edgeVertexNumber) / (double)numDivisionsPerEdge;
    weights[3] = ((double)(numDivisionsPerEdge - edgeVertexNumber)) / (double)numDivisionsPerEdge;
  }
  else
  {
    weights[0] = ((double)(numDivisionsPerEdge - edgeVertexNumber)) / (double)numDivisionsPerEdge;
    weights[1] = ((double)(numDivisionsPerEdge - edgeVertexNumber)) / (double)numDivisionsPerEdge;
    weights[2] = ((double)edgeVertexNumber) / (double)numDivisionsPerEdge;
    weights[3] = ((double)edgeVertexNumber) / (double)numDivisionsPerEdge;
  }
}

template <typename Scalar>
void TSolution<Scalar>::writeStatsToFile(const string &filePath, int precision)
{
  // writes out rows of the format: "cellID patchID x y solnValue"
  ofstream fout(filePath.c_str());
  fout << setprecision(precision);
  fout << "stat.\tmean\tmin\tmax\ttotal\n";
  fout << "localStiffness\t" << _meanTimeLocalStiffness << "\t" <<_minTimeLocalStiffness << "\t" <<_maxTimeLocalStiffness << "\t" << _totalTimeLocalStiffness << endl;
  fout << "globalAssembly\t" <<  _meanTimeGlobalAssembly << "\t" <<_minTimeGlobalAssembly << "\t" <<_maxTimeGlobalAssembly << "\t" << _totalTimeGlobalAssembly << endl;
  fout << "impose BCs\t" <<  _meanTimeBCImposition << "\t" <<_minTimeBCImposition << "\t" <<_maxTimeBCImposition << "\t" << _totalTimeBCImposition << endl;
  fout << "solve\t" << _meanTimeSolve << "\t" <<_minTimeSolve << "\t" <<_maxTimeSolve << "\t" << _totalTimeSolve << endl;
  fout << "dist. solution\t" <<  _meanTimeDistributeSolution << "\t" << _minTimeDistributeSolution << "\t" <<_maxTimeDistributeSolution << "\t" << _totalTimeDistributeSolution << endl;
}

template <typename Scalar>
void TSolution<Scalar>::writeToFile(int trialID, const string &filePath)
{
  // writes out rows of the format: "cellID patchID x y solnValue"
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...

  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);

    CellTopoPtr cellTopo = elemTypePtr->cellTopoPtr;
    Intrepid::FieldContainer<double> vertexPoints(cellTopo->getVertexCount(),cellTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(vertexPoints, cellTopo);

    int numVertices = vertexPoints.dimension(1);

    int numDivisionsPerEdge = 1; //basisDegree*basisDegree;
    int numPatchesPerCell = numDivisionsPerEdge*numDivisionsPerEdge;

    Intrepid::FieldContainer<double> refPoints(numPatchesPerCell*numVertices,spaceDim);

    if (numVertices == 4)   // only quads supported by the multi-patch cell stuff below...
    {
      //if (   ( elemTypePtr->cellTopoPtr->getKey() == shards::Quadrilateral<4>::key )
      //    || (elemTypePtr->cellTopoPtr->getKey() == shards::Triangle<3>::key ) ) {

      Intrepid::FieldContainer<double> iVertex(spaceDim), jVertex(spaceDim);
      Intrepid::FieldContainer<double> v1(spaceDim), v2(spaceDim), v3(spaceDim);

      if (numVertices == 4)
      {
        double yWeights[numVertices], xWeights[numVertices];
        for (int i=0; i<numDivisionsPerEdge; i++)
        {
          for (int j=0; j<numDivisionsPerEdge; j++)
          {
            //cout << "weights: " << xWeights[0]*yWeights[0] << " " << xWeights[1]*yWeights[1] << " " << xWeights[2]*yWeights[2] << " " << xWeights[3]*yWeights[3] << "\n";
            int patchIndex = (i*numDivisionsPerEdge + j);
            refPoints.initialize(0.0);
            for (int patchVertexIndex=0; patchVertexIndex < numVertices; patchVertexIndex++)
            {
              int xOffset = ((patchVertexIndex==0) || (patchVertexIndex==3)) ? 0 : 1;
              int yOffset = ((patchVertexIndex==0) || (patchVertexIndex==1)) ? 0 : 1;
              determineQuadEdgeWeights(xWeights,i+xOffset,numDivisionsPerEdge,true);
              determineQuadEdgeWeights(yWeights,j+yOffset,numDivisionsPerEdge,false);

              for (int vertexIndex=0; vertexIndex < numVertices; vertexIndex++)
              {
                double weight = xWeights[vertexIndex] * yWeights[vertexIndex];
                for (int dim=0; dim<spaceDim; dim++)
                {
                  refPoints(patchIndex*numVertices + patchVertexIndex, dim) += weight*vertexPoints(vertexIndex, dim);
                }
              }
            }
          }
        }
      }

    }
    else
    {
      refPoints = vertexPoints;
      numPatchesPerCell = 1;
    }

    BasisCachePtr basisCache = BasisCache::basisCacheForCellType(_mesh, elemTypePtr);

    basisCache->setPhysicalCellNodes(_mesh->physicalCellNodesGlobal(elemTypePtr), _mesh->cellIDsOfType(elemTypePtr), false);
    basisCache->setRefCellPoints(refPoints);
    int numCells = basisCache->cellIDs().size();

    Intrepid::FieldContainer<double> values(numCells, numPatchesPerCell * numVertices);
    this->solutionValues(values, trialID, basisCache);

    Intrepid::FieldContainer<double> physPoints = basisCache->getPhysicalCubaturePoints();

    for (int cellIndex=0; cellIndex < numCells; cellIndex++)
    {
      for (int patchIndex=0; patchIndex < numPatchesPerCell; patchIndex++)
      {
        for (int ptIndex=0; ptIndex< numVertices; ptIndex++)
        {
          double x = physPoints(cellIndex,patchIndex*numVertices + ptIndex,0);
          double y = physPoints(cellIndex,patchIndex*numVertices + ptIndex,1);
          double z = values(cellIndex,patchIndex*numVertices + ptIndex);
          fout << _mesh->cellID(elemTypePtr,cellIndex) << " " << patchIndex << " " << x << " " << y << " " << z << endl;
        }
      }
    }
  }
  fout.close();
}

template <typename Scalar>
void TSolution<Scalar>::writeQuadSolutionToFile(int trialID, const string &filePath)
{
  // writes out rows of the format: "cellID xIndex yIndex x y solnValue"
  // it's a goofy thing, largely because the MATLAB routine we're using
  // wants a cartesian product with every combo (x_i, y_j) somewhere in the mix...
  // The upshot is that the following will work reasonably only so long as our element
  // boundaries are in a nice Cartesian grid.  But that's more a problem for MATLAB
  // than it is for us...
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...

  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    int basisDegree = elemTypePtr->trialOrderPtr->getBasis(trialID)->getDegree();
    CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;
    // 0. Set up Cubature
    // Get numerical integration points--these will be the points we compute the solution values for...
    CubatureFactory  cubFactory;
    int cubDegree = 2*basisDegree;
    Teuchos::RCP<Intrepid::Cubature<double> > cellTopoCub = cubFactory.create(cellTopoPtr, cubDegree);

    int cubDim       = cellTopoCub->getDimension();
    int numCubPoints = cellTopoCub->getNumPoints();

    Intrepid::FieldContainer<double> cubPoints(numCubPoints, cubDim);
    Intrepid::FieldContainer<double> cubWeights(numCubPoints);

    cellTopoCub->getCubature(cubPoints, cubWeights);

    // here's a hackish bit: collect all the x and y coordinates, and add the vertices
    set<double> xCoords, yCoords;
    for (int ptIndex=0; ptIndex<numCubPoints; ptIndex++)
    {
      xCoords.insert(cubPoints(ptIndex,0));
      yCoords.insert(cubPoints(ptIndex,1));
    }

    // add vertices (for ref quad)
    xCoords.insert(-1.0);
    xCoords.insert(1.0);
    yCoords.insert(-1.0);
    yCoords.insert(1.0);

    // now, make ourselves a new set of "cubature" points:
    numCubPoints = xCoords.size() * yCoords.size();
    cubPoints.resize(numCubPoints,cubDim);
    int ptIndex = 0;
    set<double>::iterator xIt, yIt;
    for (xIt = xCoords.begin(); xIt != xCoords.end(); xIt++)
    {
      for (yIt = yCoords.begin(); yIt != yCoords.end(); yIt++)
      {
        cubPoints(ptIndex,0) = *xIt;
        cubPoints(ptIndex,1) = *yIt;
        ptIndex++;
      }
    }
    BasisCachePtr basisCache = BasisCache::basisCacheForCellType(_mesh, elemTypePtr);

    basisCache->setPhysicalCellNodes(_mesh->physicalCellNodesGlobal(elemTypePtr), _mesh->cellIDsOfType(elemTypePtr), false);
    basisCache->setRefCellPoints(cubPoints);

    Intrepid::FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();

    Intrepid::FieldContainer<double> values(numCellsOfType, numCubPoints);
    solutionValues(values,trialID,basisCache);

    map<float,int> xIndices;
    map<float,int> yIndices; // use floats to truncate insignificant digits...
    for (int cellIndex=0; cellIndex < numCellsOfType; cellIndex++)
    {
      xIndices.clear();
      yIndices.clear();
      for (int ptIndex=0; ptIndex< numCubPoints; ptIndex++)
      {
        double x = physCubPoints(cellIndex,ptIndex,0);
        double y = physCubPoints(cellIndex,ptIndex,1);
        double z = values(cellIndex,ptIndex);
        if (xIndices.find(x) == xIndices.end())
        {
          int xIndex = xIndices.size();
          xIndices[x] = xIndex;
        }
        if (yIndices.find(y) == yIndices.end())
        {
          int yIndex = yIndices.size();
          yIndices[y] = yIndex;
        }
        fout << _mesh->cellID(elemTypePtr,cellIndex) << " " << xIndices[x] << " " << yIndices[y] << " " << x << " " << y << " " << z << endl;
      }
    }
  }
  fout.close();
}

template <typename Scalar>
Intrepid::FieldContainer<Scalar> TSolution<Scalar>::solutionForElementTypeGlobal(ElementTypePtr elemType)
{
  vector< ElementPtr > elementsOfType = _mesh->elementsOfTypeGlobal(elemType);
  int numDofsForType = elemType->trialOrderPtr->totalDofs();
  int numCellsOfType = elementsOfType.size();
  Intrepid::FieldContainer<Scalar> solutionCoeffs(numCellsOfType,numDofsForType);
  for (vector< ElementPtr >::iterator elemIt = elementsOfType.begin();
       elemIt != elementsOfType.end(); elemIt++)
  {
    int globalCellIndex = (*elemIt)->globalCellIndex();
    int cellID = (*elemIt)->cellID();
    for (int dofIndex=0; dofIndex<numDofsForType; dofIndex++)
    {
      if (( _solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end())
          && (_solutionForCellIDGlobal[cellID].size() == numDofsForType))
      {
        solutionCoeffs(globalCellIndex,dofIndex) = _solutionForCellIDGlobal[cellID](dofIndex);
      }
      else     // no solution set for that cellID, return 0
      {
        solutionCoeffs(globalCellIndex,dofIndex) = 0.0;
      }
    }
  }
  return solutionCoeffs;
}

// static method interprets a set of trial ordering coefficients in terms of a specified DofOrdering
// and returns a set of weights for the appropriate basis
template <typename Scalar>
void TSolution<Scalar>::basisCoeffsForTrialOrder(Intrepid::FieldContainer<Scalar> &basisCoeffs, DofOrderingPtr trialOrder,
    const Intrepid::FieldContainer<Scalar> &allCoeffs,
    int trialID, int sideIndex)
{
  if (! trialOrder->hasBasisEntry(trialID, sideIndex))
  {
    basisCoeffs.resize(0);
    return;
  }

  BasisPtr basis = trialOrder->getBasis(trialID,sideIndex);

  int basisCardinality = basis->getCardinality();
  basisCoeffs.resize(basisCardinality);

  for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++)
  {
    int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
    basisCoeffs(dofOrdinal) = allCoeffs(localDofIndex);
  }
}

template <typename Scalar>
void TSolution<Scalar>::solnCoeffsForCellID(Intrepid::FieldContainer<Scalar> &solnCoeffs, GlobalIndexType cellID, int trialID, int sideIndex)
{
  Teuchos::RCP< DofOrdering > trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;

  if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() )
  {
    cout << "Warning: solution for cellID " << cellID << " not found; returning 0.\n";
    BasisPtr basis = trialOrder->getBasis(trialID,sideIndex);
    int basisCardinality = basis->getCardinality();
    solnCoeffs.resize(basisCardinality);
    solnCoeffs.initialize();
    return;
  }

  basisCoeffsForTrialOrder(solnCoeffs, trialOrder, _solutionForCellIDGlobal[cellID], trialID, sideIndex);
}

template <typename Scalar>
const Intrepid::FieldContainer<Scalar>& TSolution<Scalar>::allCoefficientsForCellID(GlobalIndexType cellID, bool warnAboutOffRankImports)
{
  int myRank                    = Teuchos::GlobalMPISession::getRank();
  PartitionIndexType cellRank   = _mesh->globalDofAssignment()->partitionForCellID(cellID);

  bool cellIsRankLocal = (cellRank == myRank);
  if (cellIsRankLocal)
  {
    return _solutionForCellIDGlobal[cellID];
  }
  else
  {
    if ((warnAboutOffRankImports) && (cellRank != -1))   // we don't warn about cells that don't have ranks (can happen on refinement, say)
    {
      cout << "Warning: allCoefficientsForCellID() called on rank " << myRank << " for non-rank-local cell " << cellID;
      cout << ", which belongs to rank " << cellRank << endl;
    }
    return _solutionForCellIDGlobal[cellID];
  }
}

template <typename Scalar>
void TSolution<Scalar>::setBC( TBCPtr<Scalar> bc)
{
  _bc = bc;
}

template <typename Scalar>
void TSolution<Scalar>::setFilter(Teuchos::RCP<LocalStiffnessMatrixFilter> newFilter)
{
  _filter = newFilter;
}

template <typename Scalar>
void TSolution<Scalar>::setIP( TIPPtr<Scalar> ip)
{
  _ip = ip;
  // any computed residuals will need to be recomputed with the new IP
  clearComputedResiduals();
}

template <typename Scalar>
void TSolution<Scalar>::setLagrangeConstraints( Teuchos::RCP<LagrangeConstraints> lagrangeConstraints)
{
  _lagrangeConstraints = lagrangeConstraints;
}

template <typename Scalar>
void TSolution<Scalar>::setReportConditionNumber(bool value)
{
  _reportConditionNumber = value;
}

template <typename Scalar>
void TSolution<Scalar>::setLocalCoefficientsForCell(GlobalIndexType cellID, const Intrepid::FieldContainer<Scalar> &coefficients)
{
  if (coefficients.size() != _mesh->getElementType(cellID)->trialOrderPtr->totalDofs())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coefficients container doesn't have the right # of dofs");
  }
  if (coefficients.rank() != 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coefficients container doesn't have the right shape; should be rank 1");
  }
  _solutionForCellIDGlobal[cellID] = coefficients;
}

template <typename Scalar>
void TSolution<Scalar>::setReportTimingResults(bool value)
{
  _reportTimingResults = value;
}

template <typename Scalar>
void TSolution<Scalar>::setRHS( TRHSPtr<Scalar> rhs)
{
  _rhs = rhs;
  clearComputedResiduals();
}

template <typename Scalar>
void TSolution<Scalar>::setSolnCoeffsForCellID(Intrepid::FieldContainer<Scalar> &solnCoeffsToSet, GlobalIndexType cellID)
{
  _solutionForCellIDGlobal[cellID] = solnCoeffsToSet;
  _mesh->globalDofAssignment()->interpretLocalCoefficients(cellID,solnCoeffsToSet,*_lhsVector);
}

template <typename Scalar>
void TSolution<Scalar>::setSolnCoeffsForCellID(Intrepid::FieldContainer<Scalar> &solnCoeffsToSet, GlobalIndexType cellID, int trialID, int sideIndex)
{
  ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);

  Teuchos::RCP< DofOrdering > trialOrder = elemTypePtr->trialOrderPtr;
  BasisPtr basis = trialOrder->getBasis(trialID,sideIndex);

  int basisCardinality = basis->getCardinality();
  if ( _solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() )
  {
    // allocate new storage
    _solutionForCellIDGlobal[cellID] = Intrepid::FieldContainer<Scalar>(trialOrder->totalDofs());
  }
  if (_solutionForCellIDGlobal[cellID].size() != trialOrder->totalDofs())
  {
    // resize
    _solutionForCellIDGlobal[cellID].resize(trialOrder->totalDofs());
  }
  TEUCHOS_TEST_FOR_EXCEPTION(solnCoeffsToSet.size() != basisCardinality, std::invalid_argument, "solnCoeffsToSet.size() != basisCardinality");
  for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++)
  {
    int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
    _solutionForCellIDGlobal[cellID](localDofIndex) = solnCoeffsToSet[dofOrdinal];
  }
  
  if (_lhsVector != Teuchos::null) // if _lhsVector hasn't been initialized, don't map back to global solution vector
  {
    // non-null _oldDofInterpreter is a proxy for having a condensation interpreter
    // if using static condensation, we skip storing field values
    if ((_oldDofInterpreter.get() == NULL) || (trialOrder->getNumSidesForVarID(trialID) != 1))
    {
      Intrepid::FieldContainer<Scalar> globalCoefficients;
      Intrepid::FieldContainer<GlobalIndexType> globalDofIndices;
      _dofInterpreter->interpretLocalBasisCoefficients(cellID, trialID, sideIndex, solnCoeffsToSet, globalCoefficients, globalDofIndices);

      for (int i=0; i<globalCoefficients.size(); i++)
      {
        _lhsVector->ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndices[i], 0, globalCoefficients[i]);
      }
    }
  }

  // could stand to be more granular, maybe, but if we're changing the solution, the present
  // policy is to invalidate any computed residuals
  clearComputedResiduals();
}

// protected method; used for solution comparison...
template <typename Scalar>
const map< GlobalIndexType, Intrepid::FieldContainer<Scalar> > & TSolution<Scalar>::solutionForCellIDGlobal() const
{
  return _solutionForCellIDGlobal;
}

template <typename Scalar>
void TSolution<Scalar>::setWriteMatrixToFile(bool value, const string &filePath)
{
  _writeMatrixToMatlabFile = value;
  _matrixFilePath = filePath;
}

template <typename Scalar>
void TSolution<Scalar>::setWriteMatrixToMatrixMarketFile(bool value, const string &filePath)
{
  _writeMatrixToMatrixMarketFile = value;
  _matrixFilePath = filePath;
}

template <typename Scalar>
void TSolution<Scalar>::setWriteRHSToMatrixMarketFile(bool value, const string &filePath)
{
  _writeRHSToMatrixMarketFile = value;
  _rhsFilePath = filePath;
}

template <typename Scalar>
void TSolution<Scalar>::condensedSolve(TSolverPtr<Scalar> globalSolver, bool reduceMemoryFootprint,
                                       set<GlobalIndexType> offRankCellsToInclude)
{
  // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
  vector<int> trialIDs;
  if (_bf != Teuchos::null)
    trialIDs = _bf->trialIDs();
  else
    trialIDs = _mesh->bilinearForm()->trialIDs();

  set< int > fieldsToExclude;
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++)
  {
    int trialID = *trialIt;
    if (_bc->shouldImposeZeroMeanConstraint(trialID) || _bc->singlePointBC(trialID) )
    {
      fieldsToExclude.insert(trialID);
    }
  }

  // override reduceMemoryFootprint for now (since CondensedDofInterpreter doesn't yet support a true value)
  reduceMemoryFootprint = false;

  Teuchos::RCP<DofInterpreter> dofInterpreter = Teuchos::rcp(new CondensedDofInterpreter<Scalar>(_mesh, _ip, _rhs, _lagrangeConstraints.get(), fieldsToExclude, !reduceMemoryFootprint, offRankCellsToInclude) );

  Teuchos::RCP<DofInterpreter> oldDofInterpreter = _dofInterpreter;

  setDofInterpreter(dofInterpreter);

  solve(globalSolver);

  setDofInterpreter(oldDofInterpreter);
}

// must write to .m file
template <typename Scalar>
void TSolution<Scalar>::writeFieldsToFile(int trialID, const string &filePath)
{

  //  cout << "writeFieldsToFile for trialID: " << trialID << endl;

  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...
  int num1DPts = 5;

  fout << "numCells = " << _mesh->activeElements().size() << endl;
  fout << "x=cell(numCells,1);y=cell(numCells,1);z=cell(numCells,1);" << endl;

  // initialize storage
  fout << "for i = 1:numCells" << endl;
  fout << "x{i} = zeros(" << num1DPts << ",1);"<<endl;
  fout << "y{i} = zeros(" << num1DPts << ",1);"<<endl;
  fout << "z{i} = zeros(" << num1DPts << ");"<<endl;
  fout << "end" << endl;
  int globalCellInd = 1; //matlab indexes from 1
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)   //thru quads/triangles/etc
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    CellTopoPtr cellTopo = elemTypePtr->cellTopoPtr;

    Intrepid::FieldContainer<double> vertexPoints, physPoints;
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    Intrepid::FieldContainer<double> physicalCellNodes = _mesh()->physicalCellNodesGlobal(elemTypePtr);

    int numCells = physicalCellNodes.dimension(0);
    bool createSideCacheToo = false;
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, createSideCacheToo));

    vector<GlobalIndexType> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      GlobalIndexType cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: global cellID
      cellIDs.push_back(cellID);
    }

    int numPoints = num1DPts * num1DPts;
    Intrepid::FieldContainer<double> refPoints(numPoints,spaceDim);
    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++)
    {
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++)
      {
        int pointIndex = xPointIndex*num1DPts + yPointIndex;
        double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
        double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
        refPoints(pointIndex,0) = x;
        refPoints(pointIndex,1) = y;
      }
    }

    basisCache->setRefCellPoints(refPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCacheToo);
    Intrepid::FieldContainer<double> computedValues(numCells,numPoints);

    this->solutionValues(computedValues, trialID, basisCache);
    const Intrepid::FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();

    for (int cellIndex=0; cellIndex<numCells; cellIndex++ )
    {
      for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++)
      {
        int yPointIndex = 0;
        int pointIndex = xPointIndex*num1DPts + yPointIndex;
        fout << "x{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<")=" << (*physicalPoints)(cellIndex,pointIndex,0) << ";" << endl;
      }
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++)
      {
        int xPointIndex = 0;
        int pointIndex = xPointIndex*num1DPts + yPointIndex;
        fout << "y{"<<globalCellInd+cellIndex<< "}("<<yPointIndex+1<<")=" << (*physicalPoints)(cellIndex,pointIndex,1) << ";" << endl;
      }
    }

    for (int cellIndex=0; cellIndex < numCells; cellIndex++)
    {
      for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++)
      {
        for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++)
        {
          int ptIndex = xPointIndex*num1DPts + yPointIndex;
          fout << "z{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<","<<yPointIndex+1<<")=" << computedValues(cellIndex,ptIndex) << ";" << endl;
        }
      }
    }

    //    // NOW loop over all cells to write solution to file
    //    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
    //      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
    //
    //        // for some odd reason, I cannot compute the ref-to-phys map for more than 1 point at a time
    //        int numPoints = 1;
    //        Intrepid::FieldContainer<double> refPoints(numPoints,spaceDim);
    //        double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
    //        double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
    //        refPoints(0,0) = x;
    //        refPoints(0,1) = y;
    //
    //        // map side cubature points in reference parent cell domain to physical space
    //        Intrepid::FieldContainer<double> physicalPoints(numCells, numPoints, spaceDim);
    //        CellTools::mapToPhysicalFrame(physicalPoints, refPoints, physicalCellNodes, cellTopo);
    //
    //        cout << "physicalPoints:\n" <<  physicalPoints;
    //
    //        Intrepid::FieldContainer<double> computedValues(numCells,numPoints); // first arg = 1 cell only
    //        solutionValues(computedValues, elemTypePtr, trialID, physicalPoints);
    //
    //        for (int cellIndex=0;cellIndex < numCells;cellIndex++){
    //          fout << "x{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<")=" << physicalPoints(cellIndex,0,0) << ";" << endl;
    //          fout << "y{"<<globalCellInd+cellIndex<< "}("<<yPointIndex+1<<")=" << physicalPoints(cellIndex,0,1) << ";" << endl;
    //          fout << "z{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<","<<yPointIndex+1<<")=" << computedValues(cellIndex,0) << ";" << endl;
    //        }
    //      }
    //    }
    globalCellInd+=numCells;

  } //end of element type loop
  fout.close();
}

template <typename Scalar>
void TSolution<Scalar>::writeFluxesToFile(int trialID, const string &filePath)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"This method is no longer supported.  Use HDF5Exporter instead.");
}

template <typename Scalar>
double TSolution<Scalar>::totalTimeApplyJumpTerms()
{
  return _totalTimeApplyJumpTerms;
}

template <typename Scalar>
double TSolution<Scalar>::totalTimeLocalStiffness()
{
  return _totalTimeLocalStiffness;
}

template <typename Scalar>
double TSolution<Scalar>::totalTimeGlobalAssembly()
{
  return _totalTimeGlobalAssembly;
}

template <typename Scalar>
double TSolution<Scalar>::totalTimeBCImposition()
{
  return _totalTimeBCImposition;
}

template <typename Scalar>
double TSolution<Scalar>::totalTimeSolve()
{
  return _totalTimeSolve;
}

template <typename Scalar>
double TSolution<Scalar>::totalTimeDistributeSolution()
{
  return _totalTimeDistributeSolution;
}

template <typename Scalar>
double TSolution<Scalar>::meanTimeApplyJumpTerms()
{
  return _meanTimeApplyJumpTerms;
}

template <typename Scalar>
double TSolution<Scalar>::meanTimeLocalStiffness()
{
  return _meanTimeLocalStiffness;
}

template <typename Scalar>
double TSolution<Scalar>::meanTimeGlobalAssembly()
{
  return _meanTimeGlobalAssembly;
}

template <typename Scalar>
double TSolution<Scalar>::meanTimeBCImposition()
{
  return _meanTimeBCImposition;
}

template <typename Scalar>
double TSolution<Scalar>::meanTimeSolve()
{
  return _meanTimeSolve;
}

template <typename Scalar>
double TSolution<Scalar>::meanTimeDistributeSolution()
{
  return _meanTimeDistributeSolution;
}

template <typename Scalar>
double TSolution<Scalar>::maxTimeApplyJumpTerms()
{
  return _maxTimeApplyJumpTerms;
}

template <typename Scalar>
double TSolution<Scalar>::maxTimeLocalStiffness()
{
  return _maxTimeLocalStiffness;
}

template <typename Scalar>
double TSolution<Scalar>::maxTimeGlobalAssembly()
{
  return _maxTimeGlobalAssembly;
}

template <typename Scalar>
double TSolution<Scalar>::maxTimeBCImposition()
{
  return _maxTimeBCImposition;
}

template <typename Scalar>
double TSolution<Scalar>::maxTimeSolve()
{
  return _maxTimeSolve;
}

template <typename Scalar>
double TSolution<Scalar>::maxTimeDistributeSolution()
{
  return _maxTimeDistributeSolution;
}

template <typename Scalar>
double TSolution<Scalar>::minTimeApplyJumpTerms()
{
  return _minTimeApplyJumpTerms;
}

template <typename Scalar>
double TSolution<Scalar>::minTimeLocalStiffness()
{
  return _minTimeLocalStiffness;
}

template <typename Scalar>
double TSolution<Scalar>::minTimeGlobalAssembly()
{
  return _minTimeGlobalAssembly;
}

template <typename Scalar>
double TSolution<Scalar>::minTimeBCImposition()
{
  return _minTimeBCImposition;
}

template <typename Scalar>
double TSolution<Scalar>::minTimeSolve()
{
  return _minTimeSolve;
}

template <typename Scalar>
double TSolution<Scalar>::minTimeDistributeSolution()
{
  return _minTimeDistributeSolution;
}

template <typename Scalar>
Epetra_Map TSolution<Scalar>::getPartitionMap()
{
  Epetra_CommPtr Comm = _mesh->Comm();
  int rank = Comm->MyPID();

  vector<int> zeroMeanConstraints = getZeroMeanConstraints();
  GlobalIndexType numGlobalDofs = _dofInterpreter->globalDofCount();
  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
  int numZMCDofs = _zmcsAsRankOneUpdate ? 0 : zeroMeanConstraints.size();

  Epetra_Map partMap = getPartitionMap(rank, myGlobalIndicesSet,numGlobalDofs,numZMCDofs,Comm.get());
  return partMap;
}

template <typename Scalar>
Epetra_Map TSolution<Scalar>::getPartitionMapSolutionDofsOnly()   // omits lagrange multipliers, ZMCs, etc.
{
  Epetra_Map partMapWithZMC = getPartitionMap();
  vector<int> myGlobalIndices(partMapWithZMC.NumMyElements());
  partMapWithZMC.MyGlobalElements(&myGlobalIndices[0]);
  GlobalIndexType numGlobalDofs = _dofInterpreter->globalDofCount();
  vector<int> myGlobalDofs;
  for (vector<int>::iterator myEntry = myGlobalIndices.begin(); myEntry != myGlobalIndices.end(); myEntry++)
  {
    if (*myEntry < numGlobalDofs)
    {
      myGlobalDofs.push_back(*myEntry);
    }
  }
  int indexBase = 0;
  Epetra_Map partMap(numGlobalDofs, myGlobalDofs.size(), &myGlobalDofs[0], indexBase, partMapWithZMC.Comm());
  return partMap;
}

template <typename Scalar>
Epetra_Map TSolution<Scalar>::getPartitionMap(PartitionIndexType rank, set<GlobalIndexType> & myGlobalIndicesSet, GlobalIndexType numGlobalDofs,
    int zeroMeanConstraintsSize, Epetra_Comm* Comm )
{
  int numGlobalLagrange = _lagrangeConstraints->numGlobalConstraints();
  const set<GlobalIndexType>* cellIDsInPartition = &_mesh->globalDofAssignment()->cellsInPartition(rank);
  IndexType numMyElements = cellIDsInPartition->size();
  int numElementLagrange = _lagrangeConstraints->numElementConstraints() * numMyElements;
  int globalNumElementLagrange = _lagrangeConstraints->numElementConstraints() * _mesh->numActiveElements();

  // ordering is:
  // - regular dofs
  // - element lagrange
  // - global lagrange
  // - zero-mean constraints

  // determine the local dofs we have, and what their global indices are:
  int localDofsSize = myGlobalIndicesSet.size() + numElementLagrange;
  if (rank == 0)
  {
    // global Lagrange and zero-mean constraints belong to rank 0
    localDofsSize += zeroMeanConstraintsSize + numGlobalLagrange;
  }

  GlobalIndexTypeToCast *myGlobalIndices;
  if (localDofsSize!=0)
  {
    myGlobalIndices = new GlobalIndexTypeToCast[ localDofsSize ];
  }
  else
  {
    myGlobalIndices = NULL;
  }

  // copy from set object into the allocated array
  GlobalIndexType offset = 0;
  for (set<GlobalIndexType>::iterator indexIt = myGlobalIndicesSet.begin(); indexIt != myGlobalIndicesSet.end(); indexIt++ )
  {
    myGlobalIndices[offset++] = *indexIt;
  }
  GlobalIndexType cellOffset = _mesh->activeCellOffset() * _lagrangeConstraints->numElementConstraints();
  GlobalIndexType globalIndex = cellOffset + numGlobalDofs;
  for (int elemLagrangeIndex=0; elemLagrangeIndex<_lagrangeConstraints->numElementConstraints(); elemLagrangeIndex++)
  {
    for (IndexType cellIndex=0; cellIndex<numMyElements; cellIndex++)
    {
      myGlobalIndices[offset++] = globalIndex++;
    }
  }

  if ( rank == 0 )
  {
    // set up the zmcs and global Lagrange constraints, which come at the end...
    for (int i=0; i<numGlobalLagrange; i++)
    {
      myGlobalIndices[offset++] = i + numGlobalDofs + globalNumElementLagrange;
    }
    for (int i=0; i<zeroMeanConstraintsSize; i++)
    {
      myGlobalIndices[offset++] = i + numGlobalDofs + globalNumElementLagrange + numGlobalLagrange;
    }
  }

  if (offset != localDofsSize)
  {
    cout << "WARNING: Apparent internal error in TSolution<Scalar>::getPartitionMap.  # entries filled in myGlobalDofIndices does not match its size...\n";
  }

  int totalRows = numGlobalDofs + globalNumElementLagrange + numGlobalLagrange + zeroMeanConstraintsSize;

  int indexBase = 0;
  //cout << "process " << rank << " about to construct partMap.\n";
  //Epetra_Map partMap(-1, localDofsSize, myGlobalIndices, indexBase, Comm);
//  cout << "process " << rank << " about to construct partMap; totalRows = " << totalRows;
//  cout << "; localDofsSize = " << localDofsSize << ".\n";
  Epetra_Map partMap(totalRows, localDofsSize, myGlobalIndices, indexBase, *Comm);

  if (localDofsSize!=0)
  {
    delete[] myGlobalIndices;
  }
  return partMap;
}

template <typename Scalar>
MapPtr TSolution<Scalar>::getPartitionMap2()
{
  Teuchos_CommPtr Comm = _mesh->TeuchosComm();
  int rank = Comm->getRank();
  
  vector<int> zeroMeanConstraints = getZeroMeanConstraints();
  GlobalIndexType numGlobalDofs = _dofInterpreter->globalDofCount();
  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
  int numZMCDofs = _zmcsAsRankOneUpdate ? 0 : zeroMeanConstraints.size();

  MapPtr partMap = getPartitionMap2(rank, myGlobalIndicesSet,numGlobalDofs,numZMCDofs,Comm);
  return partMap;
}

// MapPtr TSolution<Scalar>::getPartitionMapSolutionDofsOnly2() { // omits lagrange multipliers, ZMCs, etc.
//   MapPtr partMapWithZMC = getPartitionMap2();
//   vector<int> myGlobalIndices(partMapWithZMC.NumMyElements());
//   partMapWithZMC.MyGlobalElements(&myGlobalIndices[0]);
//   GlobalIndexType numGlobalDofs = _dofInterpreter->globalDofCount();
//   vector<int> myGlobalDofs;
//   for (vector<int>::iterator myEntry = myGlobalIndices.begin(); myEntry != myGlobalIndices.end(); myEntry++) {
//     if (*myEntry < numGlobalDofs) {
//       myGlobalDofs.push_back(*myEntry);
//     }
//   }
//   int indexBase = 0;
//   MapPtr partMap(numGlobalDofs, myGlobalDofs.size(), &myGlobalDofs[0], indexBase, partMapWithZMC.getComm());
//   return partMap;
// }

template <typename Scalar>
MapPtr TSolution<Scalar>::getPartitionMap2(PartitionIndexType rank, set<GlobalIndexType> & myGlobalIndicesSet, GlobalIndexType numGlobalDofs,
    int zeroMeanConstraintsSize, Teuchos_CommPtr Comm )
{
  int numGlobalLagrange = _lagrangeConstraints->numGlobalConstraints();
  vector< ElementPtr > elements = _mesh->elementsInPartition(rank);
  IndexType numMyElements = elements.size();
  int numElementLagrange = _lagrangeConstraints->numElementConstraints() * numMyElements;
  int globalNumElementLagrange = _lagrangeConstraints->numElementConstraints() * _mesh->numActiveElements();

  // ordering is:
  // - regular dofs
  // - element lagrange
  // - global lagrange
  // - zero-mean constraints

  // determine the local dofs we have, and what their global indices are:
  int localDofsSize = myGlobalIndicesSet.size() + numElementLagrange;
  if (rank == 0)
  {
    // global Lagrange and zero-mean constraints belong to rank 0
    localDofsSize += zeroMeanConstraintsSize + numGlobalLagrange;
  }

  GlobalIndexType *myGlobalIndices;
  if (localDofsSize!=0)
  {
    myGlobalIndices = new GlobalIndexType[ localDofsSize ];
  }
  else
  {
    myGlobalIndices = NULL;
  }

  // copy from set object into the allocated array
  GlobalIndexType offset = 0;
  for (set<GlobalIndexType>::iterator indexIt = myGlobalIndicesSet.begin(); indexIt != myGlobalIndicesSet.end(); indexIt++ )
  {
    myGlobalIndices[offset++] = *indexIt;
  }
  GlobalIndexType cellOffset = _mesh->activeCellOffset() * _lagrangeConstraints->numElementConstraints();
  GlobalIndexType globalIndex = cellOffset + numGlobalDofs;
  for (int elemLagrangeIndex=0; elemLagrangeIndex<_lagrangeConstraints->numElementConstraints(); elemLagrangeIndex++)
  {
    for (IndexType cellIndex=0; cellIndex<numMyElements; cellIndex++)
    {
      myGlobalIndices[offset++] = globalIndex++;
    }
  }

  if ( rank == 0 )
  {
    // set up the zmcs and global Lagrange constraints, which come at the end...
    for (int i=0; i<numGlobalLagrange; i++)
    {
      myGlobalIndices[offset++] = i + numGlobalDofs + globalNumElementLagrange;
    }
    for (int i=0; i<zeroMeanConstraintsSize; i++)
    {
      myGlobalIndices[offset++] = i + numGlobalDofs + globalNumElementLagrange + numGlobalLagrange;
    }
  }

  if (offset != localDofsSize)
  {
    cout << "WARNING: Apparent internal error in TSolution<Scalar>::getPartitionMap.  # entries filled in myGlobalDofIndices does not match its size...\n";
  }

  int totalRows = numGlobalDofs + globalNumElementLagrange + numGlobalLagrange + zeroMeanConstraintsSize;

  int indexBase = 0;
  const Teuchos::ArrayView<const GlobalIndexType> rankGlobalIndices(myGlobalIndices, localDofsSize);
  MapPtr partMap = Teuchos::rcp( new Tpetra::Map<IndexType,GlobalIndexType>(totalRows, rankGlobalIndices, indexBase, Comm) );

  if (localDofsSize!=0)
  {
    delete[] myGlobalIndices;
  }
  return partMap;
}

template <typename Scalar>
void TSolution<Scalar>::processSideUpgrades( const map<GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades )
{
  set<GlobalIndexType> cellIDsToSkip; //empty
  processSideUpgrades(cellSideUpgrades,cellIDsToSkip);
}

template <typename Scalar>
void TSolution<Scalar>::processSideUpgrades( const map<GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades, const set<GlobalIndexType> &cellIDsToSkip )
{
  for (map<GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > >::const_iterator upgradeIt = cellSideUpgrades.begin();
       upgradeIt != cellSideUpgrades.end(); upgradeIt++)
  {
    GlobalIndexType cellID = upgradeIt->first;
    if (cellIDsToSkip.find(cellID) != cellIDsToSkip.end() ) continue;
    if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end())
      continue; // no previous solution for this cell
    DofOrderingPtr oldTrialOrdering = (upgradeIt->second).first->trialOrderPtr;
    DofOrderingPtr newTrialOrdering = (upgradeIt->second).second->trialOrderPtr;
    Intrepid::FieldContainer<Scalar> newCoefficients(newTrialOrdering->totalDofs());
    newTrialOrdering->copyLikeCoefficients( newCoefficients, oldTrialOrdering, _solutionForCellIDGlobal[cellID] );
    //    cout << "processSideUpgrades: setting solution for cell ID " << cellID << endl;
    _solutionForCellIDGlobal[cellID] = newCoefficients;
  }
}

template <typename Scalar>
void TSolution<Scalar>::projectOntoMesh(const map<int, TFunctionPtr<Scalar> > &functionMap)  // map: trialID -> function
{
  if (_lhsVector.get()==NULL)
  {
    initializeLHSVector();
  }

  set<GlobalIndexType> cellIDs = _mesh->globalDofAssignment()->cellsInPartition(-1);
  for (GlobalIndexType cellID : cellIDs)
  {
    projectOntoCell(functionMap,cellID);
  }
}

template <typename Scalar>
void TSolution<Scalar>::projectOntoCell(const map<int, TFunctionPtr<Scalar> > &functionMap, GlobalIndexType cellID, int side)
{
  vector<GlobalIndexType> cellIDs(1,cellID);

  VarFactoryPtr vf;
  if (_bf != Teuchos::null)
    vf = _bf->varFactory();
  else
    vf = _mesh->bilinearForm()->varFactory();

  for (typename map<int, TFunctionPtr<Scalar> >::const_iterator functionIt = functionMap.begin(); functionIt !=functionMap.end(); functionIt++)
  {
    int trialID = functionIt->first;

    bool fluxOrTrace;
    if (_bf != Teuchos::null)
      fluxOrTrace = _bf->isFluxOrTrace(trialID);
    else
      fluxOrTrace = _mesh->bilinearForm()->isFluxOrTrace(trialID);
    VarPtr trialVar = vf->trial(trialID);
    TFunctionPtr<Scalar> function = functionIt->second;

    bool testVsTest = false; // in fact it's more trial vs trial, but this just means we'll over-integrate a bit
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID, testVsTest, _cubatureEnrichmentDegree);
    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);

    if (fluxOrTrace)
    {
      vector<int> sides;
      if (side == -1)   // handle all sides
      {
        sides = elemTypePtr->trialOrderPtr->getSidesForVarID(trialID);
      }
      else
      {
        sides = {side};
      }
      for (int sideIndex : sides)
      {
//        if (! elemTypePtr->trialOrderPtr->hasBasisEntry(trialID, sideIndex)) continue;
        BasisPtr basis = elemTypePtr->trialOrderPtr->getBasis(trialID, sideIndex);
        Intrepid::FieldContainer<Scalar> basisCoefficients(1,basis->getCardinality());
        Projector<Scalar>::projectFunctionOntoBasis(basisCoefficients, function, basis, basisCache->getSideBasisCache(sideIndex));
        basisCoefficients.resize(basis->getCardinality());

//        { // DEBUGGING
//          if ((sideIndex==2) || (sideIndex == 3)) {
//            cout << "cell " << cellID << ", side " << sideIndex << ":\n";
//            cout << "function: " << function->displayString() << endl;
//
//            BasisCachePtr sideCache = basisCache->getSideBasisCache(sideIndex);
//
//            cout << "basisCoefficients:\n" << basisCoefficients;
//            cout << "physicalCubaturePoints:\n" << sideCache->getPhysicalCubaturePoints();
//
//            int numCells = 1;
//            Intrepid::FieldContainer<double> values(numCells, basisCache->getSideBasisCache(sideIndex)->getPhysicalCubaturePoints().dimension(1));
//            function->values(values, sideCache);
//
//            cout << "function values:\n" << values;
//          }
//        }

        // at present, we understand it to be caller's responsibility to include parity in Function if the varType is a flux.
        // if we wanted to change that semantic, we'd use the below.
//        if ((_mesh->parityForSide(cellID, sideIndex) == -1) && (trialVar->varType()==FLUX)) {
//          SerialDenseWrapper::multiplyFCByWeight(basisCoefficients, -1);
//        }

        setSolnCoeffsForCellID(basisCoefficients,cellID,trialID,sideIndex);
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(side != -1, std::invalid_argument, "sideIndex for fields must = -1");
      if (! elemTypePtr->trialOrderPtr->hasBasisEntry(trialID, VOLUME_INTERIOR_SIDE_ORDINAL))   // DofOrdering uses side VOLUME_INTERIOR_SIDE_ORDINAL for fields...
      {
        continue;
      }

      BasisPtr basis = elemTypePtr->trialOrderPtr->getBasis(trialID);
      Intrepid::FieldContainer<Scalar> basisCoefficients(1,basis->getCardinality());
      Projector<Scalar>::projectFunctionOntoBasis(basisCoefficients, function, basis, basisCache);
      basisCoefficients.resize(basis->getCardinality());
      setSolnCoeffsForCellID(basisCoefficients,cellID,trialID);
    }
  }
}

template <typename Scalar>
void TSolution<Scalar>::projectFieldVariablesOntoOtherSolution(Teuchos::RCP< TSolution<Scalar> > otherSoln)
{
  vector< VarPtr > fieldVars = _mesh->bilinearForm()->varFactory()->fieldVars();

  Teuchos::RCP< TSolution<Scalar> > thisPtr = Teuchos::rcp(this, false);
  map<int, TFunctionPtr<Scalar> > solnMap = PreviousSolutionFunction<Scalar>::functionMap(fieldVars, thisPtr);
  otherSoln->projectOntoMesh(solnMap);
}

template <typename Scalar>
void TSolution<Scalar>::projectOldCellOntoNewCells(GlobalIndexType cellID,
    ElementTypePtr oldElemType,
    const vector<GlobalIndexType> &childIDs)
{
//  int rank = Teuchos::GlobalMPISession::getRank();

  if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end())
  {
//    cout << "on rank " << rank << ", no solution for " << cellID << "; skipping projection onto children.\n";
    return; // zero solution on cell
  }
//  cout << "on rank " << rank << ", projecting " << cellID << " data onto children.\n";
  const Intrepid::FieldContainer<Scalar>* oldData = &_solutionForCellIDGlobal[cellID];
//  cout << "cell " << cellID << " data: \n" << *oldData;
  projectOldCellOntoNewCells(cellID, oldElemType, *oldData, childIDs);
}

template <typename Scalar>
void TSolution<Scalar>::projectOldCellOntoNewCells(GlobalIndexType cellID,
    ElementTypePtr oldElemType,
    const Intrepid::FieldContainer<Scalar> &oldData,
    const vector<GlobalIndexType> &childIDs)
{
  VarFactoryPtr vf;
  if (_bf != Teuchos::null)
    vf = _bf->varFactory();
  else
    vf = _mesh->bilinearForm()->varFactory();

  DofOrderingPtr oldTrialOrdering = oldElemType->trialOrderPtr;
  set<int> trialIDs = oldTrialOrdering->getVarIDs();

  TEUCHOS_TEST_FOR_EXCEPTION(oldTrialOrdering->totalDofs() != oldData.size(), std::invalid_argument,
                             "oldElemType trial space does not match old data coefficients size");
  map<int, TFunctionPtr<Scalar> > fieldMap;

  CellPtr parentCell = _mesh->getTopology()->getCell(cellID);
  int dummyCubatureDegree = 1;
  BasisCachePtr parentRefCellCache = BasisCache::basisCacheForReferenceCell(parentCell->topology(), dummyCubatureDegree);

//   cout << "projecting from cell " << cellID << " onto its children.\n";

  for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    if (oldTrialOrdering->getSidesForVarID(trialID).size() == 1)   // field variable, the only kind we honor right now
    {
      BasisPtr basis = oldTrialOrdering->getBasis(trialID);
      int basisCardinality = basis->getCardinality();
      Intrepid::FieldContainer<Scalar> basisCoefficients(basisCardinality);

      for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++)
      {
        int dofIndex = oldElemType->trialOrderPtr->getDofIndex(trialID, dofOrdinal);
        basisCoefficients(dofOrdinal) = oldData(dofIndex);
      }

//       cout << "basisCoefficients for parent volume trialID " << trialID << ":\n" << basisCoefficients;

      TFunctionPtr<Scalar> oldTrialFunction = Teuchos::rcp( new BasisSumFunction(basis, basisCoefficients, parentRefCellCache) );
      fieldMap[trialID] = oldTrialFunction;
    }
  }

  map<int,TFunctionPtr<Scalar>> interiorTraceMap; // functions to use on parent interior to represent traces there
  for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    if (oldTrialOrdering->getSidesForVarID(trialID).size() != 1)   // trace (flux) variable
    {
      VarPtr var = vf->trialVars().find(trialID)->second;

      TLinearTermPtr<Scalar> termTraced = var->termTraced();
      if (termTraced.get() != NULL)
      {
        TFunctionPtr<Scalar> fieldTrace = termTraced->evaluate(fieldMap, true) + termTraced->evaluate(fieldMap, false);
        interiorTraceMap[trialID] = fieldTrace;
      }
    }
  }

  int sideDim = _mesh->getTopology()->getDimension() - 1;

  int sideCount = parentCell->topology()->getSideCount();
  vector< map<int, TFunctionPtr<Scalar>> > traceMap(sideCount);
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    CellTopoPtr sideTopo = parentCell->topology()->getSubcell(sideDim, sideOrdinal);
    BasisCachePtr parentSideTopoBasisCache = BasisCache::basisCacheForReferenceCell(sideTopo, dummyCubatureDegree);
    for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++)
    {
      int trialID = *trialIDIt;
      if (oldTrialOrdering->getSidesForVarID(trialID).size() != 1)   // trace (flux) variable
      {
        if (!oldTrialOrdering->hasBasisEntry(trialID, sideOrdinal)) continue;
        BasisPtr basis = oldTrialOrdering->getBasis(trialID, sideOrdinal);
        int basisCardinality = basis->getCardinality();
        Intrepid::FieldContainer<Scalar> basisCoefficients(basisCardinality);

        for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++)
        {
          int dofIndex = oldElemType->trialOrderPtr->getDofIndex(trialID, dofOrdinal, sideOrdinal);
          basisCoefficients(dofOrdinal) = oldData(dofIndex);
        }
        TFunctionPtr<Scalar> oldTrialFunction = Teuchos::rcp( new BasisSumFunction(basis, basisCoefficients, parentSideTopoBasisCache) );
        traceMap[sideOrdinal][trialID] = oldTrialFunction;
      }
    }
  }

  int parent_p_order = _mesh->getElementType(cellID)->trialOrderPtr->maxBasisDegree();

  for (int childOrdinal=0; childOrdinal < childIDs.size(); childOrdinal++)
  {
    GlobalIndexType childID = childIDs[childOrdinal];
    if (childID == -1) continue; // indication we should skip this child...
    CellPtr childCell = _mesh->getTopology()->getCell(childID);
    ElementTypePtr childType = _mesh->getElementType(childID);
    int childSideCount = childCell->getSideCount();

    int child_p_order = _mesh->getElementType(childID)->trialOrderPtr->maxBasisDegree();
    int cubatureDegree = parent_p_order + child_p_order;

    BasisCachePtr volumeBasisCache;
    vector<BasisCachePtr> sideBasisCache(childSideCount);

    if (parentCell->children().size() > 0)
    {
      RefinementBranch refBranch(1,make_pair(parentCell->refinementPattern().get(), childOrdinal));
      volumeBasisCache = BasisCache::basisCacheForRefinedReferenceCell(childCell->topology(), cubatureDegree, refBranch, true);
      for (int sideOrdinal = 0; sideOrdinal < childSideCount; sideOrdinal++)
      {
        CellTopoPtr sideTopo = childCell->topology()->getSubcell(sideDim, sideOrdinal);
        unsigned parentSideOrdinal = (childID==cellID) ? sideOrdinal
                                     : parentCell->refinementPattern()->mapSubcellOrdinalFromChildToParent(childOrdinal, sideDim, sideOrdinal);

        RefinementBranch sideBranch;
        if (parentSideOrdinal != -1)
          sideBranch = RefinementPattern::subcellRefinementBranch(refBranch, sideDim, parentSideOrdinal);
        if (sideBranch.size()==0)
        {
          sideBasisCache[sideOrdinal] = BasisCache::basisCacheForReferenceCell(sideTopo, cubatureDegree);
        }
        else
        {
          sideBasisCache[sideOrdinal] = BasisCache::basisCacheForRefinedReferenceCell(sideTopo, cubatureDegree, sideBranch);
        }
      }
    }
    else
    {
      volumeBasisCache = BasisCache::basisCacheForReferenceCell(childCell->topology(), cubatureDegree, true);
      for (int sideOrdinal = 0; sideOrdinal < childSideCount; sideOrdinal++)
      {
        CellTopoPtr sideTopo = childCell->topology()->getSubcell(sideDim, sideOrdinal);
        sideBasisCache[sideOrdinal] = BasisCache::basisCacheForReferenceCell(sideTopo, cubatureDegree);
      }
    }

    // (re)initialize the FieldContainer storing the solution--element type may have changed (in case of p-refinement)
    _solutionForCellIDGlobal[childID] = Intrepid::FieldContainer<Scalar>(childType->trialOrderPtr->totalDofs());
    // project fields
    Intrepid::FieldContainer<Scalar> basisCoefficients;
    for (typename map<int,TFunctionPtr<Scalar>>::iterator fieldFxnIt=fieldMap.begin(); fieldFxnIt != fieldMap.end(); fieldFxnIt++)
    {
      int varID = fieldFxnIt->first;
      TFunctionPtr<Scalar> fieldFxn = fieldFxnIt->second;
      BasisPtr childBasis = childType->trialOrderPtr->getBasis(varID);
      basisCoefficients.resize(1,childBasis->getCardinality());
      Projector<Scalar>::projectFunctionOntoBasisInterpolating(basisCoefficients, fieldFxn, childBasis, volumeBasisCache);

//      cout << "projected basisCoefficients for child volume trialID " << varID << ":\n" << basisCoefficients;

      for (int basisOrdinal=0; basisOrdinal<basisCoefficients.size(); basisOrdinal++)
      {
        int dofIndex = childType->trialOrderPtr->getDofIndex(varID, basisOrdinal);
        _solutionForCellIDGlobal[childID][dofIndex] = basisCoefficients[basisOrdinal];
      }
    }

    // project traces and fluxes
    for (int sideOrdinal=0; sideOrdinal<childSideCount; sideOrdinal++)
    {
      unsigned parentSideOrdinal = (childID==cellID) ? sideOrdinal
                                   : parentCell->refinementPattern()->mapSubcellOrdinalFromChildToParent(childOrdinal, sideDim, sideOrdinal);

      map<int,TFunctionPtr<Scalar>>* traceMapForSide = (parentSideOrdinal != -1) ? &traceMap[parentSideOrdinal] : &interiorTraceMap;
      // which BasisCache to use depends on whether we want the BasisCache's notion of "physical" space to be in the volume or on the side:
      // we want it to be on the side if parent shares the side (and we therefore have proper trace data)
      // and on the volume in parent doesn't share the side (in which case we use the interior trace map).
      BasisCachePtr basisCacheForSide;
      
      if (parentSideOrdinal != -1)
      {
        basisCacheForSide = sideBasisCache[sideOrdinal];
      }
      else
      {
        basisCacheForSide = volumeBasisCache->getSideBasisCache(sideOrdinal);
      }
      basisCacheForSide->setCellSideParities(_mesh->cellSideParitiesForCell(childID));

      for (typename map<int,TFunctionPtr<Scalar>>::iterator traceFxnIt=traceMapForSide->begin(); traceFxnIt != traceMapForSide->end(); traceFxnIt++)
      {
        int varID = traceFxnIt->first;
        TFunctionPtr<Scalar> traceFxn = traceFxnIt->second;
        if (! childType->trialOrderPtr->hasBasisEntry(varID, sideOrdinal)) continue;
        BasisPtr childBasis = childType->trialOrderPtr->getBasis(varID, sideOrdinal);
        basisCoefficients.resize(1,childBasis->getCardinality());
        Projector<Scalar>::projectFunctionOntoBasisInterpolating(basisCoefficients, traceFxn, childBasis, basisCacheForSide);
        for (int basisOrdinal=0; basisOrdinal<basisCoefficients.size(); basisOrdinal++)
        {
          int dofIndex = childType->trialOrderPtr->getDofIndex(varID, basisOrdinal, sideOrdinal);
          _solutionForCellIDGlobal[childID][dofIndex] = basisCoefficients[basisOrdinal];
          // worth noting that as now set up, the "field traces" may stomp on the true traces, depending on in what order the sides
          // are mapped to global dof ordinals.  For right now, I'm not too worried about this.
        }
      }
    }
  }

  clearComputedResiduals(); // force recomputation of energy error (could do something more incisive, just computing the energy error for the new cells)
}

template <typename Scalar>
void TSolution<Scalar>::readFromFile(const string &filePath)
{
  ifstream fin(filePath.c_str());

  while (fin.good())
  {
    string refTypeStr;
    int cellID;

    string line;
    std::getline(fin, line, '\n');
    std::istringstream linestream(line);
    linestream >> cellID;

    if (_mesh->getElement(cellID).get() == NULL)
    {
      cout << "No cellID " << cellID << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Could not find cellID in solution file in mesh.");
    }
    ElementTypePtr elemType = _mesh->getElementType(cellID);
    int numDofsExpected = elemType->trialOrderPtr->totalDofs();

    if ( linestream.good() )
    {
      int numDofs;
      linestream >> numDofs;

      // TODO: check that numDofs is right for cellID.
      if (numDofsExpected != numDofs)
      {
        cout << "ERROR in readFromFile: expected cellID " << cellID << " to have " << numDofsExpected;
        cout << ", but found " << numDofs << " instead.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "wrong number of dofs for cell");
      }

      Intrepid::FieldContainer<Scalar> dofValues(numDofs);
      Scalar dofValue;
      int dofOrdinal = 0;
      while (linestream.good())
      {
        linestream >> dofValue;
        dofValues[dofOrdinal++] = dofValue;
      }

      _solutionForCellIDGlobal[cellID] = dofValues;
    }
  }
  fin.close();
}

template <typename Scalar>
Teuchos::RCP< TSolution<Scalar> > TSolution<Scalar>::solution(TBFPtr<Scalar> bf, MeshPtr mesh, TBCPtr<Scalar> bc, TRHSPtr<Scalar> rhs, TIPPtr<Scalar> ip )
{
  return Teuchos::rcp( new TSolution<Scalar>(bf, mesh,bc,rhs,ip) );
}

template <typename Scalar>
Teuchos::RCP< TSolution<Scalar> > TSolution<Scalar>::solution(MeshPtr mesh, TBCPtr<Scalar> bc, TRHSPtr<Scalar> rhs, TIPPtr<Scalar> ip )
{
  return Teuchos::rcp( new TSolution<Scalar>(mesh,bc,rhs,ip) );
}

template <typename Scalar>
void TSolution<Scalar>::writeToFile(const string &filePath)
{
  ofstream fout(filePath.c_str());

  for (typename map<GlobalIndexType, Intrepid::FieldContainer<Scalar> >::iterator solnEntryIt = _solutionForCellIDGlobal.begin();
       solnEntryIt != _solutionForCellIDGlobal.end(); solnEntryIt++)
  {
    GlobalIndexType cellID = solnEntryIt->first;
    Intrepid::FieldContainer<Scalar>* solnCoeffs = &(solnEntryIt->second);
    fout << cellID << " " << solnCoeffs->size() << " ";
    for (int i=0; i<solnCoeffs->size(); i++)
    {
      fout << (*solnCoeffs)[i] << " ";
    }
    fout << endl;
  }

  fout.close();
}

#ifdef HAVE_EPETRAEXT_HDF5
template <typename Scalar>
void TSolution<Scalar>::save(string meshAndSolutionPrefix)
{
  saveToHDF5(meshAndSolutionPrefix+".soln");
  mesh()->saveToHDF5(meshAndSolutionPrefix+".mesh");
}

template <typename Scalar>
Teuchos::RCP< TSolution<Scalar> > TSolution<Scalar>::load(TBFPtr<Scalar> bf, string meshAndSolutionPrefix)
{
  MeshPtr mesh = MeshFactory::loadFromHDF5(bf, meshAndSolutionPrefix+".mesh");
  Teuchos::RCP< TSolution<Scalar> > solution = TSolution<Scalar>::solution(mesh);
  solution->loadFromHDF5(meshAndSolutionPrefix+".soln");
  return solution;
}

template <typename Scalar>
void TSolution<Scalar>::saveToHDF5(string filename)
{
  // Note that the format here assumes *identical* partitioning of the mesh, since
  // global dof numbering depends on the mesh partitioning.
  
  // This is not an especially good idea.  It would be better (though not optimal,
  // in terms of storage size and conversion cost) to store local coefficients for
  // each cell.  One possibility would be to store *both* -- that would trade more
  // storage cost and the headache of figuring out whether it's safe to use the
  // vector representation, for the computational savings of not having to reconstruct
  // the global vector from the local representation.

  Epetra_CommPtr Comm = _mesh->Comm();
  
  EpetraExt::HDF5 hdf5(*Comm);
  hdf5.Create(filename);
  if (_lhsVector == Teuchos::null)
  {
    // then we'll save the zero solution.
    initializeLHSVector();
  }
  hdf5.Write("Solution", *_lhsVector);
  hdf5.Close();
}

template <typename Scalar>
void TSolution<Scalar>::loadFromHDF5(string filename)
{
  initializeLHSVector();
  
  Epetra_CommPtr Comm = _mesh->Comm();

  EpetraExt::HDF5 hdf5(*Comm);
  hdf5.Open(filename);
  Epetra_MultiVector *lhsVec;
  Epetra_Map partMap = getPartitionMap();
  hdf5.Read("Solution", partMap, lhsVec);

  Epetra_Import  solnImporter(_lhsVector->Map(), lhsVec->Map());
  _lhsVector->Import(*lhsVec, solnImporter, Insert);

  hdf5.Close();
  importSolution();
}
#endif

template <typename Scalar>
vector<int> TSolution<Scalar>::getZeroMeanConstraints()
{
  // determine any zero-mean constraints:
  vector< int > trialIDs;
  if (_bf != Teuchos::null)
    trialIDs = _bf->trialIDs();
  else
    trialIDs = _mesh->bilinearForm()->trialIDs();
  vector< int > zeroMeanConstraints;
  if (_bc.get()==NULL) return zeroMeanConstraints; //empty
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++)
  {
    int trialID = *trialIt;
    if (_bc->shouldImposeZeroMeanConstraint(trialID))
    {
      zeroMeanConstraints.push_back(trialID);
    }
  }
  return zeroMeanConstraints;
}

template <typename Scalar>
void TSolution<Scalar>::setGlobalSolutionFromCellLocalCoefficients()
{
  if (_lhsVector.get() == NULL)
  {
    initializeLHSVector();
    return; // initializeLHSVector() calls setGlobalSolutionFromCellLocalCoefficients(), so return now to avoid redundant execution of the below.
  }

  _lhsVector->PutScalar(0); // unclear whether this is redundant with constructor or not

  // set initial _lhsVector (initial guess for iterative solvers)
  set<GlobalIndexType> cellIDs = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    if (_solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end())
    {
      int localTrialDofCount = _mesh->getElementType(cellID)->trialOrderPtr->totalDofs();
      if (localTrialDofCount==_solutionForCellIDGlobal[cellID].size())   // guard against cases when solutions not registered with their meshes have their meshes p-refined beneath them.  In such a case, we'll just ignore the previous solution coefficients on the cell.
      {
        _dofInterpreter->interpretLocalCoefficients(cellID, _solutionForCellIDGlobal[cellID], *_lhsVector);
      }
    }
  }
}

template <typename Scalar>
void TSolution<Scalar>::setUseCondensedSolve(bool value, set<GlobalIndexType> offRankCellsToInclude)
{
  if (value)
  {
    if (_oldDofInterpreter.get()==NULL)
    {
      // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
      vector<int> trialIDs;
      if (_bf != Teuchos::null)
        trialIDs = _bf->trialIDs();
      else
        trialIDs = _mesh->bilinearForm()->trialIDs();

      set< int > fieldsToExclude;
      for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++)
      {
        int trialID = *trialIt;
        if (_bc->shouldImposeZeroMeanConstraint(trialID) || _bc->singlePointBC(trialID) )
        {
          fieldsToExclude.insert(trialID);
        }
      }

      // override reduceMemoryFootprint for now (since CondensedDofInterpreter doesn't yet support a true value)
      bool reduceMemoryFootprint = false;

      _oldDofInterpreter = _dofInterpreter;

      Teuchos::RCP<DofInterpreter> dofInterpreter = Teuchos::rcp(new CondensedDofInterpreter<Scalar>(_mesh, _ip, _rhs, _lagrangeConstraints.get(), fieldsToExclude, !reduceMemoryFootprint, offRankCellsToInclude) );

      setDofInterpreter(dofInterpreter);
    }
  }
  else
  {
    if (_oldDofInterpreter.get() != NULL)
    {
      setDofInterpreter(_oldDofInterpreter);
      _oldDofInterpreter = Teuchos::rcp((DofInterpreter*) NULL);
    }
  }
}

template <typename Scalar>
void TSolution<Scalar>::setZeroMeanConstraintRho(double value)
{
  _zmcRho = value;
}

template <typename Scalar>
double TSolution<Scalar>::zeroMeanConstraintRho()
{
  return _zmcRho;
}

template <typename Scalar>
bool TSolution<Scalar>::usesCondensedSolve() const
{
  if (_oldDofInterpreter.get() != NULL)   // proxy for having a condensation interpreter
  {
    CondensedDofInterpreter<Scalar>* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter<Scalar>*>(_dofInterpreter.get());
    if (condensedDofInterpreter != NULL)
    {
      return true;
    }
  }
  return false;
}

template <typename Scalar>
bool TSolution<Scalar>::getZMCsAsGlobalLagrange() const
{
  return _zmcsAsLagrangeMultipliers;
}

template <typename Scalar>
void TSolution<Scalar>::setZMCsAsGlobalLagrange(bool value)
{
  _zmcsAsLagrangeMultipliers = value;
}

namespace Camellia
{
template class TSolution<double>;
}
