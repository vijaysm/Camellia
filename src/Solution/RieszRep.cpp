#include "RieszRep.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"

#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"

#include "SerialDenseWrapper.h"
#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"

#include "MPIWrapper.h"

using namespace Intrepid;
using namespace Camellia;

template <typename Scalar>
TLinearTermPtr<Scalar> TRieszRep<Scalar>::getFunctional()
{
  return _functional;
}

template <typename Scalar>
MeshPtr TRieszRep<Scalar>::mesh()
{
  return _mesh;
}

template <typename Scalar>
map<GlobalIndexType,FieldContainer<Scalar> > TRieszRep<Scalar>::integrateFunctional()
{
  // NVR: changed this to only return integrated values for rank-local cells.

  map<GlobalIndexType,FieldContainer<Scalar> > cellRHS;
  set<GlobalIndexType> cellIDs = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt=cellIDs.begin(); cellIDIt !=cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();

    int cubEnrich = 0; // set to zero for release
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,cellID,true,cubEnrich);

    FieldContainer<Scalar> rhsValues(1,numTestDofs);
    _functional->integrate(rhsValues, testOrderingPtr, basisCache);

    FieldContainer<Scalar> rhsVals(numTestDofs);
    for (int i = 0; i<numTestDofs; i++)
    {
      rhsVals(i) = rhsValues(0,i);
    }
    cellRHS[cellID] = rhsVals;
  }
  return cellRHS;
}

template <typename Scalar>
void TRieszRep<Scalar>::computeRieszRep(int cubatureEnrichment)
{
  set<GlobalIndexType> cellIDs = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt=cellIDs.begin(); cellIDIt !=cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,cellID,true,cubatureEnrichment);

    FieldContainer<Scalar> rhsValues(1,numTestDofs);
    _functional->integrate(rhsValues, testOrderingPtr, basisCache);
    if (_printAll)
    {
      cout << "RieszRep: LinearTerm values for cell " << cellID << ":\n " << rhsValues << endl;
    }

    FieldContainer<Scalar> ipMatrix(1,numTestDofs,numTestDofs);
    _ip->computeInnerProductMatrix(ipMatrix,testOrderingPtr, basisCache);

    bool printOutRiesz = false;
    if (printOutRiesz)
    {
      cout << " ============================ In RIESZ ==========================" << endl;
      cout << "matrix: \n" << ipMatrix;
    }

    FieldContainer<Scalar> rieszRepDofs(numTestDofs,1);
    ipMatrix.resize(numTestDofs,numTestDofs);
    rhsValues.resize(numTestDofs,1);
    int success = SerialDenseWrapper::solveSystemUsingQR(rieszRepDofs, ipMatrix, rhsValues);

    if (success != 0)
    {
      cout << "TRieszRep<Scalar>::computeRieszRep: Solve FAILED with error: " << success << endl;
    }

//    rieszRepDofs.Multiply(true,rhsVectorCopy, normSq); // equivalent to e^T * R_V * e
    double normSquared = SerialDenseWrapper::dot(rieszRepDofs, rhsValues);
    _rieszRepNormSquared[cellID] = normSquared;

//    cout << "normSquared for cell " << cellID << ": " << _rieszRepNormSquared[cellID] << endl;

    if (printOutRiesz)
    {
      cout << "rhs: \n" << rhsValues;
      cout << "dofs: \n" << rieszRepDofs;
      cout << " ================================================================" << endl;
    }

    FieldContainer<Scalar> dofs(numTestDofs);
    for (int i = 0; i<numTestDofs; i++)
    {
      dofs(i) = rieszRepDofs(i,0);
    }
    _rieszRepDofs[cellID] = dofs;
  }
  if (_distributeDofs)
    distributeDofs();
  _repsNotComputed = false;
}

template <typename Scalar>
double TRieszRep<Scalar>::getNorm()
{

  if (_repsNotComputed)
  {
    cout << "Computing riesz rep dofs" << endl;
    computeRieszRep();
  }

  const set<GlobalIndexType>* myCells = &_mesh->cellIDsInPartition();
  
  double normSumLocal = 0.0;
  for (GlobalIndexType cellID : *myCells)
  {
    normSumLocal += _rieszRepNormSquared[cellID];
  }
  double normSumGlobal = 0.0;
  _mesh->Comm()->SumAll(&normSumLocal, &normSumGlobal, 1);
  return sqrt(normSumGlobal);
}

template <typename Scalar>
const map<GlobalIndexType,double> & TRieszRep<Scalar>::getNormsSquared()
{
  return _rieszRepNormSquared;
}

template <typename Scalar>
const map<GlobalIndexType,double> & TRieszRep<Scalar>::getNormsSquaredGlobal()
{
  if (_distributeDofs)
    return _rieszRepNormSquaredGlobal;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_distributeDofs must be true for getNormsSquaredGlobal()");
}

template <typename Scalar>
void TRieszRep<Scalar>::distributeDofs()
{
  int myRank = _mesh->Comm()->MyPID();
  int numRanks = _mesh->Comm()->NumProc();

  // the code below could stand to be reworked; I'm pretty sure this is not the best way to distribute the data, and it would also be best to get rid of the iteration over the global set of active elements.  But a similar point could be made about this method as a whole: do we really need to distribute all the dofs to every rank?  It may be best to eliminate this method altogether. [NVR]

  vector<GlobalIndexType> cellIDsByPartitionOrdering;
  for (int rank=0; rank<numRanks; rank++)
  {
    set<GlobalIndexType> cellIDsForRank = _mesh->globalDofAssignment()->cellsInPartition(rank);
    cellIDsByPartitionOrdering.insert(cellIDsByPartitionOrdering.end(), cellIDsForRank.begin(), cellIDsForRank.end());
  }
  // determine inverse map:
  map<GlobalIndexType,int> ordinalForCellID;
  for (int ordinal=0; ordinal<cellIDsByPartitionOrdering.size(); ordinal++)
  {
    GlobalIndexType cellID = cellIDsByPartitionOrdering[ordinal];
    ordinalForCellID[cellID] = ordinal;
//    cout << "ordinalForCellID[" << cellID << "] = " << ordinal << endl;
  }

  for (int cellOrdinal=0; cellOrdinal<cellIDsByPartitionOrdering.size(); cellOrdinal++)
  {
    GlobalIndexType cellID = cellIDsByPartitionOrdering[cellOrdinal];
    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numDofs = testOrderingPtr->totalDofs();

    int cellIDPartition = _mesh->partitionForCellID(cellID);
    bool isInPartition = (cellIDPartition == myRank);

    int numMyDofs;
    FieldContainer<Scalar> dofs(numDofs);
    if (isInPartition)   // if in partition
    {
      numMyDofs = numDofs;
      dofs = _rieszRepDofs[cellID];
    }
    else
    {
      numMyDofs = 0;
    }

    Epetra_Map dofMap(numDofs,numMyDofs,0,*_mesh->Comm());
    Epetra_Vector distributedRieszDofs(dofMap);
    if (isInPartition)
    {
      for (int i = 0; i<numMyDofs; i++) // shouldn't activate on off-proc partitions
      {
        distributedRieszDofs.ReplaceGlobalValues(1,&dofs(i),&i);
      }
    }
    Epetra_Map importMap(numDofs,numDofs,0,*_mesh->Comm()); // every proc should own their own copy of the dofs
    Epetra_Import testDofImporter(importMap, dofMap);
    Epetra_Vector globalRieszDofs(importMap);
    globalRieszDofs.Import(distributedRieszDofs, testDofImporter, Insert);
    if (!isInPartition)
    {
      for (int i = 0; i<numDofs; i++)
      {
        dofs(i) = globalRieszDofs[i];
      }
    }
    _rieszRepDofsGlobal[cellID] = dofs;
//    { // debugging
//      ostringstream cellIDlabel;
//      cellIDlabel << "cell " << cellID << " _rieszRepDofsGlobal, after global import";
//      TestSuite::serializeOutput(cellIDlabel.str(), _rieszRepDofsGlobal[cellID]);
//    }
  }

  // distribute norms as well
  GlobalIndexType numElems = _mesh->numActiveElements();
  set<GlobalIndexType> rankLocalCellIDs = _mesh->cellIDsInPartition();
  IndexType numMyElems = rankLocalCellIDs.size();
  GlobalIndexType myElems[numMyElems];
  // build cell index
  GlobalIndexType myCellOrdinal = 0;

  double rankLocalRieszNorms[numMyElems];

  for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCellIDs.begin(); cellIDIt != rankLocalCellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    myElems[myCellOrdinal] = ordinalForCellID[cellID];
    rankLocalRieszNorms[myCellOrdinal] = _rieszRepNormSquared[cellID];
    myCellOrdinal++;
  }
  Epetra_Map normMap((GlobalIndexTypeToCast)numElems,(int)numMyElems,(GlobalIndexTypeToCast *)myElems,(GlobalIndexTypeToCast)0,*_mesh->Comm());

  Epetra_Vector distributedRieszNorms(normMap);
  int err = distributedRieszNorms.ReplaceGlobalValues(numMyElems,rankLocalRieszNorms,(GlobalIndexTypeToCast *)myElems);
  if (err != 0)
  {
    cout << "TRieszRep<Scalar>::distributeDofs(): on rank" << myRank << ", ReplaceGlobalValues returned error code " << err << endl;
  }

  Epetra_Map normImportMap((GlobalIndexTypeToCast)numElems,(GlobalIndexTypeToCast)numElems,0,*_mesh->Comm());
  Epetra_Import normImporter(normImportMap,normMap);
  Epetra_Vector globalNorms(normImportMap);
  globalNorms.Import(distributedRieszNorms, normImporter, Add);  // add should be OK (everything should be zeros)

  for (int cellOrdinal=0; cellOrdinal<cellIDsByPartitionOrdering.size(); cellOrdinal++)
  {
    GlobalIndexType cellID = cellIDsByPartitionOrdering[cellOrdinal];
    _rieszRepNormSquaredGlobal[cellID] = globalNorms[cellOrdinal];
//    if (myRank==0) cout << "_rieszRepNormSquaredGlobal[" << cellID << "] = " << globalNorms[cellOrdinal] << endl;
  }

}

// computes riesz representation over a single element - map is from int (testID) to FieldContainer of values (sized cellIndex, numPoints)
template <typename Scalar>
void TRieszRep<Scalar>::computeRepresentationValues(FieldContainer<Scalar> &values, int testID, Camellia::EOperator op, BasisCachePtr basisCache)
{

  if (_repsNotComputed)
  {
    cout << "Computing riesz rep dofs" << endl;
    computeRieszRep();
  }

  int spaceDim = _mesh->getTopology()->getDimension();
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

  // all elems coming in should be of same type
  ElementTypePtr elemTypePtr = _mesh->getElementType(cellIDs[0]);
  DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
  int numTestDofsForVarID = testOrderingPtr->getBasisCardinality(testID, VOLUME_INTERIOR_SIDE_ORDINAL);
  BasisPtr testBasis = testOrderingPtr->getBasis(testID);

  bool testBasisIsVolumeBasis = (spaceDim == testBasis->domainTopology()->getDimension());
  bool useCubPointsSideRefCell = testBasisIsVolumeBasis && basisCache->isSideCache();

  Teuchos::RCP< const FieldContainer<double> > transformedBasisValues = basisCache->getTransformedValues(testBasis,op,useCubPointsSideRefCell);

  int rank = values.rank() - 2; // if values are shaped as (C,P), scalar...
  if (rank > 1)
  {
    cout << "ranks greater than 1 not presently supported...\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ranks greater than 1 not presently supported...");
  }

//  Camellia::print("cellIDs",cellIDs);

  values.initialize(0.0);
  for (int cellIndex = 0; cellIndex<numCells; cellIndex++)
  {
    int cellID = cellIDs[cellIndex];
    if (_rieszRepDofs.find(cellID) == _rieszRepDofs.end())
    {
      cout << "cellID " << cellID << " not found in _riesRepDofs container.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found");
    }
    for (int j = 0; j<numTestDofsForVarID; j++)
    {
      int dofIndex = testOrderingPtr->getDofIndex(testID, j);
      for (int i = 0; i<numPoints; i++)
      {
        if (rank==0)
        {
          double basisValue = (*transformedBasisValues)(cellIndex,j,i);
          values(cellIndex,i) += basisValue*_rieszRepDofs[cellID](dofIndex);
        }
        else
        {
          for (int d = 0; d<spaceDim; d++)
          {
            double basisValue = (*transformedBasisValues)(cellIndex,j,i,d);
            values(cellIndex,i,d) += basisValue*_rieszRepDofs[cellID](dofIndex);
          }
        }
      }
    }
  }
}

template <typename Scalar>
map<GlobalIndexType,double> TRieszRep<Scalar>::computeAlternativeNormSqOnCells(TIPPtr<Scalar> ip, vector<GlobalIndexType> cellIDs)
{
  map<GlobalIndexType,double> altNorms;
  int numCells = cellIDs.size();
  for (int i = 0; i<numCells; i++)
  {
    altNorms[cellIDs[i]] = computeAlternativeNormSqOnCell(ip, cellIDs[i]);
  }
  return altNorms;
}

template <typename Scalar>
double TRieszRep<Scalar>::computeAlternativeNormSqOnCell(TIPPtr<Scalar> ip, GlobalIndexType cellID)
{
  Teuchos::RCP<DofOrdering> testOrdering= _mesh->getElementType(cellID)->testOrderPtr;
  bool testVsTest = true;
  Teuchos::RCP<BasisCache> basisCache =   BasisCache::basisCacheForCell(_mesh, cellID, testVsTest,1);

  int numDofs = testOrdering->totalDofs();
  FieldContainer<Scalar> ipMat(1,numDofs,numDofs);
  ip->computeInnerProductMatrix(ipMat,testOrdering,basisCache);

  if (_rieszRepDofs.find(cellID) == _rieszRepDofs.end())
  {
    cout << "cellID " << cellID << " not found in _riesRepDofs container.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found");
  }
  
  double sum = 0.0;
  for (int i = 0; i<numDofs; i++)
  {
    for (int j = 0; j<numDofs; j++)
    {
      sum += _rieszRepDofs[cellID](i)*_rieszRepDofs[cellID](j)*ipMat(0,i,j);
    }
  }

  return sum;
}

template <typename Scalar>
TFunctionPtr<Scalar> TRieszRep<Scalar>::repFunction( VarPtr var, TRieszRepPtr<Scalar> rep )
{
  return Teuchos::rcp( new RepFunction<Scalar>(var, rep) );
}

template <typename Scalar>
TRieszRepPtr<Scalar> TRieszRep<Scalar>::rieszRep(MeshPtr mesh, TIPPtr<Scalar> ip, TLinearTermPtr<Scalar> rhs)
{
  return Teuchos::rcp( new TRieszRep<Scalar>(mesh,ip,rhs) );
}

namespace Camellia
{
template class TRieszRep<double>;
template class RepFunction<double>;
}
