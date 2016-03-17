//
//  GMGOperator.cpp
//  Camellia
//
//  Created by Nate Roberts on 7/3/14.
//
//

#include "GMGOperator.h"

#include "AdditiveSchwarz.h"
#include "BasisFactory.h"
#include "DofOrdering.h"
#include "GlobalDofAssignment.h"
#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"
#include "CondensedDofInterpreter.h"
#include "CubatureFactory.h"
#include "GDAMinimumRule.h"
#include "SerialDenseWrapper.h"

// EpetraExt includes
#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Epetra_SerialComm.h"

#include "Ifpack_BlockRelaxation.h"
#include "Ifpack_SparseContainer.h"
#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_PointRelaxation.h"
#include "Ifpack_Amesos.h"
#include "Ifpack_ILU.h"
#include "Ifpack_IC.h"
#include "Ifpack_Graph.h"
#include "Ifpack_Graph_Epetra_CrsGraph.h"
#include "Ifpack_Graph_Epetra_RowMatrix.h"

#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_BlockRelaxation.h"
#include "Ifpack_Graph_Epetra_RowMatrix.h"
#include "Ifpack_DenseContainer.h"

#include "Epetra_Operator_to_Epetra_Matrix.h"


using namespace Intrepid;
using namespace Camellia;

#ifdef USE_HPCTW
extern "C" void HPM_Start(char *);
extern "C" void HPM_Stop(char *);
#endif

GMGOperator::GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                         MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
                         Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation) :
Narrator("GMGOperator"),
_finePartitionMap(finePartitionMap), _br(true)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  
  if (useStaticCondensation)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(coarseIP == Teuchos::null, std::invalid_argument, "GMGOperator: coarseIP is required when useStaticCondensation = true");
  }
  
  _useStaticCondensation = useStaticCondensation;
  _fineDofInterpreter = fineDofInterpreter;
  _fineCoarseRolesSwapped = false; //default

  _hierarchicalNeighborsForSchwarz = false; // default; can be overridden later
  _dimensionForSchwarzNeighborRelationship = fineMesh->getTopology()->getDimension() - 1; // side dimension default

  _debugMode = false;

  _schwarzBlockFactorizationType = Direct;

  _smootherApplicationCount = 1;
  
  // additive implies a two-level operator.
  _smootherApplicationType = MULTIPLICATIVE; // this field is deprecated
  _multigridStrategy = V_CYCLE;

  _levelOfFill = 2;
  _fillRatio = 5.0;
  _smootherWeight = 1.0;

  clearTimings();

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time constructionTimer(Comm);

  RHSPtr zeroRHS = RHS::rhs();
  _fineMesh = fineMesh;
  _coarseMesh = coarseMesh;
  _bc = zeroBCs;
  _coarseSolution = Teuchos::rcp( new TSolution<double>(coarseMesh, zeroBCs, zeroRHS, coarseIP) );
  
  set<GlobalIndexType> offRankCellsToInclude; // ignored (and empty) when useStaticCondensation is false
  if (useStaticCondensation)
  {
    const set<GlobalIndexType>* fineCellIDsForRank = &_fineMesh->globalDofAssignment()->cellsInPartition(rank);
    const set<GlobalIndexType>* coarseCellIDsForRank = &_coarseMesh->globalDofAssignment()->cellsInPartition(rank);
    for (GlobalIndexType fineCellID : *fineCellIDsForRank)
    {
      GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
      if (coarseCellIDsForRank->find(coarseCellID) == coarseCellIDsForRank->end())
      {
        offRankCellsToInclude.insert(coarseCellID);
      }
    }
  }
  _coarseSolution->setUseCondensedSolve(useStaticCondensation, offRankCellsToInclude);

  _coarseSolver = coarseSolver;
  _haveSolvedOnCoarseMesh = false;

  setSmootherType(CAMELLIA_ADDITIVE_SCHWARZ); // default
  _smootherOverlap = 0;
  _useSchwarzDiagonalWeight = false;
  _useSchwarzScalingWeight = true; // weight to ensure maximum eigenvalue of M*A <= 1.0, M being the Schwarz smoother.

  if (( coarseMesh->meshUsesMaximumRule()) || (! fineMesh->meshUsesMinimumRule()) )
  {
    cout << "GMGOperator only supports minimum rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GMGOperator only supports minimum rule.");
  }

  _coarseSolution->initializeLHSVector();
  _coarseSolution->initializeLoad(); // don't need to initialize stiffness anymore; we'll do this in computeCoarseStiffnessMatrix

//  constructProlongationOperator();

  _timeConstruction += constructionTimer.ElapsedTime();
}

GMGOperator::GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh,
                         Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
                         bool useStaticCondensation)
: GMGOperator(zeroBCs, coarseMesh, coarseIP, fineMesh,fineDofInterpreter, finePartitionMap, Teuchos::null, useStaticCondensation) {}

void GMGOperator::clearTimings()
{
  _timeMapFineToCoarse = 0, _timeMapCoarseToFine = 0, _timeConstruction = 0, _timeCoarseSolve = 0, _timeCoarseImport = 0, _timeLocalCoefficientMapConstruction = 0, _timeComputeCoarseStiffnessMatrix = 0, _timeProlongationOperatorConstruction = 0,
  _timeSetUpSmoother = 0, _timeUpdateCoarseOperator = 0, _timeApplyFineStiffness = 0, _timeApplySmoother = 0;
}

void GMGOperator::setClearFinestCondensedDofInterpreterAfterProlongation(bool value)
{
  _clearFinestCondensedDofInterpreterAfterProlongation = value;
}

void GMGOperator::computeCoarseStiffnessMatrix(Epetra_CrsMatrix *fineStiffnessMatrix)
{
  narrate("computeCoarseStiffnessMatrix");
  int globalColCount = fineStiffnessMatrix->NumGlobalCols();
  if (_P.get() == NULL)
  {
    constructProlongationOperator();
  }
  else if (prolongationRowCount() != globalColCount)
  {
    constructProlongationOperator();
    if (prolongationRowCount() != globalColCount)
    {
      cout << "GMGOperator::computeCoarseStiffnessMatrix: Even after a fresh call to constructProlongationOperator, _P->NumGlobalRows() != globalColCount (";
      cout << prolongationRowCount() << " != " << globalColCount << ").\n";

      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "prolongationRowCount() != globalColCount");
    }
  }

//  cout << "Writing fine stiffness to disk before setting up smoother.\n";
//  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/A.dat",*fineStiffnessMatrix, NULL, NULL, false); // false: don't write header
  
//  { // DEBUGGING
//    if (this->getOperatorLevel() == 2)
//    {
//      cout << "Writing fine stiffness to disk before setting up smoother.\n";
//      EpetraExt::RowMatrixToMatrixMarketFile("A_fine.dat",*fineStiffnessMatrix, NULL, NULL, false); // false: don't write header
//    }
//  }
  
  Epetra_Time coarseStiffnessTimer(Comm());

//  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/A.dat",*fineStiffnessMatrix, NULL, NULL, false); // false: don't write header

//  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*_P, NULL, NULL, false); // false: don't write header

  if (!_fineCoarseRolesSwapped)
  {
    int maxRowSize = 0; // _P->MaxNumEntries();
    
    // compute A * P
    Epetra_CrsMatrix AP(::Copy, _finePartitionMap, maxRowSize);
    //  Epetra_CrsMatrix AP(::Copy, fineStiffnessMatrix->RowMap(), maxRowSize);
    
    int err = EpetraExt::MatrixMatrix::Multiply(*fineStiffnessMatrix, false, *_P, _fineCoarseRolesSwapped, AP);
    if (err != 0)
    {
      cout << "ERROR: EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of A * P.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of A * P.");
    }
    
    // PRETTY SURE domain_P is problematic when _fineCoarseRolesSwapped == true!!
    // TODO: figure out the appropriate way to fix this
    //  Epetra_Map domain_P = _fineCoarseRolesSwapped ? _P->RangeMap() : _P->DomainMap();
    //    Epetra_Map domain_P = _coarseSolution->getPartitionMap();
    Epetra_Map domain_P = _P->DomainMap();
//    Epetra_Map range_P = _P->RangeMap();
//    printMapSummary(domain_P, "_P->DomainMap()");
//    printMapSummary(range_P, "_P->RangeMap()");
    Teuchos::RCP<Epetra_CrsMatrix> PT_A_P = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, domain_P, maxRowSize) );
    
    // compute P^T * A * P
    err = EpetraExt::MatrixMatrix::Multiply(*_P, !_fineCoarseRolesSwapped, AP, false, *PT_A_P);
    if (err != 0)
    {
      cout << "WARNING: EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of P^T * (A * P).\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of P^T * (A * P).");
    }
    
    PT_A_P->FillComplete();
    
    //  { // DEBUGGING
    //    if (this->getOperatorLevel() == 2)
    //    {
    //      cout << "Writing AP to disk.\n";
    //      EpetraExt::RowMatrixToMatrixMarketFile("AP.dat",AP, NULL, NULL, false); // false: don't write header
    //      cout << "Writing PT_A_P to disk.\n";
    //      EpetraExt::RowMatrixToMatrixMarketFile("PT_A_P.dat",*PT_A_P, NULL, NULL, false); // false: don't write header
    //    }
    //  }
    
    //  string PT_A_P_path = "/tmp/PT_AP.dat";
    //  cout << "Writing P^T * (A * P) to disk at " << PT_A_P_path << endl;
    //  EpetraExt::RowMatrixToMatrixMarketFile(PT_A_P_path.c_str(), *PT_A_P, NULL, NULL, false); // false: don't write header
    //
    //  string P_path = "/tmp/P.dat";
    //  cout << "Writing P to disk at " << P_path << endl;
    //  EpetraExt::RowMatrixToMatrixMarketFile(P_path.c_str(),*_P, NULL, NULL, false); // false: don't write header
    
    Epetra_Map coarsePartitionMap = _coarseSolution->getPartitionMap();
    Epetra_Import  coarseImporter(coarsePartitionMap, PT_A_P->RowMap());
//    { // DEBUGGING
//      Camellia::printMapSummary(coarsePartitionMap, "coarsePartitionMap");
//      Camellia::printMapSummary(PT_A_P->RowMap(), "PT_A_P->RowMap()");
//    }
    
    Teuchos::RCP<Epetra_CrsMatrix> coarseStiffness;
    if (_coarseSolution->getZMCsAsGlobalLagrange() &&
        (prolongationColCount() > _coarseSolution->getDofInterpreter()->globalDofCount()))
    {
      // essentially: there are some lagrange constraints applied in coarse solve --> we shouldn't use the fused import,
      //              since this will call FillComplete()
      
      int numEntriesPerRow = 0; // sub-optimal, but easy
      coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, coarsePartitionMap, numEntriesPerRow) );
      coarseStiffness->Import(*PT_A_P,coarseImporter,::Insert);
      _coarseSolution->setStiffnessMatrix(coarseStiffness);
      _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
      // now can call FillComplete()
      coarseStiffness->FillComplete();
    }
    else
    {
      coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(*PT_A_P, coarseImporter) );
      
      _coarseSolution->setStiffnessMatrix(coarseStiffness);
      _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
    }
  }
  else // _fineCoarseRolesSwapped == true
  {
    int maxRowSize = 0; // _P->MaxNumEntries();
    
    // compute P^T * A
    Epetra_CrsMatrix PT_A(::Copy, _P->RowMap(), maxRowSize);
    
    int err = EpetraExt::MatrixMatrix::Multiply(*_P, false, *fineStiffnessMatrix, false, PT_A);
    if (err != 0)
    {
      cout << "ERROR: EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of P^T * A.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of P^T * A.");
    }
    
    Epetra_Map domain_P = _P->RowMap(); // when _fineCoarseRolesSwapped is true, _P really refers to P^T, so the row map of _P is the domain of P.
    Teuchos::RCP<Epetra_CrsMatrix> PT_A_P = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, domain_P, maxRowSize) );
    
    // compute P^T * A * P
    err = EpetraExt::MatrixMatrix::Multiply(PT_A, false, *_P, true, *PT_A_P); // transpose _P == P^T (since _fineCoarseRolesSwapped is true) to get P
    if (err != 0)
    {
      cout << "WARNING: EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of (P^T * A) * P.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of (P^T * A) * P.");
    }
    
    PT_A_P->FillComplete();
    
    Epetra_Map coarsePartitionMap = _coarseSolution->getPartitionMap();
    Epetra_Import  coarseImporter(coarsePartitionMap, PT_A_P->RowMap());
    { // DEBUGGING
      Camellia::printMapSummary(coarsePartitionMap, "coarsePartitionMap");
      Camellia::printMapSummary(PT_A_P->RowMap(), "PT_A_P->RowMap()");
    }
    
    Teuchos::RCP<Epetra_CrsMatrix> coarseStiffness;
    if (_coarseSolution->getZMCsAsGlobalLagrange() &&
        (prolongationColCount() > _coarseSolution->getDofInterpreter()->globalDofCount()))
    {
      // essentially: there are some lagrange constraints applied in coarse solve --> we shouldn't use the fused import,
      //              since this will call FillComplete()
      
      int numEntriesPerRow = 0; // sub-optimal, but easy
      coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, coarsePartitionMap, numEntriesPerRow) );
      coarseStiffness->Import(*PT_A_P,coarseImporter,::Insert);
      _coarseSolution->setStiffnessMatrix(coarseStiffness);
      _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
      // now can call FillComplete()
      coarseStiffness->FillComplete();
    }
    else
    {
      int numEntriesPerRow = 0; // sub-optimal, but easy
      coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, coarsePartitionMap, numEntriesPerRow) );
      coarseStiffness->Import(*PT_A_P,coarseImporter,::Insert);
      _coarseSolution->setStiffnessMatrix(coarseStiffness);
      _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
      // now can call FillComplete()
      coarseStiffness->FillComplete();

      // fused import below isn't working in this context... (Hence I copied the lagrange implementation above)
//      
//      coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(*PT_A_P, coarseImporter) );
//      
//      _coarseSolution->setStiffnessMatrix(coarseStiffness);
//      _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
    }
  }

  _timeComputeCoarseStiffnessMatrix = coarseStiffnessTimer.ElapsedTime();

  _haveSolvedOnCoarseMesh = false; // having recomputed coarseStiffness, any existing factorization is invalid
}

// res should hold the RHS on entry
void GMGOperator::computeResidual(const Epetra_MultiVector& Y, Epetra_MultiVector& res, Epetra_MultiVector& A_Y) const
{
  Epetra_Time timer(Comm());
  int err = _fineStiffnessMatrix->Apply(Y, A_Y);
  if (err != 0)
  {
    cout << "_fineStiffnessMatrix->Apply returned non-zero error code " << err << endl;
  }
  res.Update(-1.0, A_Y, 1.0);
  _timeApplyFineStiffness += timer.ElapsedTime();
  if (_debugMode)
  {
    if (Teuchos::GlobalMPISession::getRank()==0) cout << "Updated residual after smoother application:\n";
    res.Comm().Barrier();
    cout << res;
  }
}

// ! Computed as 1/(1+N), where N = max #neighbors of any cell's overlap region.
double GMGOperator::computeSchwarzSmootherWeight()
{
  // (For IfPack, this weight may not be exactly correct, but it's probably kinda close.  Likely we will deprecate support for
  // IFPACK_ADDITIVE_SCHWARZ soon.)
  
  // Suppose that E is a matrix with rows and columns corresponding to the elements, with 0s for disconnected elements, and 1s for
  // connected elements (an element is connected to itself and its face neighbors).
  // Conjecture: rho(E) is bounded above by the maximum side count of an element plus 1.
  // Based on results in Smith et al., I *think* that we can bound the maximum eigenvalue of S*A by 1 + rho(E)
  set<GlobalIndexType> myCellIndices = _fineMesh->globalDofAssignment()->cellsInPartition(-1);
  Epetra_CommPtr Comm = _fineMesh->Comm();
  
  int localMaxNeighbors = 0;
  MeshTopologyViewPtr fineMeshTopo = _fineMesh->getTopology();
  for (GlobalIndexType cellIndex : myCellIndices)
  {
    set<GlobalIndexType> subdomainCellsAndNeighbors = OverlappingRowMatrix::overlappingCells(cellIndex, _fineMesh, _smootherOverlap,
                                                                                             _hierarchicalNeighborsForSchwarz,
                                                                                             _dimensionForSchwarzNeighborRelationship);
    
    set<GlobalIndexType> subdomainNeighbors;
    for (GlobalIndexType subdomainCellIndex : subdomainCellsAndNeighbors)
    {
      CellPtr subdomainCell = _fineMesh->getTopology()->getCell(subdomainCellIndex);
      int sideCount = subdomainCell->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
      {
        pair<GlobalIndexType,unsigned> neighborInfo = subdomainCell->getNeighborInfo(sideOrdinal, fineMeshTopo);
        GlobalIndexType neighborCellID = neighborInfo.first;
        if (neighborCellID == -1) continue;
        unsigned neighborSideOrdinal = neighborInfo.second;
        CellPtr neighbor = fineMeshTopo->getCell(neighborCellID);
        if (!neighbor->isParent(fineMeshTopo))
        {
          subdomainNeighbors.insert(neighborCellID);
        }
        else
        {
          bool leafCellsOnly = true;
          vector<pair<GlobalIndexType,unsigned>> neighborDescendentsInfo = neighbor->getDescendantsForSide(neighborSideOrdinal, fineMeshTopo, leafCellsOnly);
          for (auto neighborDescendentInfo : neighborDescendentsInfo)
          {
            subdomainNeighbors.insert(neighborDescendentInfo.first);
          }
        }
      }
    }
    
    subdomainCellsAndNeighbors.insert(subdomainNeighbors.begin(),subdomainNeighbors.end());
//    {
//      // DEBUGGING
//      if (localMaxNeighbors < subdomainCellsAndNeighbors.size())
//      {
//        cout << "cell " << cellIndex << " has " << subdomainCellsAndNeighbors.size() << " neighbors.\n";
//        print("neighbors", subdomainCellsAndNeighbors);
//      }
//    }
    localMaxNeighbors = max(localMaxNeighbors,(int)subdomainCellsAndNeighbors.size());
  }
  int globalMaxNeighbors;
  Comm->MaxAll(&localMaxNeighbors, &globalMaxNeighbors, 1);
  
  double weight = 1.0 / (globalMaxNeighbors + 1); // aiming for a weight that guarantees max eig of weight * S * A is 1.0
  
  return weight;
}

void GMGOperator::constructLocalCoefficientMaps()
{
  narrate("constructLocalCoefficientMaps()");
  Epetra_Time timer(Comm());

  set<GlobalIndexType> cellsInPartition = _fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local

  for (set<GlobalIndexType>::iterator cellIDIt=cellsInPartition.begin(); cellIDIt != cellsInPartition.end(); cellIDIt++)
  {
    GlobalIndexType fineCellID = *cellIDIt;
    LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
  }
  _timeLocalCoefficientMapConstruction += timer.ElapsedTime();
}

Teuchos::RCP<Epetra_FECrsMatrix> GMGOperator::constructProlongationOperator(Teuchos::RCP<DofInterpreter> coarseDofInterpreter,
                                                                            Teuchos::RCP<DofInterpreter> fineDofInterpreter,
                                                                            bool useStaticCondensation,
                                                                            Epetra_Map &coarseMap, Epetra_Map &fineMap,
                                                                            MeshPtr coarseMesh, MeshPtr fineMesh)
{
  // row indices belong to the fine grid, columns to the coarse
  // maps coefficients from coarse to fine
  
  int maxRowSizeToPrescribe = 0; //_coarseMesh->rowSizeUpperBound();
  
  CondensedDofInterpreter<double>* condensedDofInterpreterCoarse = NULL;
  CondensedDofInterpreter<double>* condensedDofInterpreterFine = NULL;
  
  set<int> varsToExclude;
  if (useStaticCondensation)
  {
    condensedDofInterpreterCoarse = dynamic_cast<CondensedDofInterpreter<double>*>(coarseDofInterpreter.get());
    condensedDofInterpreterCoarse->setCanSkipLocalFieldInInterpretGlobalCoefficients(true);
    condensedDofInterpreterFine = dynamic_cast<CondensedDofInterpreter<double>*>(fineDofInterpreter.get());
    condensedDofInterpreterFine->setCanSkipLocalFieldInInterpretGlobalCoefficients(true);
    varsToExclude = condensedDofInterpreterCoarse->condensibleVariableIDs();
  }
  
  Teuchos::RCP<Epetra_FECrsMatrix> P = Teuchos::rcp( new Epetra_FECrsMatrix(::Copy, fineMap, maxRowSizeToPrescribe) );
  
  // by convention, all constraints come at the end.  (Element and global lagrange constraints, zero-mean constraints.)
  // There should be the same number in the coarse and the fine Solution objects, and the mapping is taken to be the identity.
  
  GlobalIndexType firstFineConstraintRowIndex = fineDofInterpreter->globalDofCount();
  GlobalIndexType firstCoarseConstraintRowIndex = coarseDofInterpreter()->globalDofCount();
  
  // strategy: we iterate on the rank-local global dof indices on the fine mesh.
  //           we construct X, a canonical basis vector for the global dof index.
  //           we then determine the coarse mesh coefficients for X.
  //           This is a row of the matrix P.
  set<GlobalIndexType> cellsInPartition = fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local
  
  {
    Epetra_SerialComm SerialComm; // rank-local map
    
    GlobalIndexTypeToCast* myGlobalIndicesPtr;
    fineMap.MyGlobalElementsPtr(*&myGlobalIndicesPtr);
    
    map<GlobalIndexTypeToCast, set<GlobalIndexType> > cellsForGlobalDofOrdinal;
    vector<GlobalIndexTypeToCast> myGlobalIndices(fineMap.NumMyElements());
    {
      set<GlobalIndexTypeToCast> globalIndicesForRank;
      for  (int i=0; i <fineMap.NumMyElements(); i++)
      {
        globalIndicesForRank.insert(myGlobalIndicesPtr[i]);
        myGlobalIndices[i] = myGlobalIndicesPtr[i];
      }
      
      // for dof interpreter's sake, want to put 0's in slots for any seen-but-not-owned global coefficients
      set<GlobalIndexType> myCellIDs = fineMesh->globalDofAssignment()->cellsInPartition(-1);
      for (GlobalIndexType cellID : myCellIDs)
      {
        set<GlobalIndexType> globalDofsForCell = fineDofInterpreter->globalDofIndicesForCell(cellID);
        for (GlobalIndexType globalDofIndex : globalDofsForCell)
        {
          cellsForGlobalDofOrdinal[globalDofIndex].insert(cellID);
          if (globalIndicesForRank.find(globalDofIndex) == globalIndicesForRank.end())
          {
            myGlobalIndices.push_back(globalDofIndex);
          }
        }
      }
    }

    // TODO: consider changing this to use an Epetra_Map with a single entry (given the usage below, this should be possible).
    Epetra_Map    localXMap(myGlobalIndices.size(), myGlobalIndices.size(), &myGlobalIndices[0], 0, SerialComm);
    Teuchos::RCP<Epetra_Vector> XLocal = Teuchos::rcp( new Epetra_Vector(localXMap) );
    
//    Epetra_Map    mapOneElement(1,1,&myGlobalIndices[0], 0, SerialComm);
//    Teuchos::RCP<Epetra_Vector> oneElementVector = Teuchos::rcp( new Epetra_Vector(mapOneElement) );
//    (*oneElementVector)[0] = 1.0;
    
    // to (possibly) reduce the cost of allocating these containers, declare them outside and resize inside as needed
    FieldContainer<GlobalIndexTypeToCast> coarseGlobalIndices;
    FieldContainer<double> coarseGlobalValues;
    FieldContainer<double> coarseCellCoefficients;
    FieldContainer<double> mappedCoarseCellCoefficients;
    FieldContainer<double> globalCoefficients;
    FieldContainer<GlobalIndexType> globalDofIndices;
    FieldContainer<double> fineCellCoefficients;
    FieldContainer<double> basisCoefficients;
    
    for (int localID=0; localID < fineMap.NumMyElements(); localID++)
    {
      GlobalIndexTypeToCast globalRow = fineMap.GID(localID);
      map<GlobalIndexTypeToCast, double> coarseXVectorLocal; // rank-local representation, so we just use an STL map.  Has the advantage of growing as we need it to.
//      mapOneElement = Epetra_Map(1,1,&myGlobalIndices[localID],0,SerialComm);
//      oneElementVector->ReplaceMap(mapOneElement);
      (*XLocal)[localID] = 1.0;
      
      if (globalRow >= firstFineConstraintRowIndex)
      {
        // belongs to a lagrange degree of freedom (zero-mean constraints are a special case), so we do a one-to-one map
        // NOTE: this isn't going to work properly for element Lagrange constraints in the context of h-multigrid, since
        //       for these we have one row per element, and the number of elements is reduced for the coarse mesh;
        //       it's not entirely clear to me what we should do in the general case, though for zero-mean constraints
        //       (even element-wise ones, like local conservation), probably simply weighting by relative element
        //       volume will be fine.  But we need a bit more information here; we don't have access to the fine Solution's
        //       Lagrange constraints on the present interface.
        int offset = globalRow - firstFineConstraintRowIndex;
        GlobalIndexType coarseGlobalRow = firstCoarseConstraintRowIndex + offset;
        coarseXVectorLocal[coarseGlobalRow] = 1.0;
      }
      else
      {
        for (GlobalIndexType fineCellID : cellsForGlobalDofOrdinal[globalRow])
        {
          DofOrderingPtr fineTrialOrdering = fineMesh->getElementType(fineCellID)->trialOrderPtr;
          int fineDofCount = fineTrialOrdering->totalDofs();
          fineCellCoefficients.resize(fineDofCount);
        
          /*
           The following is a bit backwards; it should be that DofInterpreter itself could tell us which variables
           and sides belong to the provided global degree of freedom.  But adding this is a change that would take more
           effort than the below: here, we take the local coefficients and work out which variable matches.
           
           (For GDAMinimumRule, it wouldn't be *that* much effort to go from global dof index to variable; we have
           lookup tables and at worst we could do a brute force search through the lookup table for the fine cell.
           But in fact the brute force search wouldn't likely be much/any cheaper than variablesWithNonZeroEntries() below.)
           */
          fineDofInterpreter->interpretGlobalCoefficients(fineCellID, fineCellCoefficients, *XLocal);
//          fineDofInterpreter->interpretGlobalCoefficients(fineCellID, fineCellCoefficients, *oneElementVector);
          
//          cout << "fineCellCoefficients:\n" << fineCellCoefficients;
          
          vector<pair<int,vector<int>>> fineNonZeroVarIDs = fineTrialOrdering->variablesWithNonZeroEntries(fineCellCoefficients);
          if (fineNonZeroVarIDs.size() != 1)
          {
            GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(_fineMesh->globalDofAssignment().get());
            minRule->printGlobalDofInfo();
            fineTrialOrdering->print(cout);
            cout << fineCellCoefficients;
            TEUCHOS_TEST_FOR_EXCEPTION(fineNonZeroVarIDs.size() != 1, std::invalid_argument, "There should be exactly one variable corresponding to a given global dof ordinal in fine mesh");
          }
          
          LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
          GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
          
          if (_fineCoarseRolesSwapped && (coarseCellID != fineCellID))
          {
            // then getLocalCoefficientMap() will have gotten the wrong thing.
            // (in fact, it's only the right thing when _fineCoarseRolesSwapped == true
            //  if fineMapper is *identity* -- we use role-swapping to allow construction of
            //  mapping from a "conforming" mesh to a "non-conforming" one.)
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "swapping fine and coarse roles is not supported for h-refinements");
          }
          DofOrderingPtr coarseTrialOrdering = coarseMesh->getElementType(coarseCellID)->trialOrderPtr;
          int coarseDofCount = coarseMesh->getElementType(coarseCellID)->trialOrderPtr->totalDofs();
          coarseCellCoefficients.resize(coarseDofCount);
          mappedCoarseCellCoefficients.resize(fineMapper->globalIndices().size());
          
          coarseCellCoefficients = fineMapper->mapLocalData(fineCellCoefficients, true, fineNonZeroVarIDs);
          
//          cout << "coarseCellCoefficients:\n" << coarseCellCoefficients;
          
          vector<pair<int,vector<int>>> coarseNonZeroVarIDs = coarseTrialOrdering->variablesWithNonZeroEntries(coarseCellCoefficients);
          
          map<GlobalIndexType,double> fittedGlobalCoefficients;
          for (auto varIDSidesEntry : coarseNonZeroVarIDs)
          {
            int varID = varIDSidesEntry.first;
            for (int sideOrdinal : varIDSidesEntry.second)
            {
              const vector<int>* localDofIndices = &coarseTrialOrdering->getDofIndices(varID,sideOrdinal);
              basisCoefficients.resize(localDofIndices->size());
              for (int basisOrdinal=0; basisOrdinal < localDofIndices->size(); basisOrdinal++)
              {
                basisCoefficients[basisOrdinal] = coarseCellCoefficients[(*localDofIndices)[basisOrdinal]];
              }
              
              coarseDofInterpreter->interpretLocalBasisCoefficients(coarseCellID, varID, sideOrdinal, basisCoefficients,
                                                                    globalCoefficients, globalDofIndices);
              
//              cout << "basisCoefficients for varID " << varID << " on side " << sideOrdinal << ":\n" << basisCoefficients;
//              cout << "globalCoefficients:\n" << globalCoefficients;
              for (int i=0; i<globalCoefficients.size(); i++)
              {
                fittedGlobalCoefficients[globalDofIndices[i]] = globalCoefficients[i];
              }
            }
          }
          
//          FieldContainer<double> interpretedCoarseData;
//          map<GlobalIndexType,double> fittedGlobalCoefficients;
//          coarseDofInterpreter->interpretLocalCoefficients(coarseCellID, coarseCellCoefficients, fittedGlobalCoefficients,
//                                                           varsToExclude);
          
//          { // DEBUGGING
//            if (globalRow == 4)
//            {
//              cout << "globalRow " << globalRow << ", fineCellCoefficients:\n" << fineCellCoefficients;
//              cout << "globalRow " << globalRow << ", coarseCellCoefficients:\n" << coarseCellCoefficients;
//              print("fittedGlobalCoefficients",fittedGlobalCoefficients);
//            }
//          }
          
          for (auto globalEntry : fittedGlobalCoefficients)
          {
            GlobalIndexType globalDofIndex = globalEntry.first;
            double value = globalEntry.second;
            coarseXVectorLocal[globalDofIndex] = value;
          }
        }
      }
      
      coarseGlobalIndices.resize(coarseXVectorLocal.size());
      coarseGlobalValues.resize(coarseXVectorLocal.size());
      int nnz = 0; // nonzero entries
      double tol = 1e-13; // count anything less as a zero entry
      //      cout << "P global row " << globalRow << ": ";
      for (map<GlobalIndexTypeToCast, double>::iterator coarseXIt=coarseXVectorLocal.begin(); coarseXIt != coarseXVectorLocal.end(); coarseXIt++)
      {
        if (abs(coarseXIt->second) > tol)
        {
          coarseGlobalIndices[nnz] = coarseXIt->first;
          coarseGlobalValues[nnz] = coarseXIt->second;
          //          cout << coarseGlobalIndices[nnz] << " --> " << coarseGlobalValues[nnz] << "; ";
          nnz++;
        }
      }
      //      cout << endl;
      if (nnz > 0)
      {
        P->InsertGlobalValues(globalRow, nnz, &coarseGlobalValues[0], &coarseGlobalIndices[0]);
        //        cout << "Inserting values for row " << globalRow << endl;
      }
      (*XLocal)[localID] = 0.0;
    }
  }
  
//  cout << "Fine mesh has " << _fineMesh->numGlobalDofs() << " global dofs.\n";
  
  //  cout << "before FillComplete(), P has " << P->NumGlobalRows64() << " rows and " << P->NumGlobalCols64() << " columns.\n";
  
  P->GlobalAssemble(coarseMap, fineMap);
  
  const static bool DEBUGGING = false;
  if (DEBUGGING)
  { // DEBUGGING
    // check whether all domain map (coarseMap) GIDs are present as column indices
    // (they should be, but it appears there are cases when they aren't)
    int numDomainElements = coarseMap.NumMyElements();
    
    int numMyRows = P->NumMyRows();
    
    vector<bool> localGID(numDomainElements,false); // track which LIDs we have found
    int numLocalColGIDs = 0;
    
    vector<int> colIndices;
    vector<double> values;
    for (int i=0; i<numMyRows; i++)
    {
      int numRowEntries = P->NumMyEntries(i);
      colIndices.resize(numRowEntries);
      values.resize(numRowEntries);
      P->ExtractMyRowCopy(i, numRowEntries, numRowEntries, &values[0], &colIndices[0]);
      for (int j=0; j<numRowEntries; j++)
      {
        int GID = colIndices[j];
        if (i==8622)
        {
          cout << "Row 8622 has column entry for GID " << GID << endl;
        }
        
        int LID = coarseMap.LID(GID);
        if (LID != -1)
        {
          bool previouslyFound = localGID[LID];
          if (!previouslyFound)
          {
            localGID[LID] = true;
            numLocalColGIDs++;
          }
        }
      }
    }
    cout << "Found " << numLocalColGIDs << " of " << numDomainElements << " in coarseMap.\n";
    
    if (numLocalColGIDs != numDomainElements)
    {
      cout << "Missing coarse GIDs: ";
      for (int colLID=0; colLID<numDomainElements; colLID++)
      {
        if (localGID[colLID]) continue;
        int GID = coarseMap.GID(colLID);
        cout << GID << " ";
      }
      cout << endl;
      
      GDAMinimumRule* gdaMinRuleCoarseMesh = dynamic_cast<GDAMinimumRule *>(_coarseMesh->globalDofAssignment().get());
      gdaMinRuleCoarseMesh->printGlobalDofInfo();
    }
  }
  
  if (condensedDofInterpreterFine != NULL)
  {
    condensedDofInterpreterFine->setCanSkipLocalFieldInInterpretGlobalCoefficients(false); //just true while in this call
    if (!_isFinest || _clearFinestCondensedDofInterpreterAfterProlongation)
      condensedDofInterpreterFine->clearStiffnessAndLoad();
  }
  
  if (condensedDofInterpreterCoarse != NULL)
  {
    condensedDofInterpreterCoarse->setCanSkipLocalFieldInInterpretGlobalCoefficients(false);
  }
  
  return P;
}

Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::constructProlongationOperator()
{
  Epetra_Time prolongationTimer(Comm());
  narrate("constructProlongationOperator");
  
  if (! _fineCoarseRolesSwapped)
  {
    Epetra_Map coarseMap = _coarseSolution->getPartitionMap();
    _P = this->constructProlongationOperator(_coarseSolution->getDofInterpreter(), _fineDofInterpreter, _useStaticCondensation,
                                             coarseMap, _finePartitionMap, _coarseMesh, _fineMesh);
  }
  else
  {
    Epetra_Map coarseMap = _coarseSolution->getPartitionMap();
    _P = this->constructProlongationOperator(_fineDofInterpreter, _coarseSolution->getDofInterpreter(), _useStaticCondensation,
                                             _finePartitionMap, coarseMap, _fineMesh, _coarseMesh);
  }

  _timeProlongationOperatorConstruction = prolongationTimer.ElapsedTime();
  
  ostringstream prolongationTimingReport;
  prolongationTimingReport << "Prolongation operator constructed in " << _timeProlongationOperatorConstruction << " seconds.";
  narrate(prolongationTimingReport.str());
  
  return _P;
}

void GMGOperator::exportFunction()
{
  if (_functionExporter != Teuchos::null)
  {
    _functionExporter->exportFunction(_functionToExport, _functionToExportName, _exportTimeStep);
    _exportTimeStep++;
  }
}

Teuchos::RCP<Epetra_SerialDenseMatrix> GMGOperator::fluxDuplicationMapForCoarseCell(GlobalIndexType coarseCellID) const
{
  // a null return value indicates that the operator is the identity (this happens for non-conforming ElementTypes)
 
  // otherwise, returns a matrix that can right-multiply the fluxToFieldMap
  
  DofOrdering* trialOrderRawPtr = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr.get();
  if (_fluxDuplicationMap.find(trialOrderRawPtr) == _fluxDuplicationMap.end())
  {
    bool hasConformingTraces = false;
    for (int varID : trialOrderRawPtr->getVarIDs())
    {
      if (trialOrderRawPtr->getSidesForVarID(varID).size() > 1)
      {
        // trace or flux; use basis on first side as example
        EFunctionSpace fs = trialOrderRawPtr->getBasis(varID, trialOrderRawPtr->getSidesForVarID(varID)[0])->functionSpace();
        if (! functionSpaceIsDiscontinuous(fs))
        {
          hasConformingTraces = true;
          break;
        }
      }
    }
    if (!hasConformingTraces)
    {
      _fluxDuplicationMap[trialOrderRawPtr] = Teuchos::null;
      return _fluxDuplicationMap[trialOrderRawPtr];
    }
    
    CondensedDofInterpreter<double>* condensedDofInterpreterCoarse = NULL;
    
    if (_useStaticCondensation)
    {
      condensedDofInterpreterCoarse = dynamic_cast<CondensedDofInterpreter<double>*>(_coarseSolution->getDofInterpreter().get());
    }
    TEUCHOS_TEST_FOR_EXCEPTION(condensedDofInterpreterCoarse == NULL, std::invalid_argument, "fluxDuplicationMapForCoarseCell() requires a coarse condensed dof interpreter...");
    
    vector<int> fluxIndices = condensedDofInterpreterCoarse->fluxIndexLookupLocalCell(coarseCellID);
    
    Epetra_CommPtr SerialComm = MPIWrapper::CommSerial();
    MeshPtr coarseSingleCellMesh = Teuchos::rcp( new Mesh(_coarseMesh, coarseCellID, SerialComm) );
    
    // the fact that the local view does not enforce conformity means that for conforming traces
    // the coarseFluxToFieldMapMatrix does not do the right thing if we only map, say, one of two
    // dofs identified with a vertex.  Since this is the way we approach things below, here we manipulate
    // the matrix, essentially enforcing conformity:
    // (NOTE: the same thing could be accomplished at the abstract, reference-element level.  It would be more
    //  efficient to construct a matrix like this once for the element type, and then multiply matrices to enforce
    //  conformity...)
    FieldContainer<double> localData(trialOrderRawPtr->totalDofs());
    FieldContainer<double> localCoefficients(trialOrderRawPtr->totalDofs());
    FieldContainer<double> globalCoefficients;
    FieldContainer<GlobalIndexType> globalDofIndices;
    GlobalIndexType zeroCellID = 0;
    int indexBase = 0;
    int numDofsSingleElementMesh = coarseSingleCellMesh->numGlobalDofs();
    Epetra_Map serialGlobalDofMap(numDofsSingleElementMesh,indexBase,*SerialComm);
    Epetra_MultiVector globalCoefficientsVector(serialGlobalDofMap,1,true);
    
    Teuchos::RCP<Epetra_SerialDenseMatrix> fluxMap = Teuchos::rcp( new Epetra_SerialDenseMatrix(fluxIndices.size(),fluxIndices.size()) );
    
    int fluxMapColumn = 0;
    for (int fluxIndex : fluxIndices)
    {
      localData.initialize(0.0);
      localData(fluxIndex) = 1.0;
      
      coarseSingleCellMesh->globalDofAssignment()->interpretLocalData(zeroCellID, localData, globalCoefficients, globalDofIndices);
      
      for (int i=0; i<globalCoefficients.size(); i++)
      {
        int globalDofIndex = globalDofIndices[i];
        globalCoefficientsVector[0][globalDofIndex] = globalCoefficients[i];
      }
      
      coarseSingleCellMesh->globalDofAssignment()->interpretGlobalCoefficients(zeroCellID, localCoefficients, globalCoefficientsVector);
      for (int fluxMapRow=0; fluxMapRow<fluxIndices.size(); fluxMapRow++)
      {
        (*fluxMap)(fluxMapRow,fluxMapColumn) = localCoefficients(fluxIndices[fluxMapRow]);
      }
      fluxMapColumn++;
    }
    _fluxDuplicationMap[trialOrderRawPtr] = fluxMap;
  }
  return _fluxDuplicationMap[trialOrderRawPtr];
}

//! Dimension that will be used for neighbor relationships for Schwarz blocks, if CAMELLIA_ADDITIVE_SCHWARZ is the smoother choice.
int GMGOperator::getDimensionForSchwarzNeighborRelationship()
{
  return _dimensionForSchwarzNeighborRelationship;
}

//! Set dimension to use for neighbor relationships for Schwarz blocks.  Requires CAMELLIA_ADDITIVE_SCHWARZ as the smoother choice.
void GMGOperator::setDimensionForSchwarzNeighborRelationship(int value)
{
  _dimensionForSchwarzNeighborRelationship = value;
}

GlobalIndexType GMGOperator::getCoarseCellID(GlobalIndexType fineCellID) const
{
  const set<IndexType>* coarseCellIDs = &_coarseMesh->getTopology()->getActiveCellIndices();
  CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
  CellPtr ancestor = fineCell;
  RefinementBranch refBranch;
  while (coarseCellIDs->find(ancestor->cellIndex()) == coarseCellIDs->end())
  {
    CellPtr parent = ancestor->getParent();
    if (parent.get() == NULL)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestor for fine cell not found in coarse mesh");
    }
    unsigned childOrdinal = parent->childOrdinal(ancestor->cellIndex());
    refBranch.insert(refBranch.begin(), make_pair(parent->refinementPattern().get(), childOrdinal));
    ancestor = parent;
  }
  return ancestor->cellIndex();
}

Teuchos::RCP<GMGOperator> GMGOperator::getCoarseOperator()
{
  return _coarseOperator;
}

TSolutionPtr<double> GMGOperator::getCoarseSolution()
{
  return _coarseSolution;
}

SolverPtr GMGOperator::getCoarseSolver()
{
  return _coarseSolver;
}

Teuchos::RCP<DofInterpreter> GMGOperator::getFineDofInterpreter()
{
  return _fineDofInterpreter;
}

MeshPtr GMGOperator::getFineMesh() const
{
  return _fineMesh;
}

Epetra_CrsMatrix* GMGOperator::getFineStiffnessMatrix()
{
  return _fineStiffnessMatrix;
}

LocalDofMapperPtr GMGOperator::getLocalCoefficientMap(GlobalIndexType fineCellID) const
{
  GDAMinimumRule* gdaMinRuleFineMesh = dynamic_cast<GDAMinimumRule *>(_fineMesh->globalDofAssignment().get());

  // define lambda for filtering out fine dofs that don't belong:
  auto filterOutDuplicateFineDofs = [gdaMinRuleFineMesh, this] (BasisPtr fineBasis, CellPtr fineCell,
                                                                int sideOrdinal,
                                                                bool coarseInteriorDofsOnly,
                                                                RefinementBranch &refBranch,
                                                                SubBasisReconciliationWeights &weights) -> void
  {
    // Now: Zero out weights on fine dofs that are
    // - *not* interior to the coarse domain, or
    // - on interior subcells that are not owned by the child cell.
    GlobalIndexType fineCellID = fineCell->cellIndex();
    int minSubcellDimension = BasisReconciliation::minimumSubcellDimension(fineBasis);
    CellTopoPtr fineTopo = fineBasis->domainTopology();
    vector<vector<OwnershipInfo>> subcellOwnershipInfo = gdaMinRuleFineMesh->getCellConstraints(fineCellID).owningCellIDForSubcell;
    
    int sideDim = fineCell->topology()->getDimension() - 1;
    
    set<int> fineDofsToDiscard;
    for (int d=minSubcellDimension; d<=sideDim; d++)
    {
      for (int subcord=0; subcord<fineTopo->getSubcellCount(d); subcord++)
      {
        unsigned childCellSubcord = CamelliaCellTools::subcellOrdinalMap(fineCell->topology(), sideDim, sideOrdinal, d, subcord);
        bool shouldDiscard = false;
        if (subcellOwnershipInfo[d][childCellSubcord].cellID != fineCellID)
        {
          shouldDiscard = true;
        }
        else
        {
          if (d < sideDim)
          {
            // if we get here, this cell owns, but we need to check whether there is an earlier side on this cell that
            // also sees this subcell
            for (int earlierSideOrdinal = 0; earlierSideOrdinal < sideOrdinal; earlierSideOrdinal++)
            {
              CellTopoPtr earlierSide = fineCell->topology()->getSide(earlierSideOrdinal);
              int earlierSubcellCount = earlierSide->getSubcellCount(d);
              for (int earlierSubcord=0; earlierSubcord<earlierSubcellCount; earlierSubcord++)
              {
                unsigned earlierSubcordInChildCell = CamelliaCellTools::subcellOrdinalMap(fineCell->topology(),
                                                                                          sideDim, earlierSideOrdinal,
                                                                                          d, earlierSubcord);
                if (earlierSubcordInChildCell == childCellSubcord)
                {
                  shouldDiscard = true;
                  break;
                }
              }
            }
          }
          if (!shouldDiscard) // if shouldDiscard is already true, then we don't need to check whether subcell is interior
          {
            if (coarseInteriorDofsOnly)
            {
              pair<unsigned, unsigned> coarseSubcell = RefinementPattern::mapSubcellFromDescendantToAncestor(refBranch, d,
                                                                                                             childCellSubcord);
              unsigned coarseSubcellDim = coarseSubcell.first;
              if (coarseSubcellDim <= sideDim)
              {
                // *NOT* interior to the coarse cell
                shouldDiscard = true;
              }
            }
          }
        }
        if (shouldDiscard)
        {
          const vector<int>* dofOrdinals = &fineBasis->dofOrdinalsForSubcell(d, subcord);
          fineDofsToDiscard.insert(dofOrdinals->begin(),dofOrdinals->end());
        }
      }
    }
    int row = 0; // in weights container
    for (int fineDofOrdinal : weights.fineOrdinals)
    {
      if (fineDofsToDiscard.find(fineDofOrdinal) != fineDofsToDiscard.end())
      {
        // zero out
        for (int col=0; col<weights.weights.dimension(1); col++)
        {
          weights.weights(row,col) = 0;
        }
      }
      row++;
    }
    weights = BasisReconciliation::filterOutZeroRowsAndColumns(weights);
  };
  
  const set<IndexType>* coarseCellIDs = &_coarseMesh->getTopology()->getActiveCellIndices();
  CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
  CellPtr ancestor = fineCell;
  RefinementBranch refBranch;

  while (coarseCellIDs->find(ancestor->cellIndex()) == coarseCellIDs->end())
  {
    CellPtr parent = ancestor->getParent();
    if (parent.get() == NULL)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestor for fine cell not found in coarse mesh");
    }
    unsigned childOrdinal = parent->childOrdinal(ancestor->cellIndex());
    refBranch.insert(refBranch.begin(), make_pair(parent->refinementPattern().get(), childOrdinal));
    ancestor = parent;
  }
  GlobalIndexType coarseCellID = ancestor->cellIndex();
  CellPtr coarseCell = ancestor;
  if (_fineMesh->globalDofAssignment()->getH1Order(fineCellID).size() > 1)
  {
    // we have potentially multiple polynomial orders (one for space, one for time, say), whereas below we use just the
    // first polynomial order as a key.  This is OK, so long as there aren't anisotropic refinements in polynomial order
    // (as of this writing, we don't offer these).
    int rank = Teuchos::GlobalMPISession::getRank();
    static bool haveWarned = false;
    if ((rank==0) && (!haveWarned))
    {
      cout << "Note: using tensor-product polynomial orders in GMGOperator.  This is supported so long as no anisotropic";
      cout << " refinements are done in the polynomial order.  A small upgrade to GMGOperator's lookup tables would be";
      cout << " required to support anistropic p-refinements.\n";
      haveWarned = true;
    }
  }
  int fineOrder = _fineMesh->globalDofAssignment()->getH1Order(fineCellID)[0];
  int coarseOrder = _coarseMesh->globalDofAssignment()->getH1Order(coarseCellID)[0];

  DofOrderingPtr coarseTrialOrdering = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr;
  DofOrderingPtr fineTrialOrdering = _fineMesh->getElementType(fineCellID)->trialOrderPtr;

//  cout << "coarse trial ordering: \n";
//  coarseTrialOrdering->print(cout);
  
  pair< pair<int,int>, RefinementBranch > key = make_pair(make_pair(fineOrder, coarseOrder), refBranch);

  CondensedDofInterpreter<double>* condensedDofInterpreterCoarse = NULL;

  if (_useStaticCondensation)
  {
    condensedDofInterpreterCoarse = dynamic_cast<CondensedDofInterpreter<double>*>(_coarseSolution->getDofInterpreter().get());
  }
  
  // When doing static condensation for h-multigrid, the prolongation operator involves inversion of the
  // coarse local stiffness matrix to determine the fields on the coarse element; we take the traces of
  // these fields to determine the fine traces on the interior of the element.  Since these computations
  // may be spatially dependent, we can't reuse the prolongation that we determined for some other cell whose
  // refinement branch and element types match ours.
  bool cellProlongationCanMatchPatterns = (coarseCellID == fineCellID) || (condensedDofInterpreterCoarse == NULL);

  int fineSideCount = fineCell->getSideCount();
  int spaceDim = _fineMesh->getTopology()->getDimension();
  int sideDim = spaceDim - 1;
  vector<unsigned> ancestralSideOrdinals(fineSideCount);
  vector< RefinementBranch > sideRefBranches(fineSideCount);
  for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++)
  {
    ancestralSideOrdinals[sideOrdinal] = RefinementPattern::ancestralSubcellOrdinal(refBranch, sideDim, sideOrdinal);
    if (ancestralSideOrdinals[sideOrdinal] != -1)
    {
      sideRefBranches[sideOrdinal] = RefinementPattern::sideRefinementBranch(refBranch, sideOrdinal);
    }
  }

  LocalDofMapperPtr dofMapper;
  if (cellProlongationCanMatchPatterns && (_localCoefficientMap.find(key) != _localCoefficientMap.end()))
  {
    dofMapper = _localCoefficientMap[key];
  }
  else
  {
    Teuchos::RCP<Epetra_SerialDenseMatrix> coarseFluxToFieldMapMatrix;
    vector<int> fluxOrdinalToLocalDofIndex;
    set<int> condensibleVarIDs;
    MeshPtr coarseSingleCellMesh;
    if (!cellProlongationCanMatchPatterns)
    {
      // then we're doing static condensation in context of h-refinement: we'll need to compute coarseFluxToFieldMap
      coarseFluxToFieldMapMatrix = condensedDofInterpreterCoarse->fluxToFieldMapForIterativeSolves(coarseCellID);
      fluxOrdinalToLocalDofIndex = condensedDofInterpreterCoarse->fluxIndexLookupLocalCell(coarseCellID);
      condensibleVarIDs = condensedDofInterpreterCoarse->condensibleVariableIDs();

      Teuchos::RCP<Epetra_SerialDenseMatrix> fluxDuplicationMap = fluxDuplicationMapForCoarseCell(coarseCellID);
      if (fluxDuplicationMap != Teuchos::null)
      {
        Teuchos::RCP<Epetra_SerialDenseMatrix> modifiedMatrix = Teuchos::rcp( new Epetra_SerialDenseMatrix(coarseFluxToFieldMapMatrix->M(),
                                                                                                           coarseFluxToFieldMapMatrix->N()));
        coarseFluxToFieldMapMatrix->Apply(*fluxDuplicationMap, *modifiedMatrix);
        coarseFluxToFieldMapMatrix = modifiedMatrix;
      }
    }
    
    VarFactoryPtr vf = _fineMesh->bilinearForm()->varFactory();

    typedef vector< SubBasisDofMapperPtr > BasisMap; // taken together, these maps map a whole basis
    map< int, BasisMap > volumeMaps;

    vector< map< int, BasisMap > > sideMaps(fineSideCount);

    set<int> trialIDs = coarseTrialOrdering->getVarIDs();

    // set up fittability: which coarse dof ordinals have support on volume and each side
    set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume;
    vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides(fineSideCount);
    
    int coarseSideCount = coarseTrialOrdering->cellTopology()->getSideCount();
    
    vector<vector<double>> fineSideReferenceUnitNormals(fineSideCount);   // lazily filled for interior sides when static condensation is enabled
    vector<vector<double>> coarseSideReferenceUnitNormals(coarseSideCount); // lazily filled for interior sides when static condensation is enabled
    
    // for the moment, we skip the mapping from traces to fields based on traceTerm
    unsigned vertexNodePermutation = 0; // because we're "reconciling" to an ancestor, the views of the cells and sides are necessarily the same
    for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++)
    {
      int trialID = *trialIDIt;

      VarPtr trialVar = vf->trialVars().find(trialID)->second;

      if (coarseTrialOrdering->getSidesForVarID(trialID).size() == 1)   // field variable
      {
        if (condensedDofInterpreterCoarse != NULL)
        {
          if (condensedDofInterpreterCoarse->varDofsAreCondensible(trialID, 0, coarseTrialOrdering)) continue;
        }
//        cout << "Warning: for debugging purposes, skipping projection of fields in GMGOperator.\n";
        BasisPtr coarseBasis = coarseTrialOrdering->getBasis(trialID);
        BasisPtr fineBasis = fineTrialOrdering->getBasis(trialID);
        if (fineBasis->functionSpace() != coarseBasis->functionSpace())
        {
          // This will cause an error in BasisReconciliation; can we find equivalent continuous basis?
          // (necessary when using a non-conforming coarse mesh, though not at present an issue for fields, as here)
          coarseBasis = BasisFactory::basisFactory()->getContinuousBasis(coarseBasis);
          TEUCHOS_TEST_FOR_EXCEPTION(fineBasis->functionSpace() != coarseBasis->functionSpace(), std::invalid_argument, "Even after getting continuous basis for coarseBasis, fine and coarse function spaces disagree");
        }
        SubBasisReconciliationWeights weights = _br.constrainedWeights(fineBasis, refBranch, coarseBasis, vertexNodePermutation);
        
        vector<GlobalIndexType> coarseDofIndices;
        for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++)
        {
          unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(trialID, *coarseOrdinalIt);
          coarseDofIndices.push_back(coarseDofIndex);
        }
        fittableGlobalDofOrdinalsInVolume.insert(coarseDofIndices.begin(),coarseDofIndices.end());
        BasisMap basisMap(1,SubBasisDofMapper::subBasisDofMapper(weights.fineOrdinals, coarseDofIndices, weights.weights));
        volumeMaps[trialID] = basisMap;
//        cout << "weights for trial ID " << trialID << ":\n" << weights.weights;
      }
      else     // flux/trace
      {
//        cout << "Warning: for debugging purposes, skipping projection of fluxes and traces in GMGOperator.\n";
        for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++)
        {
          if (! fineTrialOrdering->hasBasisEntry(trialID, sideOrdinal)) continue;
          
          if (condensedDofInterpreterCoarse != NULL)
          {
            if (condensedDofInterpreterCoarse->varDofsAreCondensible(trialID, sideOrdinal, coarseTrialOrdering)) continue;
          }
          FunctionPtr sideParity = Function::sideParity();
          
          unsigned coarseSideOrdinal = ancestralSideOrdinals[sideOrdinal];
          BasisPtr coarseBasis, fineBasis;
          BasisMap basisMap;
          bool useTermTracedIfAvailable = true; // flag for debugging purposes (true is the production setting)
          if (coarseSideOrdinal == -1)   // the fine side falls inside a coarse volume
          {
            // we map trace to field using the traceTerm LinearTermPtr
            VarPtr trialVar = vf->trial(trialID);

            LinearTermPtr termTraced = trialVar->termTraced();
            if (!useTermTracedIfAvailable || (termTraced.get() == NULL)) // nothing we can do if we don't know what term we're tracing
              continue;

            if (! fineTrialOrdering->hasBasisEntry(trialID, sideOrdinal) ) continue;
            fineBasis = fineTrialOrdering->getBasis(trialID, sideOrdinal);

//            cout << "Processing termTraced for variable " << trialVar->name() << endl;
            
            set<int> varsTraced = termTraced->varIDs();
            for (set<int>::iterator varTracedIt = varsTraced.begin(); varTracedIt != varsTraced.end(); varTracedIt++)
            {
              int varTracedID = *varTracedIt;
              
              if ((condensedDofInterpreterCoarse == NULL) || (condensibleVarIDs.find(varTracedID) == condensibleVarIDs.end())) // the latter case is like the pressure for Stokes VGP
              {
                coarseBasis = coarseTrialOrdering->getBasis(varTracedID);
                
                unsigned coarseSubcellOrdinal = 0, coarseDomainOrdinal = 0; // the volume
                unsigned coarseSubcellPermutation = 0;
                unsigned fineSubcellOrdinalInFineDomain = 0; // the side is the whole fine domain...
                SubBasisReconciliationWeights weights = _br.constrainedWeightsForTermTraced(termTraced, varTracedID,
                                                                                            sideDim, fineBasis, fineSubcellOrdinalInFineDomain, refBranch, sideOrdinal,
                                                                                            ancestor->topology(),
                                                                                            spaceDim, coarseBasis, coarseSubcellOrdinal, coarseDomainOrdinal, coarseSubcellPermutation);
                // Now: Zero out weights on fine dofs that are
                // - *not* interior to the coarse domain, or
                // - on interior subcells that are not owned by the child cell.
                filterOutDuplicateFineDofs(fineBasis, fineCell, sideOrdinal, true, refBranch, weights);
                // NOTE: applying the above filter is a bit fragile: since we are doing pattern-matching, it relies
                //       on ownership of the fine dofs on the interior of a refined element being assigned in the same
                //       way on every refined element.  This is in fact the case in GDAMinimumRule, and it seems a natural
                //       thing, but if it were to change, that would be a problem.
                
                // **** end new weight-adjusting code ****
                
                vector<GlobalIndexType> coarseDofIndices;
                for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++)
                {
                  unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(varTracedID, *coarseOrdinalIt);
                  coarseDofIndices.push_back(coarseDofIndex);
                  fittableGlobalDofOrdinalsOnSides[sideOrdinal].insert(coarseDofIndex);
  //                cout << "termTraced weights:\n" << weights.weights;
                }

                basisMap.push_back(SubBasisDofMapper::subBasisDofMapper(weights.fineOrdinals, coarseDofIndices, weights.weights));
              }
              else
              {
                // If we're doing static condensation, map from coarse flux to coarse field using CondensedDofInterpreter, then
                // back to fluxes using termTraced.
                
                BasisPtr volumeBasis = coarseTrialOrdering->getBasis(varTracedID);
                CellTopoPtr volumeTopo = volumeBasis->domainTopology();
                
                unsigned volumeSubcellOrdinal = 0, volumeDomainOrdinal = 0;
                unsigned volumeSubcellPermutation = 0;
                unsigned fineSubcellOrdinalInFineDomain = 0; // the side is the whole fine domain...
                SubBasisReconciliationWeights weights = _br.constrainedWeightsForTermTraced(termTraced, varTracedID,
                                                                                            sideDim, fineBasis,
                                                                                            fineSubcellOrdinalInFineDomain, refBranch,
                                                                                            sideOrdinal, volumeTopo,
                                                                                            spaceDim, volumeBasis, volumeSubcellOrdinal,
                                                                                            volumeDomainOrdinal, volumeSubcellPermutation);
                
                weights = BasisReconciliation::filterOutZeroRowsAndColumns(weights);
                
                // TODO: treat the case (as with pressure in Stokes VGP) where termTraced includes some field variables that are not condensible.
                
//                print("weights, fineOrdinals", weights.fineOrdinals);
//                print("weights, coarseOrdinals", weights.coarseOrdinals);
//                cout << "weights, weights:\n" << weights.weights;
                
                // extract a Epetra_SerialDenseMatrix corresponding to just the varTracedID dofs
                set<int> volumeBasisOrdinals = weights.coarseOrdinals;
                Epetra_SerialDenseMatrix coarseFluxToFieldTraced(volumeBasisOrdinals.size(),coarseFluxToFieldMapMatrix->ColDim()); // columns here correspond to our coarse dof indices -- what we map to
                
                vector<int> fieldTracedIndices = condensedDofInterpreterCoarse->fieldRowIndices(coarseCellID, varTracedID);
                int reducedRowIndex = 0;
                for (int volumeBasisOrdinal : volumeBasisOrdinals)
                {
                  int fieldTracedIndex = fieldTracedIndices[volumeBasisOrdinal];
                  int numCols = coarseFluxToFieldMapMatrix->ColDim();
                  for (int col = 0; col < numCols; col++)
                  {
                    coarseFluxToFieldTraced(reducedRowIndex,col) = (*coarseFluxToFieldMapMatrix)(fieldTracedIndex,col);
                  }
                  reducedRowIndex++;
                }
                int n = weights.weights.dimension(0);
                int m = weights.weights.dimension(1);
                if ((n==0) || (m==0))
                {
                  continue;
                }
                double *firstEntry = (double *) &weights.weights[0];
                Epetra_SerialDenseMatrix fineBasisToCoarseFieldMatrix(::Copy,firstEntry,m,m,n);
                fineBasisToCoarseFieldMatrix.SetUseTranspose(true); // use transpose, because SDM is column-major, while FC is row-major
                
                Epetra_SerialDenseMatrix fineBasisToCoarseFluxMatrix(weights.fineOrdinals.size(),coarseFluxToFieldTraced.ColDim());
                fineBasisToCoarseFieldMatrix.Apply(coarseFluxToFieldTraced, fineBasisToCoarseFluxMatrix);
                
                vector<GlobalIndexType> coarseDofIndicesVector;
                
                // Look for zero columns in fineBasisToCoarseFluxMatrix
                vector<unsigned> nonZeroColumnIndices;
                map<GlobalIndexType,unsigned> columnOrdinalForCoarseDofIndex; // allows sorting by coarse dof index
                double tol = 1e-15;
                for (int j=0; j<fineBasisToCoarseFluxMatrix.ColDim(); j++)
                {
                  bool nonZero = false;
                  for (int i=0; i<fineBasisToCoarseFluxMatrix.RowDim(); i++)
                  {
                    if (abs(fineBasisToCoarseFluxMatrix(i,j)) > tol)
                    {
                      nonZero = true;
                      nonZeroColumnIndices.push_back(j);
                      columnOrdinalForCoarseDofIndex[fluxOrdinalToLocalDofIndex[j]] = coarseDofIndicesVector.size();
                      coarseDofIndicesVector.push_back(fluxOrdinalToLocalDofIndex[j]);
                      break;
                    }
                  }
                }
                
                vector<unsigned> permutedNonZeroColumnIndices;
                for (auto entry : columnOrdinalForCoarseDofIndex) {
                  permutedNonZeroColumnIndices.push_back(nonZeroColumnIndices[entry.second]);
                }
                
                weights.weights.resize(fineBasisToCoarseFluxMatrix.RowDim(),nonZeroColumnIndices.size());
                for (int i=0; i<weights.weights.dimension(0); i++)
                {
                  for (int j=0; j<weights.weights.dimension(1); j++)
                  {
                    weights.weights(i,j) = fineBasisToCoarseFluxMatrix(i,permutedNonZeroColumnIndices[j]);
                  }
                }
                weights.coarseOrdinals.clear();
                weights.coarseOrdinals.insert(coarseDofIndicesVector.begin(),coarseDofIndicesVector.end());
                
//                print("coarseDofIndices", weights.coarseOrdinals);
//                cout << "weights:\n" << weights.weights;
                
                // Now: Zero out weights on fine dofs that are
                // - *not* interior to the coarse domain, or
                // - on interior subcells that are not owned by the child cell.
                filterOutDuplicateFineDofs(fineBasis, fineCell, sideOrdinal, true, refBranch, weights);
                
                coarseDofIndicesVector.clear();
                coarseDofIndicesVector.insert(coarseDofIndicesVector.begin(),
                                              weights.coarseOrdinals.begin(), weights.coarseOrdinals.end());
                // **** end new weight-adjusting code ****
                
                basisMap.push_back(SubBasisDofMapper::subBasisDofMapper(weights.fineOrdinals, coarseDofIndicesVector, weights.weights));
                fittableGlobalDofOrdinalsOnSides[sideOrdinal].insert(coarseDofIndicesVector.begin(),coarseDofIndicesVector.end());
              }
            }
          }
          else     // fine side maps to a coarse side
          {
            if (! coarseTrialOrdering->hasBasisEntry(trialID, coarseSideOrdinal))
            {
              cout << "ERROR: no entry for trial var " << trialVar->name() << " on side " << coarseSideOrdinal << " in coarse mesh.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missing basis entry on coarse side!");
            }
            if (! fineTrialOrdering->hasBasisEntry(trialID, sideOrdinal))
            {
              cout << "ERROR: no entry for trial var " << trialVar->name() << " on side " << sideOrdinal << " in fine mesh.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missing basis entry on fine side!");
            }
            coarseBasis = coarseTrialOrdering->getBasis(trialID, coarseSideOrdinal);
            fineBasis = fineTrialOrdering->getBasis(trialID, sideOrdinal);
            if (fineBasis->functionSpace() != coarseBasis->functionSpace())
            {
              // This will cause an error in BasisReconciliation; can we find equivalent continuous basis?
              // (necessary when using a non-conforming coarse mesh, though not at present an issue for fields, as here)
              coarseBasis = BasisFactory::basisFactory()->getContinuousBasis(coarseBasis);
              TEUCHOS_TEST_FOR_EXCEPTION(fineBasis->functionSpace() != coarseBasis->functionSpace(), std::invalid_argument, "Even after getting continuous basis for coarseBasis, fine and coarse function spaces disagree");
            }
            SubBasisReconciliationWeights weights = _br.constrainedWeights(fineBasis, sideRefBranches[sideOrdinal], coarseBasis, vertexNodePermutation);
            
            vector<GlobalIndexType> coarseDofIndices(weights.coarseOrdinals.size());
            int i = 0;
            for (int coarseOrdinal : weights.coarseOrdinals)
            {
              coarseDofIndices[i++] = coarseTrialOrdering->getDofIndex(trialID, coarseOrdinal, coarseSideOrdinal);
            }

            basisMap.push_back(SubBasisDofMapper::subBasisDofMapper(weights.fineOrdinals, coarseDofIndices, weights.weights));
          }

          sideMaps[sideOrdinal][trialID] = basisMap;
        }
      }
    }

    int coarseDofCount = coarseTrialOrdering->totalDofs();
    set<GlobalIndexType> allCoarseDofs; // include these even when not mapped (which can happen with static condensation) to guarantee that the dof mapper interprets coarse cell data correctly...
    for (int coarseOrdinal=0; coarseOrdinal<coarseDofCount; coarseOrdinal++)
    {
      allCoarseDofs.insert(coarseOrdinal);
    }
    
    // set up side maps from fine to coarse cell
    map<int,int> fineSideToCoarseSide;
    int fineSideCount = _fineMesh->getElementType(fineCellID)->cellTopoPtr->getSideCount();
    for (int fineSideOrdinal=0; fineSideOrdinal < fineSideCount; fineSideOrdinal++)
    {
      // is there a corresponding coarse side ordinal?
      RefinementBranch refBranch;
      GlobalIndexType ancestralCellID = fineCellID;
      while (ancestralCellID != coarseCellID) {
        CellPtr ancestralCell = _fineMesh->getTopology()->getCell(ancestralCellID);
        int childOrdinal = ancestralCell->getParent()->findChildOrdinal(ancestralCellID);
        TEUCHOS_TEST_FOR_EXCEPTION(childOrdinal==-1, std::invalid_argument, "internal error: cell not found in its own parent");
        refBranch.insert(refBranch.begin(),{ancestralCell->getParent()->refinementPattern().get(), childOrdinal});
        ancestralCellID = ancestralCell->getParent()->cellIndex();
      }
      int coarseSideOrdinal = RefinementPattern::mapSideOrdinalFromLeafToAncestor(fineSideOrdinal, refBranch);
      fineSideToCoarseSide[fineSideOrdinal] = coarseSideOrdinal; // -1 if there isn't a corresponding coarse side ordinal
    }
    
    for (int trialID : trialIDs)
    {
      VarPtr trialVar = vf->trial(trialID);

      const vector<int>* fineSidesForVarID = &fineTrialOrdering->getSidesForVarID(trialID);
      if (fineSidesForVarID->size() == 1)   // field variable
      {
        if ((condensedDofInterpreterCoarse == NULL) || (condensedDofInterpreterCoarse->varDofsAreCondensible(trialID, 0, fineTrialOrdering)))
        {
          vector<int> dofIndices = coarseTrialOrdering->getDofIndices(trialID);
          fittableGlobalDofOrdinalsInVolume.insert(dofIndices.begin(),dofIndices.end());
        }
      }
      else
      {
        for (int fineSideOrdinal : *fineSidesForVarID)
        {
          int coarseSideOrdinal = fineSideToCoarseSide[fineSideOrdinal];
          if (coarseSideOrdinal != -1)
          {
            vector<int> dofIndices = coarseTrialOrdering->getDofIndices(trialID,coarseSideOrdinal);
            fittableGlobalDofOrdinalsOnSides[fineSideOrdinal].insert(dofIndices.begin(),dofIndices.end());
          }
        }
      }
    }

    dofMapper = Teuchos::rcp( new LocalDofMapper(fineTrialOrdering, volumeMaps, fittableGlobalDofOrdinalsInVolume,
                                                 sideMaps, fittableGlobalDofOrdinalsOnSides, allCoarseDofs) );
    
//    dofMapper->printMappingReport();
    if (cellProlongationCanMatchPatterns)
      _localCoefficientMap[key] = dofMapper;
  }

  // now, correct side parities in dofMapper if the ref space situation differs from the physical space one.
  FieldContainer<double> coarseCellSideParities = _coarseMesh->globalDofAssignment()->cellSideParitiesForCell(coarseCellID);
  FieldContainer<double> fineCellSideParities = _fineMesh->globalDofAssignment()->cellSideParitiesForCell(fineCellID);
//  cout << "fineCell parities:\n" << fineCellSideParities;
  set<unsigned> fineSidesToCorrect;
  for (unsigned fineSideOrdinal=0; fineSideOrdinal<fineSideCount; fineSideOrdinal++)
  {
    unsigned coarseSideOrdinal = ancestralSideOrdinals[fineSideOrdinal];
    if (coarseSideOrdinal != -1)   // ancestor shares side
    {
      double coarseParity = coarseCellSideParities(0,coarseSideOrdinal);
      double fineParity = fineCellSideParities(0,fineSideOrdinal);
      if (coarseParity != fineParity)
      {
        fineSidesToCorrect.insert(fineSideOrdinal);
//        cout << "fine side to correct on cell " << fineCellID << ": " << fineSideOrdinal << endl;
      }
    }
    else
    {
      // when we have done a map from field to trace, no parities have been taken into account; the termTraced will involve
      // the outward facing normal, but not the potential negation of this to make neighbors agree.  Therefore, we need to "correct"
      // any interior fine side whose parity is -1.

      double fineParity = fineCellSideParities(0,fineSideOrdinal);
      if (fineParity < 0)
      {
        fineSidesToCorrect.insert(fineSideOrdinal);
        //        cout << "fine side to correct on cell " << fineCellID << ": " << fineSideOrdinal << endl;
      }
    }
  }

  if (fineSidesToCorrect.size() > 0)
  {
    // copy before changing dofMapper:
    dofMapper = Teuchos::rcp( new LocalDofMapper(*dofMapper.get()) );
    set<int> fluxIDs;
    VarFactoryPtr vf = _fineMesh->bilinearForm()->varFactory();
    vector<VarPtr> fluxVars = vf->fluxVars();
    for (vector<VarPtr>::iterator fluxIt = fluxVars.begin(); fluxIt != fluxVars.end(); fluxIt++)
    {
      fluxIDs.insert((*fluxIt)->ID());
    }
    dofMapper->reverseParity(fluxIDs, fineSidesToCorrect);
  }

  return dofMapper;
}

int GMGOperator::getOperatorLevel() const
{
  if (_coarseOperator != Teuchos::null)
    return _coarseOperator->getOperatorLevel() + 1;
  else
    return 0;
}

int GMGOperator::getSmootherApplicationCount() const
{
  return _smootherApplicationCount;
}

GMGOperator::SmootherChoice GMGOperator::getSmootherType()
{
  return _smootherType;
}

Teuchos::RCP<Epetra_MultiVector> GMGOperator::getSmootherWeightVector()
{
  return _smootherDiagonalWeight;
}

int GMGOperator::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  // Belos thinks preconditioners *are* inverses, where Aztec thinks they're forward operators.  We do the same thing regardless of which you call...
  return ApplyInverse(X,Y);
  //  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method.");}
}

int GMGOperator::ApplyInverse(const Epetra_MultiVector& X_in, Epetra_MultiVector& Y) const
{
  narrate("ApplyInverse");
  //  cout << "GMGOperator::ApplyInverse.\n";
  int rank = Teuchos::GlobalMPISession::getRank();
  bool printVerboseOutput = (rank==0) && _debugMode;
  
  Epetra_Time timer(Comm());
  
  if (_debugMode)
  {
    if (printVerboseOutput) cout << "X_in:\n";
    X_in.Comm().Barrier();
    cout << X_in;
  }
  
  Epetra_MultiVector res(X_in); // Res: residual.  Starting with Y = 0, then this is just the RHS
  Epetra_MultiVector f(X_in);   // f: the RHS.  Don't change this.
  
  // initialize Y (important to do this only after X_in has been copied--X and Y can be in the same location)
  Y.PutScalar(0.0);
  Epetra_MultiVector A_Y(Y.Map(), Y.NumVectors());
  
  if ((_multigridStrategy == FULL_MULTIGRID_V) || (_multigridStrategy == FULL_MULTIGRID_W))
  {
    // full multigrid takes the coarse operator applied to the RHS as its initial guess
    Epetra_MultiVector Y2(Y.Map(), Y.NumVectors());
    this->ApplyInverseCoarseOperator(res, Y2);
    Y.Update(1.0, Y2, 1.0);
    // recompute residual:
    res = f;
    computeResidual(Y,res,A_Y);
  }
  
  int numApplications;
  if ((_multigridStrategy == W_CYCLE) || (_multigridStrategy == FULL_MULTIGRID_W))
  {
    numApplications = 2;
  }
  else
  {
    numApplications = 1;
  }
  
  if (_smootherType != NONE)
  {
    Epetra_MultiVector B1_res(Y.Map(), Y.NumVectors()); // B1_res: the smoother applied to res.
    for (int i=0; i<_smootherApplicationCount; i++)
    {
      // if we have a smoother S, set Y = S^-1 f =: B1 * f
      ApplySmoother(res, B1_res, true); // true: apply weight on left
      Y.Update(1.0, B1_res, 1.0);
      if (_debugMode)
      {
        if (printVerboseOutput) cout << "B1 * res:\n";
        B1_res.Comm().Barrier();
        cout << B1_res;
      }
      
      if (_multigridStrategy == TWO_LEVEL)
      {
        if (_useSchwarzDiagonalWeight && (_smootherDiagonalWeight != Teuchos::null))
        {
          // then we need to average the left-application we already did with an application on the right
          ApplySmoother(res, B1_res, false); // false: apply weight on right
          Y.Update(0.5, B1_res, 0.5); // 0.5: average
        }
        break; // don't apply multiple times if doing two-level (no point...)
      }
      else
      {
        // compute a new residual: res := f - A*Y = res - A*Y
        res = f;
        computeResidual(Y,res,A_Y);
      }
    }
  }
  
  for (int applicationOrdinal = 0; applicationOrdinal < numApplications; applicationOrdinal++)
  {
    Epetra_MultiVector Y2(Y.Map(), Y.NumVectors());
    this->ApplyInverseCoarseOperator(res, Y2);
    Y.Update(1.0, Y2, 1.0);
    
    if ((_smootherType != NONE) && (_multigridStrategy != TWO_LEVEL))
    {
      Epetra_MultiVector B1_res(Y.Map(), Y.NumVectors());
      
      for (int i=0; i<_smootherApplicationCount; i++)
      {
        // compute Y + B1 * (f - A*y)
        res = f;
        computeResidual(Y, res, A_Y);
        
        if (((_useSchwarzDiagonalWeight && (_smootherDiagonalWeight != Teuchos::null)))
            && (numApplications > 1) && (applicationOrdinal < numApplications-1))
        {
          // then, to maintain symmetry, apply smoother twice, once with left weighting, and once with right:
          ApplySmoother(res, B1_res, true);
          Y.Update(1.0, B1_res, 1.0);
          res = f;
          computeResidual(Y, res, A_Y);
          ApplySmoother(res, B1_res, false);
          Y.Update(1.0, B1_res, 1.0);
        }
        else
        {
          // just apply once, on the right
          ApplySmoother(res, B1_res, false);
          Y.Update(1.0, B1_res, 1.0);
        }
      }
    }
    
    if (_debugMode)
    {
      if (printVerboseOutput) cout << "Y:\n";
      Y.Comm().Barrier();
      cout << Y;
    }
    if (applicationOrdinal < numApplications-1)
    {
      // another application will follow, so let's recompute the residual
      res = f;
      computeResidual(Y, res, A_Y);
    }
  }
  
  /*
   We zero out the lagrange multiplier solutions on the fine mesh, because we're relying on the coarse solve to impose these;
   we just use an identity block in the lower right of the fine matrix for the Lagrange constraints themselves.
   
   The argument for this, at least for zero mean constraints:
   If the coarse solution has zero mean, then the fact that the prolongation operator is exact (i.e. coarse mesh solution is
   exactly reproduced on fine mesh) means that the fine mesh solution will have zero mean.
   
   This neglects the smoothing operator.  If the smoothing operator isn't guaranteed to produce an update with zero mean,
   then this may not work.
   */
  GlobalIndexType firstFineConstraintRowIndex = _fineDofInterpreter->globalDofCount();
  for (GlobalIndexTypeToCast fineRowIndex=firstFineConstraintRowIndex; fineRowIndex<Y.GlobalLength(); fineRowIndex++)
  {
    int LID = Y.Map().LID(fineRowIndex);
    if (LID != -1)
    {
      Y[0][LID] = 0.0;
    }
  }
  
  return 0;
}


int GMGOperator::ApplyInverseCoarseOperator(const Epetra_MultiVector &res, Epetra_MultiVector &Y) const
{
  int rank = Teuchos::GlobalMPISession::getRank();
  bool printVerboseOutput = (rank==0) && _debugMode;

  Teuchos::RCP<Epetra_FEVector> coarseRHSVector = _coarseSolution->getRHSVector();
  
  if (coarseRHSVector->GlobalLength() != prolongationColCount())
  {
    // TODO: add support for coarseRHSVector that may have lagrange/zmc constraints applied even though fine solution neglects these...
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseRHSVector->GlobalLength() != _P->NumGlobalCols()");
  }
  
  Epetra_Time timer(Comm());
  
  narrate("multiply _P * coarseRHSVector");
  if (printVerboseOutput) cout << "calling _P->Multiply(true, res, *coarseRHSVector);\n";
  if (_debugMode)
  {
    if (rank==0) cout << "res:\n";
    res.Comm().Barrier();
    cout << res;
    res.Comm().Barrier();
  }
  _P->Multiply(!_fineCoarseRolesSwapped, res, *coarseRHSVector);
  if (printVerboseOutput) cout << "finished _P->Multiply(!_fineCoarseRolesSwapped, X, *coarseRHSVector);\n";
  if (_debugMode)
  {
    if (rank==0) cout << "coarseRHSVector:\n";
    coarseRHSVector->Comm().Barrier();
    cout << *coarseRHSVector;
    coarseRHSVector->Comm().Barrier();
  }
  _timeMapFineToCoarse += timer.ElapsedTime();
  
  Teuchos::RCP<Epetra_FEVector> coarseLHSVector;
  
  timer.ResetStartTime();
  if (_coarseSolver == Teuchos::null)
  {
    // then we must be at a finer level than the coarsest, and the appropriate thing is to simply apply
    // our _coarseOperator
    TEUCHOS_TEST_FOR_EXCEPTION(_coarseOperator == Teuchos::null, std::invalid_argument, "GMGOperator internal error: _coarseOperator and _coarseSolver are both null");
    _coarseOperator->ApplyInverse(*coarseRHSVector, *_coarseSolution->getLHSVector());
  }
  else if (!_haveSolvedOnCoarseMesh)
  {
    if (printVerboseOutput) cout << "solving on coarse mesh\n";
    narrate("_coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, false)");
    _coarseSolution->setProblem(_coarseSolver);
    _coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, false);
    if (printVerboseOutput) cout << "finished solving on coarse mesh\n";
    _haveSolvedOnCoarseMesh = true;
  }
  else
  {
    if (printVerboseOutput) cout << "re-solving on coarse mesh\n";
    narrate("_coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, true)");
    _coarseSolver->setRHS(coarseRHSVector);
    _coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, true); // call resolve() instead of solve() -- reuse factorization
    if (printVerboseOutput) cout << "finished re-solving on coarse mesh\n";
  }
  _timeCoarseSolve += timer.ElapsedTime();
  
  timer.ResetStartTime();
  if (printVerboseOutput) cout << "calling _coarseSolution->getLHSVector()\n";
  coarseLHSVector = _coarseSolution->getLHSVector();
  
  if (_debugMode)
  {
    if (rank==0) cout << "coarseLHSVector:\n";
    coarseLHSVector->Comm().Barrier();
    cout << *coarseLHSVector;
    coarseLHSVector->Comm().Barrier();
  }
  
  if (printVerboseOutput) cout << "finished _coarseSolution->getLHSVector()\n";
  if (printVerboseOutput) cout << "calling _P->Multiply(_fineCoarseRolesSwapped, *coarseLHSVector, Y)\n";
  narrate("multiply _P * coarseLHSVector");

  _P->Multiply(_fineCoarseRolesSwapped, *coarseLHSVector, Y);
  if (printVerboseOutput) cout << "finished _P->Multiply(_fineCoarseRolesSwapped, *coarseLHSVector, Y)\n";
  _timeMapCoarseToFine += timer.ElapsedTime();
  
  if (_debugMode)
  {
    if (rank==0) cout << "_P * coarseLHSVector:\n";
    Y.Comm().Barrier();
    cout << Y;
    Y.Comm().Barrier();
  }
  return 0;
}

int GMGOperator::ApplySmoother(const Epetra_MultiVector &res, Epetra_MultiVector &Y, bool weightOnLeft) const
{
  
  narrate("ApplySmoother()");
  Epetra_Time timer(Comm());
  
  int err;

  if ((_smootherDiagonalWeight == Teuchos::null) || !_useSchwarzDiagonalWeight)
  {
    err = _smoother->ApplyInverse(res, Y);
  }
  else
  {
//    cout << "_smootherDiagonalWeight:\n";
//    Comm().Barrier();
//    cout << *_smootherDiagonalWeight;
//    Comm().Barrier();
//    cout << "res:\n";
//    cout << res;
    Epetra_MultiVector temp(res.Map(),res.NumVectors());
    if (!weightOnLeft)
      temp.Multiply(1.0, res, *_smootherDiagonalWeight, 0.0);
    else
      temp = res;
    
//    cout << "temp:\n";
//    Comm().Barrier();
//    cout << temp;
    
    err = _smoother->ApplyInverse(temp, Y);
//    cout << "Y after ApplyInverse:\n";
//    Comm().Barrier();
//    cout << Y;
    
    if (weightOnLeft)
    {
      temp = Y;
      Y.Multiply(1.0, temp, *_smootherDiagonalWeight, 0.0);
    }
//    cout << "Y:\n";
//    Comm().Barrier();
//    cout << Y;
//    Comm().Barrier();
  }
  Y.Scale(_smootherWeight);
  _timeApplySmoother += timer.ElapsedTime();
  
  if (err != 0)
  {
    cout << "_smoother->ApplyInverse returned non-zero error code " << err << endl;
  }
  return err;
}

double GMGOperator::NormInf() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method.");
}

const char * GMGOperator::Label() const
{
  return "Camellia Geometric Multi-Grid Preconditioner";
}

int GMGOperator::SetUseTranspose(bool UseTranspose)
{
  return -1; // not supported for now.  (wouldn't be hard, but I don't see the point.)
}

bool GMGOperator::UseTranspose() const
{
  return false; // not supported for now.  (wouldn't be hard, but I don't see the point.)
}

//! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
bool GMGOperator::HasNormInf() const
{
  return false;
}

//! Returns a pointer to the Epetra_Comm communicator associated with this operator.
const Epetra_Comm & GMGOperator::Comm() const
{
  return _finePartitionMap.Comm();
}

//! Returns the Epetra_Map object associated with the domain of this operator.
const Epetra_Map & GMGOperator::OperatorDomainMap() const
{
  return _finePartitionMap;
}

//! Returns the Epetra_Map object associated with the range of this operator.
const Epetra_Map & GMGOperator::OperatorRangeMap() const
{
  return _finePartitionMap;
}

int GMGOperator::prolongationColCount() const
{
  return _fineCoarseRolesSwapped ? _P->NumGlobalRows() : _P->NumGlobalCols();
}

int GMGOperator::prolongationRowCount() const
{
  return _fineCoarseRolesSwapped ? _P->NumGlobalCols() : _P->NumGlobalRows();
}

TimeStatistics GMGOperator::getStatistics(double timeValue) const
{
  TimeStatistics stats;
  int indexBase = 0;
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_Map timeMap(numProcs,indexBase,Comm());
  Epetra_Vector timeVector(timeMap);
  timeVector[0] = timeValue;

  int err = timeVector.Norm1( &stats.sum );
  err = timeVector.MeanValue( &stats.mean );
  err = timeVector.MinValue( &stats.min );
  err = timeVector.MaxValue( &stats.max );

  return stats;
}

void GMGOperator::reportTimings(StatisticChoice whichStat) const
{
  reportTimings(whichStat,false);
}

void GMGOperator::reportTimings(StatisticChoice whichStat, bool sumAllOperators) const
{
  //   mutable double _timeMapFineToCoarse, _timeMapCoarseToFine, _timeCoarseImport, _timeConstruction, _timeCoarseSolve;  // totals over the life of the object
  int rank = Teuchos::GlobalMPISession::getRank();
  
  map<string, double> reportValues = sumAllOperators ? timingReportSumOfOperators() : timingReport() ;
  
  if (rank == 0)
  {
    if (sumAllOperators)
      cout << "Sum of all grid levels, ";
    switch (whichStat)
    {
      case MIN:
        cout << "Timing MINIMUM values:\n";
        break;
      case MAX:
        cout << "Timing MAXIMUM values:\n";
        break;
      case MEAN:
        cout << "Timing mean values:\n";
        break;
      case SUM:
        cout << "SUM of timing values:\n";
        break;
      case ALL:
        break;
    }
  }
  for (map<string,double>::iterator reportIt = reportValues.begin(); reportIt != reportValues.end(); reportIt++)
  {
    TimeStatistics stats = getStatistics(reportIt->second);
    if (rank==0)
    {
      cout << setprecision(2);
      cout << std::scientific;
      cout << setw(35) << reportIt->first << ": ";
      switch (whichStat) {
        case ALL:
          cout << endl;
          cout <<  "mean = " << stats.mean << " seconds\n";
          cout << "max =  " << stats.max << " seconds\n";
          cout << "min =  " << stats.min << " seconds\n";
          cout << "sum =  " << stats.sum << " seconds\n";
          break;
        case MIN:
          cout << stats.min << " seconds\n";
          break;
        case MAX:
          cout << stats.max << " seconds\n";
          break;
        case MEAN:
          cout << stats.mean << " seconds\n";
          break;
        case SUM:
          cout << stats.sum << " seconds\n";
          break;
      }
    }
  }
}

void GMGOperator::reportTimingsSumOfOperators(StatisticChoice whichStat) const
{
  reportTimings(whichStat,true);
}

std::map<string, double> GMGOperator::timingReport() const
{
  map<string, double> reportValues;
  reportValues["apply fine stiffness"] = _timeApplyFineStiffness;
  reportValues["apply smoother"] = _timeApplySmoother;
  reportValues["total construction time"] = _timeConstruction;
  reportValues["construct prolongation operator"] = _timeProlongationOperatorConstruction;
  reportValues["construct local coefficient maps"] = _timeLocalCoefficientMapConstruction;
  reportValues["coarse import"] = _timeCoarseImport;
  reportValues["coarse solve"] = _timeCoarseSolve;
  reportValues["map coarse to fine"] = _timeMapCoarseToFine;
  reportValues["map fine to coarse"] = _timeMapFineToCoarse;
  reportValues["compute coarse stiffness matrix"] = _timeComputeCoarseStiffnessMatrix;
  reportValues["set up smoother"] = _timeSetUpSmoother;
  reportValues["update coarse operator"] = _timeUpdateCoarseOperator;
  return reportValues;
}

std::map<string, double> GMGOperator::timingReportSumOfOperators() const
{
  map<string, double> reportValues = timingReport();
  Teuchos::RCP<GMGOperator> coarseOperator = _coarseOperator;
  while (coarseOperator != Teuchos::null)
  {
    map<string,double> coarseReportValues = coarseOperator->timingReport();
    for (pair<string,double> entry : coarseReportValues)
    {
      // the "coarse solve" entry we should take from the coarsest operator, and not sum
      if (entry.first != "coarse solve")
      {
        reportValues[entry.first] += entry.second;
      }
      else
      {
        reportValues[entry.first] = entry.second;
      }
    }
    coarseOperator = coarseOperator->getCoarseOperator();
  }
  
  reportValues.erase("update coarse operator"); // this one's costs are accounted for in other entries from the coarse operators (set up smoother and compute coarse stiffness)
  return reportValues;
}

void GMGOperator::setCoarseOperator(Teuchos::RCP<GMGOperator> coarseOperator)
{
  _coarseOperator = coarseOperator;
  _coarseOperator->setIsFinest(false);
}

void GMGOperator::setCoarseSolver(SolverPtr coarseSolver)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_coarseOperator != Teuchos::null, std::invalid_argument, "coarseSolver may not be set when _coarseOperator is set");
  _coarseSolver = coarseSolver;
}

void GMGOperator::setDebugMode(bool value)
{
  _debugMode = value;
}

//! When true, the roles of fine and coarse are swapped during prolongation operator construction,
//! and the transpose of the prolongation operator is used.
bool GMGOperator::getFineCoarseRolesSwapped() const
{
  return _fineCoarseRolesSwapped;
}

//! When true, the roles of fine and coarse are swapped during prolongation operator construction,
//! and the transpose of the prolongation operator is used.
void GMGOperator::setFineCoarseRolesSwapped(bool value)
{
  if (value != _fineCoarseRolesSwapped)
  {
    _fineCoarseRolesSwapped = value;
    // force reconstruction of _P, if it has already been computed:
    _P = Teuchos::null;
  }
}

void GMGOperator::setFineStiffnessMatrix(Epetra_CrsMatrix *fineStiffness)
{
  _fineStiffnessMatrix = fineStiffness;
  computeCoarseStiffnessMatrix(fineStiffness);
  setUpSmoother(fineStiffness);
  
  if (_coarseOperator != Teuchos::null)
  {
    Epetra_Time coarseOperatorTimer(Comm());
    // then our changed coarse matrix is coarse operator's fine matrix, and we need to ask coarse operator recompute *its* coarse matrix and smoother
    _coarseOperator->setFineStiffnessMatrix(getCoarseStiffnessMatrix().get());
    _timeUpdateCoarseOperator += coarseOperatorTimer.ElapsedTime();
  }
}

void GMGOperator::setFillRatio(double fillRatio)
{
  _fillRatio = fillRatio;
//  cout << "fill ratio set to " << fillRatio << endl;
}

void GMGOperator::setFunctionExporter(Teuchos::RCP<HDF5Exporter> exporter, FunctionPtr function, string functionName)
{
  _functionExporter = exporter;
  _functionToExport = function;
  _functionToExportName = functionName;
  _exportTimeStep = 0;
}

void GMGOperator::setIsFinest(bool value)
{
  _isFinest = value;
}

void GMGOperator::setLevelOfFill(int fillLevel)
{
  _levelOfFill = fillLevel;
//  cout << "level of fill set to " << fillLevel << endl;
}

void GMGOperator::setMultigridStrategy(MultigridStrategy choice)
{
  _multigridStrategy = choice;
}

void GMGOperator::setSchwarzFactorizationType(FactorType choice)
{
  _schwarzBlockFactorizationType = choice;
}

void GMGOperator::setSmootherApplicationCount(int count)
{
  _smootherApplicationCount = count;
}

void GMGOperator::setSmootherApplicationType(SmootherApplicationType applicationType)
{
  // smoother application type is deprecated
  // TODO: eliminate it (in favor of MultigridStrategy)
  _smootherApplicationType = applicationType;
  if (_smootherApplicationType == ADDITIVE)
    _multigridStrategy = TWO_LEVEL;
  else
    _multigridStrategy = V_CYCLE;
}

void GMGOperator::setSmootherOverlap(int overlap)
{
  _smootherOverlap = overlap;
}

void GMGOperator::setSmootherType(GMGOperator::SmootherChoice smootherType)
{
  _smootherType = smootherType;
}

double GMGOperator::getSmootherWeight()
{
  return _smootherWeight;
}

void GMGOperator::setSmootherWeight(double weight)
{
  _smootherWeight = weight;
  if (_smootherWeight != -1.0)
  {
    // prevent overwriting this value later
    cout << "Turning OFF _useSchwarzScalingWeight, because smootherWeight value is provided.\n";
    setUseSchwarzScalingWeight(false);
  }
}

void GMGOperator::setUpSmoother(Epetra_CrsMatrix *fineStiffnessMatrix)
{
  narrate("setUpSmoother()");
  Epetra_Time smootherSetupTimer(Comm());
  
  SmootherChoice choice = _smootherType;

  Teuchos::ParameterList List;

  Teuchos::RCP<Ifpack_Preconditioner> smoother;

  switch (choice)
  {
  case NONE:
    _smoother = Teuchos::null;
    return;
  case POINT_JACOBI:
  {
    List.set("relaxation: type", "Jacobi");
    smoother = Teuchos::rcp(new Ifpack_PointRelaxation(fineStiffnessMatrix) );
  }
  break;
  case POINT_SYMMETRIC_GAUSS_SEIDEL:
  {
    List.set("relaxation: type", "symmetric Gauss-Seidel");
    smoother = Teuchos::rcp(new Ifpack_PointRelaxation(fineStiffnessMatrix) );
  }
  break;
  case BLOCK_JACOBI:
  {
    // TODO: work out how we're supposed to specify partitioning scheme.
    //      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Block Gauss-Seidel smoother not yet supported.");

    smoother = Teuchos::rcp(new Ifpack_BlockRelaxation<Ifpack_SparseContainer<Ifpack_Amesos> >(fineStiffnessMatrix) );
    Teuchos::ParameterList List;
    // TODO: work out what the various parameters do, and how they should depend on the problem...
    int overlapBlocks = _smootherOverlap;
    int sweeps = 2;
    int localParts = 4;

    List.set("relaxation: type", "Jacobi");
    List.set("relaxation: sweeps", sweeps);
    List.set("amesos: solver type", "Amesos_Klu");

    List.set("partitioner: overlap", overlapBlocks);
#ifdef HAVE_IFPACK_METIS
    // use METIS to create the blocks. This requires --enable-ifpack-metis.
    // If METIS is not installed, the user may select "linear".
    List.set("partitioner: type", "metis");
#else
    // or a simple greedy algorithm is METIS is not enabled
    List.set("partitioner: type", "greedy");
#endif
    // defines here the number of local blocks. If 1,
    // and only one process is used in the computation, then
    // the preconditioner must converge in one iteration.
    List.set("partitioner: local parts", localParts);
  }
  break;
  case BLOCK_SYMMETRIC_GAUSS_SEIDEL:
  {
    // TODO: work out how we're supposed to specify partitioning scheme.
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Block Gauss-Seidel smoother not yet supported.");

    smoother = Teuchos::rcp(new Ifpack_BlockRelaxation<Ifpack_SparseContainer<Ifpack_Amesos> >(fineStiffnessMatrix) );
    Teuchos::ParameterList List;
    // TODO: work out what the various parameters do, and how they should depend on the problem...
    int overlapBlocks = _smootherOverlap;
    int sweeps = 2;
    int localParts = 4;

    List.set("relaxation: type", "symmetric Gauss-Seidel");
    List.set("relaxation: sweeps", sweeps);
    List.set("amesos: solver type", "Amesos_Klu"); // default

    List.set("partitioner: overlap", overlapBlocks);
#ifdef HAVE_IFPACK_METIS
    // use METIS to create the blocks. This requires --enable-ifpack-metis.
    // If METIS is not installed, the user may select "linear".
    List.set("partitioner: type", "metis");
#else
    // or a simple greedy algorithm if METIS is not enabled
    List.set("partitioner: type", "greedy");
#endif
    // defines here the number of local blocks. If 1,
    // and only one process is used in the computation, then
    // the preconditioner must converge in one iteration.
    List.set("partitioner: local parts", localParts);
  }
  break;
  case IFPACK_ADDITIVE_SCHWARZ:
  case CAMELLIA_ADDITIVE_SCHWARZ:
  {
//      cout << "Using additive Schwarz smoother.\n";
    int OverlapLevel = _smootherOverlap;

//    List.set("amesos: solver type", "Amesos_Lapack");
    
    if (choice==IFPACK_ADDITIVE_SCHWARZ)
    {
      switch (_schwarzBlockFactorizationType)
      {
        case Direct:
        {
          Ifpack_AdditiveSchwarz<Ifpack_Amesos>* ifpackSmoother = new Ifpack_AdditiveSchwarz<Ifpack_Amesos>(fineStiffnessMatrix, OverlapLevel);
          smoother = Teuchos::rcp( ifpackSmoother );
        }
        break;
      case ILU:
        {
          Ifpack_AdditiveSchwarz<Ifpack_ILU>* ifpackSmoother = new Ifpack_AdditiveSchwarz<Ifpack_ILU>(fineStiffnessMatrix, OverlapLevel);
          smoother = Teuchos::rcp( ifpackSmoother );
          List.set("fact: level-of-fill", _levelOfFill);
        }
        break;
      case IC:
        {
          Ifpack_AdditiveSchwarz<Ifpack_IC>* ifpackSmoother = new Ifpack_AdditiveSchwarz<Ifpack_IC>(fineStiffnessMatrix, OverlapLevel);
          smoother = Teuchos::rcp( ifpackSmoother );
          List.set("fact: ict level-of-fill", _fillRatio);
        }
        break;
      default:
        break;
      }
    }
    else
    {
      switch (_schwarzBlockFactorizationType)
      {
      case Direct:
        {
          Camellia::AdditiveSchwarz<Ifpack_Amesos>* camelliaSmoother = new Camellia::AdditiveSchwarz<Ifpack_Amesos>(fineStiffnessMatrix, OverlapLevel, _fineMesh, _fineDofInterpreter, _hierarchicalNeighborsForSchwarz, _dimensionForSchwarzNeighborRelationship);
          smoother = Teuchos::rcp( camelliaSmoother );
        }
        break;
      case ILU:
        {
          Camellia::AdditiveSchwarz<Ifpack_ILU>* camelliaSmoother = new Camellia::AdditiveSchwarz<Ifpack_ILU>(fineStiffnessMatrix, OverlapLevel, _fineMesh, _fineDofInterpreter, _hierarchicalNeighborsForSchwarz, _dimensionForSchwarzNeighborRelationship);
          List.set("fact: level-of-fill", _levelOfFill);
          smoother = Teuchos::rcp( camelliaSmoother );
        }
        break;
      case IC:
        {
          Camellia::AdditiveSchwarz<Ifpack_IC>* camelliaSmoother = new Camellia::AdditiveSchwarz<Ifpack_IC>(fineStiffnessMatrix, OverlapLevel, _fineMesh, _fineDofInterpreter, _hierarchicalNeighborsForSchwarz, _dimensionForSchwarzNeighborRelationship);
          List.set("fact: ict level-of-fill", _fillRatio);
          smoother = Teuchos::rcp( camelliaSmoother );
        }
        break;
      }
    }

    List.set("schwarz: combine mode", "Add"); // The PDF doc says to use "Insert" to maintain symmetry, but the HTML docs (which are more recent) say to use "Add".  http://trilinos.org/docs/r11.10/packages/ifpack/doc/html/index.html
  }
  break;

  default:
    break;
  }

  int err = smoother->SetParameters(List);
  if (err != 0)
  {
    cout << "WARNING: In GMGOperator, smoother->SetParameters() returned with err " << err << endl;
  }

//  int rank = Teuchos::GlobalMPISession::getRank();
//  if (rank == 0) {
//    cout << "Smoother info:\n";
//    cout << *smoother;
//  }

//  if (_smootherType != IFPACK_ADDITIVE_SCHWARZ) {
  // not real sure why, but in the doc examples, this isn't called for additive schwarz
  err = smoother->Initialize();
  if (err != 0)
  {
    cout << "WARNING: In GMGOperator, smoother->Initialize() returned with err " << err << endl;
  }
//  }
//  cout << "Calling smoother->Compute()\n";
  err = smoother->Compute();
//  cout << "smoother->Compute() completed\n";

  int myLevel = getOperatorLevel();
  if (err != 0)
  {
    cout << "WARNING: In GMGOperator (level " << myLevel << "), smoother->Compute() returned with err = " << err << endl;
  }

  if ((choice == CAMELLIA_ADDITIVE_SCHWARZ) && _useSchwarzDiagonalWeight)
  {
    // setup weight vector:
    const Epetra_Map* rangeMap = NULL;
    switch (_schwarzBlockFactorizationType)
    {
      case Direct:
      {
        Camellia::AdditiveSchwarz<Ifpack_Amesos>* camelliaSmoother = dynamic_cast<Camellia::AdditiveSchwarz<Ifpack_Amesos>*>(smoother.get());
        rangeMap = &camelliaSmoother->OverlapMap();
      }
        break;
      case ILU:
      {
        Camellia::AdditiveSchwarz<Ifpack_ILU>* camelliaSmoother = dynamic_cast<Camellia::AdditiveSchwarz<Ifpack_ILU>*>(smoother.get());
        rangeMap = &camelliaSmoother->OverlapMap();
      }
        break;
      case IC:
      {
        Camellia::AdditiveSchwarz<Ifpack_IC>* camelliaSmoother = dynamic_cast<Camellia::AdditiveSchwarz<Ifpack_IC>*>(smoother.get());
        rangeMap = &camelliaSmoother->OverlapMap();
      }
        break;
    }
    
    Epetra_FEVector multiplicities(fineStiffnessMatrix->RowMap(),1);
    vector<GlobalIndexTypeToCast> overlappingEntries(rangeMap->NumMyElements());
    rangeMap->MyGlobalElements(&overlappingEntries[0]);
    vector<double> myOverlappingValues(overlappingEntries.size(),1.0);
    multiplicities.SumIntoGlobalValues(myOverlappingValues.size(), &overlappingEntries[0], &myOverlappingValues[0]);
    multiplicities.GlobalAssemble();
    
    double maxMultiplicity = 0;
    multiplicities.MaxValue(&maxMultiplicity);
    if (_useSchwarzScalingWeight)
    {
      _smootherWeight = 1.0 / maxMultiplicity;
    }
    
    if (_useSchwarzDiagonalWeight)
    {
      _smootherDiagonalWeight = Teuchos::rcp(new Epetra_MultiVector(fineStiffnessMatrix->RowMap(), 1) );
      GlobalIndexTypeToCast numMyElements = fineStiffnessMatrix->RowMap().NumMyElements();
      for (int LID=0; LID < numMyElements; LID++)
      {
        double value = multiplicities[0][LID];
        TEUCHOS_TEST_FOR_EXCEPTION(value == 0.0, std::invalid_argument, "internal error: value should never be 0");
        (*_smootherDiagonalWeight)[0][LID] = 1.0/value;
      }
    }
    // debugging:
//    printMapSummary(*rangeMap, "Schwarz matrix range map");
//    cout << *_smootherWeight_sqrt;
  }
  
  if (_useSchwarzScalingWeight && ((_smootherType == CAMELLIA_ADDITIVE_SCHWARZ) || (_smootherType == IFPACK_ADDITIVE_SCHWARZ)))
  {
    // (For IfPack, this weight may not be exactly correct, but it's probably kinda close.  Likely we will deprecate support for
    // IFPACK_ADDITIVE_SCHWARZ soon.)
    
//    double oldWeight = _smootherWeight;
    
    _smootherWeight = computeSchwarzSmootherWeight();
    
//    int rank = Teuchos::GlobalMPISession::getRank();
//    static bool haveWarned = false;
//    if (!haveWarned && (rank==0))
//    {
//      cout << "NOTE: using new approach to Schwarz scaling weight, based on Nate's conjecture regarding the spectral radius of the subdomain connectivity matrix and some results in Smith et al.";
//      cout << " First _smootherWeight value: " << _smootherWeight << " (previous weight was " << oldWeight << ").\n";
//      haveWarned = true;
//    }
    // DEBUGGING
//    cout << "_smootherWeight value: " << _smootherWeight << " (previous weight was " << oldWeight << ").\n";
  }
  
  _smoother = smoother;
  _timeSetUpSmoother = smootherSetupTimer.ElapsedTime();
}

void GMGOperator::setUseSchwarzDiagonalWeight(bool value)
{
  _useSchwarzDiagonalWeight = value;
}

void GMGOperator::setUseSchwarzScalingWeight(bool value)
{
  _useSchwarzScalingWeight = value;
}

std::string GMGOperator::smootherString(SmootherChoice choice)
{
  switch(choice)
  {
    case IFPACK_ADDITIVE_SCHWARZ:
      return "Ifpack additive Schwarz";
    case CAMELLIA_ADDITIVE_SCHWARZ:
      return "Camellia additive Schwarz";
    case NONE:
      return "None";
    case BLOCK_JACOBI:
      return "Block Jacobi";
    case BLOCK_SYMMETRIC_GAUSS_SEIDEL:
      return "Block Symmetric Gauss-Seidel";
    case POINT_JACOBI:
      return "Point Jacobi";
    case POINT_SYMMETRIC_GAUSS_SEIDEL:
      return "Point Symmetric Gauss-Seidel";
  }
}

Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::getMatrixRepresentation()
{
  return Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*this, _finePartitionMap);
}

void GMGOperator::setUseHierarchicalNeighborsForSchwarz(bool value)
{
  _hierarchicalNeighborsForSchwarz = value;
}

Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::getProlongationOperator()
{
  return _P;
}

Teuchos::RCP<Epetra_Operator> GMGOperator::getSmoother() const
{
  return _smoother;
}

Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::getSmootherAsMatrix()
{
  narrate("getSmootherAsMatrix()");
  return Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*_smoother, _finePartitionMap);
}

//! Returns the coarse stiffness matrix (an Epetra_CrsMatrix).
Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::getCoarseStiffnessMatrix()
{
  return _coarseSolution->getStiffnessMatrix();
}