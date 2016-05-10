//
//  GDAMinimumRuleTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 5/14/15.
//
//

#include "Epetra_Import.h"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include "Epetra_MpiDistributor.h"
#endif

#include "Epetra_SerialComm.h"
#include "Epetra_SerialDistributor.h"

#include "BasisFactory.h"
#include "CamelliaCellTools.h"
#include "CamelliaTestingHelpers.h"
#include "CubatureFactory.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MeshTestUtility.h"
#include "MPIWrapper.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "SpaceTimeHeatDivFormulation.h"
#include "TypeDefs.h"
#include "VarFactory.h"

using namespace Camellia;
using namespace Intrepid;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
  /*
   Three essential tests that GDAMinimumRule is doing its job:
   1. Cell and subcell constraints are correctly identified.
   2. Take a coarse (constraining) basis and compute some values on subcell.  Compute
   fine (constrained) basis values along the same physical region, now weighting
   according to getBasisMap().  Are the values the same?
   3. Actually try solving something.
   
   Of these, #1 would be good for a few hand-computed instances -- I'm unclear on how
   easily we can check this on a generic mesh topology.  (Sanity checks are certainly
   possible--check that the constraining subcell is in fact an ancestor of the constrained,
   for instance.  This is implemented by testSubcellConstraintIsAncestor, below.)
   #1 is also cheap to run.
   
   #2 is a bit more expensive, but not crazy.  It should be possible to repurpose some of
   the ideas already present in BasisReconciliationTests for this purpose.  It also seems
   very possible to do this for a generic mesh.
   
   #3 is what the DPGTests have done to date; the HangingNodePoisson3D_Slow test below,
   moved over from DPGTests, is an example.  This is an "integration" test, and it's
   expensive and when it fails it doesn't reveal much in terms of where the failure came
   from.  However, it does have the advantage of checking that we end up with the right
   answers in the context of a real problem!
   
   TODO: implement #2.  Implement #3 for a generic mesh (can imitate HangingNodePoisson3D_Slow, perhaps).
   */
  
  // copied from BasisReconciliation.cpp -- likely, this method should be in a utility class somewhere
  void filterFCValues(FieldContainer<double> &filteredFC, const FieldContainer<double> &fc,
                      set<int> &ordinals, int basisCardinality)
  {
    // we use pointer arithmetic here, which doesn't allow Intrepid's safety checks, for two reasons:
    // 1. It's easier to manage in a way that's independent of the rank of the basis
    // 2. It's faster.
    int numEntriesPerBasisField = fc.size() / basisCardinality;
    int filteredFCIndex = 0;
    // we can do a check of our own, though, that the filteredFC is the right total length:
    TEUCHOS_TEST_FOR_EXCEPTION(filteredFC.size() != numEntriesPerBasisField * ordinals.size(), std::invalid_argument,
                               "filteredFC.size() != numEntriesPerBasisField * ordinals.size()");
    for (auto ordinal : ordinals)
    {
      const double *fcEntry = &fc[ordinal * numEntriesPerBasisField];
      double *filteredEntry = &filteredFC[ filteredFCIndex * numEntriesPerBasisField ];
      for (int i=0; i<numEntriesPerBasisField; i++)
      {
        *filteredEntry = *fcEntry;
        filteredEntry++;
        fcEntry++;
      }
      
      filteredFCIndex++;
    }
  }
  
  MeshPtr minimalPoissonHangingNodeMesh(int H1Order, int delta_k, int spaceDim, PoissonFormulation::PoissonFormulationChoice formChoice)
  {
    // exact solution: for now, we just use a linear phi
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    
    FunctionPtr phi_exact = -x + y;
    
    bool useConformingTraces = true; // doesn't matter for continuous Galerkin
    PoissonFormulation poissonForm(spaceDim, useConformingTraces, formChoice);
    
    vector<double> dimensions(spaceDim,1.0);
    vector<int> elementCounts(spaceDim,1);
    elementCounts[0] = 2; // so that we can have a hanging node, make the mesh be two wide in the x direction
    MeshPtr mesh = MeshFactory::rectilinearMesh(poissonForm.bf(), dimensions, elementCounts, H1Order, delta_k);
    
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    return mesh;
  }
  
  void testContiguousGlobalDofNumbering(GDAMinimumRule* gdaMinRule, Teuchos::FancyOStream &out, bool &success)
  {
    // check that all dofs are assigned, basically
    set<GlobalIndexType> myGlobalDofIndices = gdaMinRule->globalDofIndicesForPartition(-1); // -1 means this rank
    // while some of the local cells will see dofs that don't belong to the partition, they should see every dof
    // that *does* belong to the partition -- by iterating through local cells and asking for their global dofs,
    // we should find all members of myGlobalDofIndices
    set<GlobalIndexType> foundGlobalIndices;
    const set<GlobalIndexType>* myCellIDs = &gdaMinRule->cellsInPartition(-1);
    for (GlobalIndexType cellID : *myCellIDs)
    {
      set<GlobalIndexType> globalIndicesForCell = gdaMinRule->globalDofIndicesForCell(cellID);
      foundGlobalIndices.insert(globalIndicesForCell.begin(),globalIndicesForCell.end());
    }
    // rule is, foundGlobalIndices should be a superset of myGlobalDofIndices
    set<GlobalIndexType> missingDofIndices;
    for (GlobalIndexType myDofIndex : myGlobalDofIndices)
    {
      if (foundGlobalIndices.find(myDofIndex) == foundGlobalIndices.end())
      {
        missingDofIndices.insert(myDofIndex);
      }
    }
    if (missingDofIndices.size() != 0)
    {
      success = false;
      out << "Missing global dof ordinals: ";
      for (GlobalIndexType missingDofIndex : missingDofIndices)
      {
        out << missingDofIndex << " ";
      }
      out << endl;
    }
  }
  
  void testContiguousGlobalDofNumberingComplexSpaceTimeMesh(bool useConformingTraces, Teuchos::FancyOStream &out, bool &success)
  {
    int spaceDim = 2;
    double pi = atan(1)*4;
    
    double t0 = 0;
    double t1 = pi;
    int temporalDivisions = 1;
    
    vector<double> x0 = {0.0, 0.0};;
    vector<double> dims = {2*pi, 2*pi};
    vector<int> numElements = {2,2};
    
    MeshTopologyPtr spatial2DMeshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    MeshTopologyPtr meshTopo = MeshFactory::spaceTimeMeshTopology(spatial2DMeshTopo, t0, t1, temporalDivisions);
    
    // some refinements in an effort to replicate an issue...
    // 1. Uniform refinement
    IndexType nextElement = meshTopo->cellCount();
    vector<IndexType> cellsToRefine = meshTopo->getActiveCellIndicesGlobal();
    CellTopoPtr cellTopo = meshTopo->getCell(0)->topology();
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
    for (IndexType cellIndex : cellsToRefine)
    {
      meshTopo->refineCell(cellIndex, refPattern, nextElement);
      nextElement += refPattern->numChildren();
    }
    // 2. Selective refinement
    cellsToRefine = {4,15,21,30};
    for (IndexType cellIndex : cellsToRefine)
    {
      meshTopo->refineCell(cellIndex, refPattern, nextElement);
      nextElement += refPattern->numChildren();
    }
    
    int fieldPolyOrder = 1;
    double epsilon = 1.0;
    SpaceTimeHeatDivFormulation form(spaceDim, epsilon);
    form.initializeSolution(meshTopo, fieldPolyOrder);
    
    MeshPtr formMesh = form.solution()->mesh();
    
    GDAMinimumRule* gdaMinRule = dynamic_cast<GDAMinimumRule*>(formMesh->globalDofAssignment().get());
    testContiguousGlobalDofNumbering(gdaMinRule, out, success);
  }
  
  void testSolvePoissonHangingNode(int spaceDim, PoissonFormulation::PoissonFormulationChoice formChoice,
                                   Teuchos::FancyOStream &out, bool &success)
  {
    // exact solution: for now, we just use a linear phi
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    
    FunctionPtr phi_exact = -x + y;
    
    int H1Order = 2;
    bool useConformingTraces = true; // doesn't matter for continuous Galerkin
    int delta_k = -1;
    PoissonFormulation poissonForm(spaceDim, useConformingTraces, formChoice);
    
    IPPtr ip; // null for Bubnov-Galerkin
    switch (formChoice) {
      case Camellia::PoissonFormulation::PRIMAL:
      {
        VarPtr q = poissonForm.q();
        ip = IP::ip();
        ip->addTerm(q->grad());
        ip->addTerm(q);
        delta_k = spaceDim;
      }
        break;
      case Camellia::PoissonFormulation::ULTRAWEAK:
        ip = poissonForm.bf()->graphNorm();
        delta_k = spaceDim;
        break;
      case Camellia::PoissonFormulation::CONTINUOUS_GALERKIN:
        delta_k = 0; // Bubnov-Galerkin
        break;
      default:
        out << "Unsupported formulation choice!\n";
        success = false;
        return;
        break;
    }
    
    MeshPtr mesh = minimalPoissonHangingNodeMesh(H1Order, delta_k, spaceDim, formChoice);
    
    // rhs = f * v, where f = \Delta phi
    RHSPtr rhs = RHS::rhs();
    FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz();
    rhs->addTerm(f * poissonForm.q());
    
    if (! MeshTestUtility::checkLocalGlobalConsistency(mesh) )
    {
      cout << "FAILURE: 1-irregular Poisson 2D mesh fails local-to-global consistency check.\n";
      success = false;
    }

//    {
//      //DEBUGGING
//      GDAMinimumRule * minRule = dynamic_cast<GDAMinimumRule *>(mesh->globalDofAssignment().get());
//      minRule->printGlobalDofInfo();
//      GlobalIndexType cellWithHangingNode = 4;
//      CellConstraints cellConstraints = minRule->getCellConstraints(cellWithHangingNode);
//      minRule->getDofMapper(cellWithHangingNode, cellConstraints)->printMappingReport();
//    }
    
    VarPtr phi = poissonForm.phi();
    BCPtr bc = BC::bc();
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    bc->addDirichlet(phi, boundary, phi_exact);
    
    SolutionPtr soln = Solution::solution(poissonForm.bf(), mesh, bc, rhs, ip);
    
    map<int, FunctionPtr> phi_exact_map;
    phi_exact_map[phi->ID()] = phi_exact;
    soln->projectOntoMesh(phi_exact_map);
    
    FunctionPtr phi_soln = Function::solution(phi, soln);
    FunctionPtr phi_err = phi_soln - phi_exact;
    
    double tol = 1e-12;
    double phi_err_l2 = phi_err->l2norm(mesh);
    
    if (phi_err_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for 1-irregular ";
      cout << spaceDim << "D mesh and exactly recoverable solution, phi error after projection is " << phi_err_l2 << endl;
      
      string outputSuperDir = ".";
      string outputDir = "poisson2DHangingNodeProjection";
      HDF5Exporter exporter(mesh, outputDir, outputSuperDir);
      cout << "Writing phi err to " << outputSuperDir << "/" << outputDir << endl;
      
      exporter.exportFunction(phi_err, "phi_err");
    }
    
    soln->clear();
    soln->solve();
    
    Epetra_MultiVector *lhsVector = soln->getGlobalCoefficients();
    Epetra_SerialComm Comm;
    Epetra_Map partMap = soln->getPartitionMap();
    
    // Import solution onto current processor
    GlobalIndexTypeToCast numNodesGlobal = mesh->numGlobalDofs();
    GlobalIndexTypeToCast numMyNodes = numNodesGlobal;
    Epetra_Map     solnMap(numNodesGlobal, numMyNodes, 0, Comm);
    Epetra_Import  solnImporter(solnMap, partMap);
    Epetra_Vector  solnCoeff(solnMap);
    solnCoeff.Import(*lhsVector, solnImporter, Insert);
    
// TODO: work out whether MeshTestUtility::neighborBasesAgreeOnSides() is valid on CG meshes
    if ( ! MeshTestUtility::neighborBasesAgreeOnSides(mesh, solnCoeff))
    {
      cout << "GDAMinimumRuleTests failure: for 1-irregular 2D Poisson mesh with hanging nodes (after solving), neighboring bases do not agree on sides." << endl;
      success = false;
    }

//    cout << "...solution continuities checked.\n";
    
    phi_err_l2 = phi_err->l2norm(mesh);
    if (phi_err_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for 1-irregular ";
      cout << spaceDim << "D mesh and exactly recoverable solution, phi error is " << phi_err_l2 << endl;
      
      string outputSuperDir = ".";
      string outputDir = "poisson2DHangingNodeError";
      HDF5Exporter exporter(mesh, outputDir, outputSuperDir);
      cout << "Writing phi err to " << outputSuperDir << "/" << outputDir << endl;
      
      exporter.exportFunction(phi_err, "phi_err");
      
      outputDir = "poisson2DHangingNodeSolution";
      cout << "Writing solution to " << outputSuperDir << "/" << outputDir << endl;
      HDF5Exporter solutionExporter(mesh, outputDir, outputSuperDir);
      solutionExporter.exportSolution(soln);
    }
    
//    HDF5Exporter exporter(mesh, "poisson2DHangingNodeSolution", "/tmp");
//    exporter.exportSolution(soln);
  }
  
  void testSolvePoissonContinuousGalerkinHangingNode(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    testSolvePoissonHangingNode(spaceDim, PoissonFormulation::CONTINUOUS_GALERKIN, out, success);
  }
 
  void testSolvePoissonPrimalHangingNode(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    testSolvePoissonHangingNode(spaceDim, PoissonFormulation::PRIMAL, out, success);
  }
  
  void testSolvePoissonUltraweakHangingNode(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    testSolvePoissonHangingNode(spaceDim, PoissonFormulation::ULTRAWEAK, out, success);
  }
  
  void testSubcellConstraintIsAncestor(MeshPtr mesh, Teuchos::FancyOStream &out, bool &success)
  {
    /*
     This test iterates through all the active cells in the mesh, and checks that for each of their
     subcells, the subcell that constrains them is an ancestor of that subcell.
     
     One other thing we could test, but don't yet, is that the constraining subcells are not themselves
     constrained by some other subcell--the rule is they should be the end of the line.
     */
    GDAMinimumRule* gda = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
    if (gda == NULL)
    {
      out << "Mesh appears not to use GDAMinimumRule.  testSubcellConstraintIsAncestor() requires this.\n";
      success = false;
      return;
    }
    auto activeCellIDs = mesh->cellIDsInPartition();
    int minSubcellDim = mesh->globalDofAssignment()->minimumSubcellDimensionForContinuityEnforcement();
    MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
    int sideDim = meshTopo->getDimension() - 1;
    for (auto cellID : activeCellIDs)
    {
      CellPtr cell = meshTopo->getCell(cellID);
      CellConstraints cellConstraints = gda->getCellConstraints(cellID);
      CellTopoPtr cellTopo = cell->topology();
      for (int subcdim=minSubcellDim; subcdim<cellTopo->getDimension(); subcdim++)
      {
        int subcellCount = cellTopo->getSubcellCount(subcdim);
        for (int subcord=0; subcord<subcellCount; subcord++)
        {
          IndexType subcellEntityIndex = cell->entityIndex(subcdim, subcord);
          AnnotatedEntity constrainingEntity = cellConstraints.subcellConstraints[subcdim][subcord];
          CellPtr constrainingCell = meshTopo->getCell(constrainingEntity.cellID);
          CellTopoPtr constrainingCellTopo = constrainingCell->topology();
          // When a side is involved in the constraint, then AnnotatedEntity.subcellOrdinal is the subcell ordinal in the side.
          // We therefore map to the subcell in the cell:
          unsigned constrainingCellSubcord = CamelliaCellTools::subcellOrdinalMap(constrainingCellTopo, sideDim, constrainingEntity.sideOrdinal, constrainingEntity.dimension, constrainingEntity.subcellOrdinal);
          IndexType constrainingEntityIndex = constrainingCell->entityIndex(constrainingEntity.dimension, constrainingCellSubcord);
          bool isAncestor = meshTopo->entityIsGeneralizedAncestor(constrainingEntity.dimension, constrainingEntityIndex,
                                                                  subcdim, subcellEntityIndex);
          if (!isAncestor)
          {
            out << "cell " << cellID << ", " << CamelliaCellTools::entityTypeString(subcdim);
            out << " ordinal " << subcord << " is constrained by cell " << constrainingEntity.cellID;
            out << ", " << CamelliaCellTools::entityTypeString(constrainingEntity.dimension);
            out << " ordinal " << constrainingCellSubcord << ", which is not its ancestor!\n";
            success = false;
          }
        }
      }
    }
  }
  
  void testCoarseBasisEqualsWeightedFineBasis(MeshPtr mesh, Teuchos::FancyOStream &out, bool &success)
  {
    /*
     For each active cell:
     1. Construct the BasisCache for that cell.
     2. For each side of the cell:
     a. Compute the BasisMap for that side.
     b. For each subcell of that side:
     i.    Compute cubature points on the subcell.
     ii.   Map the cubature points to the side.
     iii.  Set the reference cell points of the side BasisCache to be the mapped cubature points.
     iv.   Compute transformed basis values on that side.
     v.    Weight those basis values according to the BasisMap.
     vi.   Determine the constraining subcell and side.
     vii.  Map the fine subcell cubature points to the constraining subcell's reference space.
     viii. Map the constraining subcell points to the constraining side's reference space.
     ix.   Create a side BasisCache for the constraining cell/side.
     x.    Set the reference points for the side BasisCache to be those computed.
     xi.   Check that the physical points for the constraining side cache agree with the fine side cache.
     xii.  Compute the constraining basis at those points.
     xiii. Compare the values.
     
     Notes about BasisMap: this is a typedef:
     typedef vector< SubBasisDofMapperPtr > BasisMap;
     
     Where SubBasisDofMapper allows computation of a weighted sum of basis values via its mapFineData() method.
     Iterating through the BasisMap will allow such maps for the basis as a whole.
     
     One does need to attend to the mappedGlobalDofOrdinals() in the SubBasisDofMapper; this tells you which global
     ordinals's data you get from mapFineData().  If a global dof ordinal is mapped by several SubBasisDofMappers,
     one should accumulate the values.
     
     So to be precise, one will want to construct BasisMaps for both the coarse and the fine domains, and check that
     they agree on the global data once the mapFineData() thing has been done.  (You still call mapFineData for the coarse
     domain; it just happens that this is a 1-1 mapping because the coarse domain constrains itself.)
     
     */
    VarFactoryPtr vf = mesh->bilinearForm()->varFactory();
    
    // for now, we just check a single var, preferring traces:
    VarPtr var;
    if (vf->traceVars().size() > 0)
    {
      var = vf->traceVars()[0];
    }
    else if (vf->fluxVars().size() > 0)
    {
      var = vf->fluxVars()[0];
    }
    else
    {
      var = vf->fieldVars()[0];
    }
    // TODO: test flux var when both traces and fluxes are present (as with Poisson for spaceDim > 1)
    
    GDAMinimumRule* gda = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
    if (gda == NULL)
    {
      out << "Mesh appears not to use GDAMinimumRule.  testSubcellConstraintIsAncestor() requires this.\n";
      success = false;
      return;
    }
    
    auto myCellIDs = mesh->cellIDsInPartition();
    MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());

//    { // DEBUGGING
//      gda->printGlobalDofInfo();
//      meshTopo->printAllEntities();
//      
//      GlobalIndexType cellID = 5;
//      CellConstraints constraints = gda->getCellConstraints(cellID);
//      cout << "Mapping report for cell " << cellID << endl;
//      gda->getDofMapper(cellID, constraints)->printMappingReport();
//    }
//    unsigned edgeDim = 1;
//    meshTopo->printConstraintReport(edgeDim);
    
    typedef vector< SubBasisDofMapperPtr > BasisMap;
    
    int sideDim = meshTopo->getDimension() - 1;
    Camellia::CubatureFactory cubFactory;
    for (auto cellID : myCellIDs)
    {
      BasisCachePtr cellBasisCache = BasisCache::basisCacheForCell(mesh, cellID);
      CellPtr cell = meshTopo->getCell(cellID);
      auto dofOwnershipInfo = gda->getGlobalDofIndices(cellID);
      CellTopoPtr cellTopo = cell->topology();
      CellConstraints cellConstraints = gda->getCellConstraints(cellID);
      
      DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
      
      BasisMap fineBasisMap;
      BasisPtr fineBasis;
      CellTopoPtr domainTopo;
      int domainOrdinalInCell;
      
      bool isVolumeVar = (var->varType() == FIELD);
      
      if (isVolumeVar)
      {
        fineBasisMap = gda->getBasisMap(cellID, dofOwnershipInfo, var);
        fineBasis = trialOrder->getBasis(var->ID());
      }
      
      for (int sideOrdinal : trialOrder->getSidesForVarID(var->ID()))
      {
        BasisCachePtr domainBasisCache;
        if (isVolumeVar)
        {
          domainBasisCache = cellBasisCache;
          domainOrdinalInCell = 0;
        }
        else
        {
          domainBasisCache = cellBasisCache->getSideBasisCache(sideOrdinal);
          fineBasis = trialOrder->getBasis(var->ID(),sideOrdinal);
          fineBasisMap = gda->getBasisMap(cellID, dofOwnershipInfo, var, sideOrdinal);
          domainOrdinalInCell = sideOrdinal;
        }
        
        int cubDegree = fineBasis->getDegree();
        domainTopo = fineBasis->domainTopology();
        int domainDim = domainTopo->getDimension();
        int minSubcellDimension = BasisReconciliation::minimumSubcellDimension(fineBasis);
        for (int subcdim=minSubcellDimension; subcdim<=sideDim; subcdim++)
        {
          int subcellCount = domainTopo->getSubcellCount(subcdim);
          for (int subcord=0; subcord<subcellCount; subcord++)
          {
            CellTopoPtr subcell = domainTopo->getSubcell(subcdim, subcord);
            unsigned subcordInCell = CamelliaCellTools::subcellOrdinalMap(cellTopo, domainDim, domainOrdinalInCell, subcdim, subcord);
            FieldContainer<double> subcellPoints;
            if (subcdim==0)
            {
              subcellPoints.resize(1,1); // vertex; don't need points as such
            }
            else
            {
              CellTopoPtr subcellTopo = domainTopo->getSubcell(subcdim, subcord);
              auto cubature = cubFactory.create(subcellTopo, cubDegree);
              subcellPoints.resize(cubature->getNumPoints(),cubature->getDimension());
              FieldContainer<double> weights(cubature->getNumPoints()); // we ignore these
              cubature->getCubature(subcellPoints, weights);
            }
            int numPoints = subcellPoints.dimension(0);
            
            // map the subcellPoints to the fine domain
            FieldContainer<double> fineDomainPoints(numPoints,domainDim);
            CamelliaCellTools::mapToReferenceSubcell(fineDomainPoints, subcellPoints, subcdim, subcord, domainTopo);
            
            domainBasisCache->setRefCellPoints(fineDomainPoints);
            FieldContainer<double> fineValuesAllPoints = *domainBasisCache->getTransformedValues(fineBasis, OP_VALUE);
            // strip cell dimension:
            fineValuesAllPoints.resize(fineBasis->getCardinality(), numPoints);

            RefinementBranch cellRefinementBranch = cell->refinementBranchForSubcell(subcdim, subcordInCell, mesh->getTopology());
            if (cellRefinementBranch.size() == 0)
            {
              RefinementPatternPtr noRefPattern = RefinementPattern::noRefinementPattern(cell->topology());
              cellRefinementBranch = {{noRefPattern.get(), 0}};
            }
            
            AnnotatedEntity constrainingEntityInfo = cellConstraints.subcellConstraints[subcdim][subcordInCell];
            unsigned constrainingSideOrdinal = constrainingEntityInfo.sideOrdinal;
            DofOrderingPtr constrainingTrialOrder = mesh->getElementType(constrainingEntityInfo.cellID)->trialOrderPtr;
            
            CellPtr constrainingCell = meshTopo->getCell(constrainingEntityInfo.cellID);
            CellTopoPtr constrainingCellTopo = constrainingCell->topology();
            
            BasisCachePtr constrainingCellBasisCache = BasisCache::basisCacheForCell(mesh, constrainingEntityInfo.cellID);
            BasisCachePtr constrainingDomainBasisCache;
            BasisPtr constrainingBasis;
            int constrainingDomainOrdinal;
            int constrainingSubcellOrdinalInDomain;
            if (isVolumeVar)
            {
              constrainingBasis = constrainingTrialOrder->getBasis(var->ID());
              constrainingDomainBasisCache = constrainingCellBasisCache;
              constrainingDomainOrdinal = 0;
              constrainingSubcellOrdinalInDomain = CamelliaCellTools::subcellOrdinalMap(constrainingCellTopo, sideDim,
                                                                                        constrainingEntityInfo.sideOrdinal,
                                                                                        constrainingEntityInfo.dimension,
                                                                                        constrainingEntityInfo.subcellOrdinal);
            }
            else
            {
              constrainingBasis = constrainingTrialOrder->getBasis(var->ID(), constrainingSideOrdinal);
              constrainingDomainBasisCache = constrainingCellBasisCache->getSideBasisCache(constrainingSideOrdinal);
              constrainingDomainOrdinal = constrainingSideOrdinal;
              constrainingSubcellOrdinalInDomain = constrainingEntityInfo.subcellOrdinal;
            }
            
            unsigned canonicalToAncestralSubcellPermutation = cell->ancestralPermutationForSubcell(subcdim, subcordInCell,
                                                                                                   mesh->getTopology());
            
            unsigned constrainingSubcellInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCellTopo, domainDim,
                                                                                                  constrainingDomainOrdinal,
                                                                                                  constrainingEntityInfo.dimension,
                                                                                                  constrainingSubcellOrdinalInDomain);
            CellTopoPtr constrainingSubcellTopo = constrainingCellTopo->getSubcell(constrainingEntityInfo.dimension,
                                                                                   constrainingSubcellInConstrainingCell);
            
            // sanity check: confirm that ancestral subcell and constraining subcell refer to the same entity
            pair<unsigned, unsigned> subcellOrdinalAndDimension = cell->ancestralSubcellOrdinalAndDimension(subcdim, subcordInCell, mesh->getTopology());
            CellPtr ancestralCell = cell->ancestralCellForSubcell(subcdim, subcordInCell, mesh->getTopology());
            unsigned ancestralSubcellOrdinal = subcellOrdinalAndDimension.first;
            unsigned ancestralSubcellDimension = subcellOrdinalAndDimension.second;
            TEUCHOS_TEST_FOR_EXCEPTION(ancestralSubcellDimension != constrainingEntityInfo.dimension, std::invalid_argument, "Internal test error: constraining entity has different dimension than ancestral subcell");
            IndexType ancestralSubcellEntityIndex = ancestralCell->entityIndex(ancestralSubcellDimension, ancestralSubcellOrdinal);
            IndexType constrainingSubcellEntityIndex = constrainingCell->entityIndex(constrainingEntityInfo.dimension, constrainingSubcellInConstrainingCell);
            TEUCHOS_TEST_FOR_EXCEPTION(ancestralSubcellEntityIndex != constrainingSubcellEntityIndex, std::invalid_argument, "Internal test error: constraining entity has different entity index in MeshTopology than ancestral subcell");

            unsigned canonicalToConstrainingSubcellPermutation = constrainingCell->subcellPermutation(constrainingEntityInfo.dimension, constrainingSubcellInConstrainingCell);
            unsigned ancestralToCanonicalSubcellPermutation = CamelliaCellTools::permutationInverse(constrainingSubcellTopo, canonicalToAncestralSubcellPermutation);
            unsigned ancestralToConstrainingSubcellPermutation = CamelliaCellTools::permutationComposition(constrainingSubcellTopo, ancestralToCanonicalSubcellPermutation,
                                                                                                           canonicalToConstrainingSubcellPermutation);

            FieldContainer<double> constrainingDomainPoints;
            BasisReconciliation::mapFineSubcellPointsToCoarseDomain(constrainingDomainPoints,
                                                                    subcellPoints,
                                                                    subcdim,
                                                                    subcord,
                                                                    domainDim,
                                                                    domainOrdinalInCell,
                                                                    cellRefinementBranch,
                                                                    constrainingCellTopo,
                                                                    constrainingEntityInfo.dimension,
                                                                    constrainingSubcellOrdinalInDomain,
                                                                    domainDim,
                                                                    constrainingDomainOrdinal,
                                                                    ancestralToConstrainingSubcellPermutation);
            constrainingDomainBasisCache->setRefCellPoints(constrainingDomainPoints);
            
            // As a sanity check, compare the physical points for coarse and fine:
            FieldContainer<double> finePhysicalPoints = domainBasisCache->getPhysicalCubaturePoints();
            FieldContainer<double> coarsePhysicalPoints = constrainingDomainBasisCache->getPhysicalCubaturePoints();
            bool oldSuccess = success;
            success = true;
            TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(finePhysicalPoints, coarsePhysicalPoints, 1e-15);
            if (!success)
            {
              out << "Failure: finePhysicalPoints != coarsePhysicalPoints; therefore skipping weighted basis comparison.\n";
              out << "finePhysicalPoints:\n" << finePhysicalPoints;
              out << "coarsePhysicalPoints:\n" << coarsePhysicalPoints;
              break;
            }
            success = success && oldSuccess;
            
            FieldContainer<double> constrainingBasisValuesAllPoints = *constrainingDomainBasisCache->getTransformedValues(constrainingBasis, OP_VALUE);
            // strip cell dimension:
            constrainingBasisValuesAllPoints.resize(constrainingBasis->getCardinality(), numPoints);
            
            if (myCellIDs.find(constrainingEntityInfo.cellID) == myCellIDs.end())
            {
              // then we can't reliably call getBasisMap().
              // we rely on the same test on fewer processors (when the two cells end up in the same partition) to test this.
              // (Note that we *could* communicate the BasisMaps for the cell halo; that would do the trick.)
              
              continue;
            }
            
            auto coarseGlobalDofInfo = gda->getGlobalDofIndices(constrainingEntityInfo.cellID);
            BasisMap coarseBasisMap;
            if (isVolumeVar)
            {
              coarseBasisMap = gda->getBasisMap(constrainingEntityInfo.cellID, coarseGlobalDofInfo, var);
            }
            else
            {
              coarseBasisMap = gda->getBasisMap(constrainingEntityInfo.cellID, coarseGlobalDofInfo, var, constrainingSideOrdinal);
            }
            
            CellConstraints coarseCellConstraints = gda->getCellConstraints(constrainingEntityInfo.cellID);
            
            for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
            {
              FieldContainer<double> fineValues(fineBasis->getCardinality());
              for (int basisOrdinal=0; basisOrdinal<fineBasis->getCardinality(); basisOrdinal++)
              {
                fineValues(basisOrdinal) = fineValuesAllPoints(basisOrdinal,pointOrdinal);
              }
              map<GlobalIndexType,double> fineGlobalValues;
              for (auto subBasisMap : fineBasisMap)
              {
                // filter fine values according to what subBasisMap knows about.
                set<int> basisOrdinalFilter = subBasisMap->basisDofOrdinalFilter();
                FieldContainer<double> fineFilteredValues(basisOrdinalFilter.size());
                filterFCValues(fineFilteredValues, fineValues, basisOrdinalFilter, fineBasis->getCardinality());
                FieldContainer<double> mappedValues = subBasisMap->mapFineData(fineFilteredValues);
                vector<GlobalIndexType> globalIndices = subBasisMap->mappedGlobalDofOrdinals();
                for (int i=0; i<globalIndices.size(); i++)
                {
                  fineGlobalValues[globalIndices[i]] += mappedValues(i);
                }
              }
              FieldContainer<double> constrainingBasisValues(constrainingBasis->getCardinality());
              for (int basisOrdinal=0; basisOrdinal<constrainingBasis->getCardinality(); basisOrdinal++)
              {
                constrainingBasisValues(basisOrdinal) = constrainingBasisValuesAllPoints(basisOrdinal,pointOrdinal);
              }
              map<GlobalIndexType,double> coarseGlobalValues;
              for (auto subBasisMap : coarseBasisMap)
              {
                // filter constrainingBasisValues here according to what subBasisMap knows about.
                set<int> basisOrdinalFilter = subBasisMap->basisDofOrdinalFilter();
                FieldContainer<double> filteredConstrainingValues(basisOrdinalFilter.size());
                filterFCValues(filteredConstrainingValues, constrainingBasisValues, basisOrdinalFilter,
                               constrainingBasis->getCardinality());
                FieldContainer<double> mappedValues = subBasisMap->mapFineData(filteredConstrainingValues);
                vector<GlobalIndexType> globalIndices = subBasisMap->mappedGlobalDofOrdinals();
                for (int i=0; i<globalIndices.size(); i++)
                {
                  coarseGlobalValues[globalIndices[i]] += mappedValues(i);
                }
              }
              
              map<GlobalIndexType,double> nonzeroCoarseGlobalValues;
              map<GlobalIndexType,double> nonzeroFineGlobalValues;
              double tol = 1e-14;
              for (auto entry : coarseGlobalValues)
              {
                if (abs(entry.second) > tol)
                {
                  nonzeroCoarseGlobalValues[entry.first] = entry.second;
                }
              }
              // it can happen that fine basis participates in some global dofs that
              // coarse does not.  Below, we filter not only for nonzeros, but also to
              // eliminate any dofs that coarse basis does not see.
              for (auto entry : fineGlobalValues)
              {
                bool coarseBasisSkips = (coarseGlobalValues.find(entry.first) == coarseGlobalValues.end());
                if ((abs(entry.second) > tol) && !coarseBasisSkips)
                {
                  nonzeroFineGlobalValues[entry.first] = entry.second;
                }
              }
              
              // Compare coarseGlobalValues to fineGlobalValues
              if (nonzeroCoarseGlobalValues.size() != nonzeroFineGlobalValues.size())
              {
                success = false;
                cout << "Failure on fine cell " << cellID << ", side " << sideOrdinal;
                cout << ", subcell ordinal " << subcord << " of dimension " << subcdim << endl;
                cout << "(comparing with coarse cell " << constrainingCell->cellIndex();
                cout << ", side " << constrainingSideOrdinal << ")\n";
                out << "nonzeroCoarseGlobalValues.size() = " << nonzeroCoarseGlobalValues.size();
                out << " != " << nonzeroFineGlobalValues.size() << " = nonzeroFineGlobalValues().size()\n";
                print("nonzeroCoarseGlobalValues", nonzeroCoarseGlobalValues);
                print("nonzeroFineGlobalValues", nonzeroFineGlobalValues);
                cout << "physical point: (";
                for (int d=0; d < meshTopo->getDimension(); d++)
                {
                  cout << coarsePhysicalPoints(0,pointOrdinal,d);
                  if (d<sideDim) cout << ", ";
                }
                cout << ")\n";
                cout << "fine reference point: (";
                for (int d=0; d < sideDim; d++)
                {
                  cout << fineDomainPoints(pointOrdinal,d);
                  if (d<sideDim-1) cout << ", ";
                }
                cout << ")\n";
                cout << "coarse reference point: (";
                for (int d=0; d < sideDim; d++)
                {
                  cout << constrainingDomainPoints(pointOrdinal,d);
                  if (d<sideDim-1) cout << ", ";
                }
                cout << ")\n";
                
                for (auto subBasisMap : fineBasisMap)
                {
                  // filter fine values according to what subBasisMap knows about.
                  set<int> basisOrdinalFilter = subBasisMap->basisDofOrdinalFilter();
                  FieldContainer<double> fineFilteredValues(basisOrdinalFilter.size());
                  filterFCValues(fineFilteredValues, fineValues, basisOrdinalFilter, fineBasis->getCardinality());
                  FieldContainer<double> mappedValues = subBasisMap->mapFineData(fineFilteredValues);
                  vector<GlobalIndexType> globalIndices = subBasisMap->mappedGlobalDofOrdinals();
                  print("basisOrdinalFilter",basisOrdinalFilter);
                  print("globalIndices", globalIndices);
                  cout << "fineFilteredValues:\n" << fineFilteredValues;
                  cout << "mappedValues:\n" << mappedValues;
                }
                
              }
              else
              {
                bool savedSuccess = success;
                success = true;

                for (auto coarseGlobalValue : nonzeroCoarseGlobalValues)
                {
                  GlobalIndexType valueIndex = coarseGlobalValue.first;
                  double valueCoarse = coarseGlobalValue.second;
                  if (nonzeroFineGlobalValues.find(valueIndex) == nonzeroFineGlobalValues.end())
                  {
                    out << "failure: nonzeroFineGlobalValues does not have an entry for global dof ordinal " << valueIndex << endl;
                    success = false;
                  }
                  else
                  {
                    double valueFine = fineGlobalValues[valueIndex];
                    TEST_FLOATING_EQUALITY(valueCoarse, valueFine, tol);
                  }
                }
                
                if (!success)
                {
                  cout << "Failure on fine cell " << cellID << ", side " << sideOrdinal;
                  cout << ", subcell ordinal " << subcord << " of dimension " << subcdim << endl;
                  cout << "(comparing with coarse cell " << constrainingCell->cellIndex();
                  cout << ", side " << constrainingSideOrdinal << ")\n";
                  print("nonzeroCoarseGlobalValues", nonzeroCoarseGlobalValues);
                  print("nonzeroFineGlobalValues", nonzeroFineGlobalValues);
                  cout << "physical point: (";
                  for (int d=0; d < meshTopo->getDimension(); d++)
                  {
                    cout << coarsePhysicalPoints(0,pointOrdinal,d);
                    if (d<sideDim) cout << ", ";
                  }
                  cout << ")\n";
                  cout << "fine reference point: (";
                  for (int d=0; d < sideDim; d++)
                  {
                    cout << fineDomainPoints(pointOrdinal,d);
                    if (d<sideDim-1) cout << ", ";
                  }
                  cout << ")\n";
                  cout << "coarse reference point: (";
                  for (int d=0; d < sideDim; d++)
                  {
                    cout << constrainingDomainPoints(pointOrdinal,d);
                    if (d<sideDim-1) cout << ", ";
                  }
                  cout << ")\n";
                  
                  for (auto subBasisMap : fineBasisMap)
                  {
                    // filter fine values according to what subBasisMap knows about.
                    set<int> basisOrdinalFilter = subBasisMap->basisDofOrdinalFilter();
                    FieldContainer<double> fineFilteredValues(basisOrdinalFilter.size());
                    filterFCValues(fineFilteredValues, fineValues, basisOrdinalFilter, fineBasis->getCardinality());
                    FieldContainer<double> mappedValues = subBasisMap->mapFineData(fineFilteredValues);
                    vector<GlobalIndexType> globalIndices = subBasisMap->mappedGlobalDofOrdinals();
                    print("basisOrdinalFilter",basisOrdinalFilter);
                    print("globalIndices", globalIndices);
                    cout << "fineFilteredValues:\n" << fineFilteredValues;
                    cout << "mappedValues:\n" << mappedValues;
                  }
                }
                
                success = savedSuccess && success;
              }
            }
          }
        }
      }
    }
  }
  
  // ! copied from DPGTests GDAMinimumRuleTests
  class GDAMinimumRuleTests_UnitHexahedronBoundary : public SpatialFilter
  {
  public:
    bool matchesPoint(double x, double y, double z)
    {
      double tol = 1e-14;
      bool xMatch = (abs(x) < tol) || (abs(x-1.0) < tol);
      bool yMatch = (abs(y) < tol) || (abs(y-1.0) < tol);
      bool zMatch = (abs(z) < tol) || (abs(z-1.0) < tol);
      return xMatch || yMatch || zMatch;
    }
  };
  
  // ! copied from DPGTests GDAMinimumRuleTests
  SolutionPtr poissonExactSolution3D(int horizontalCells, int verticalCells, int depthCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces)
  {
    bool usePenaltyBCs = false;
    
    int spaceDim = 3;
    PoissonFormulation poissonForm(spaceDim, useH1Traces);
    
    VarPtr tau = poissonForm.tau();
    VarPtr q = poissonForm.q();
    
    VarPtr phi_hat = poissonForm.phi_hat();
    VarPtr psi_hat = poissonForm.psi_n_hat();
    
    VarPtr phi = poissonForm.phi();
    VarPtr psi = poissonForm.psi();
    
    BFPtr bf = poissonForm.bf();
    
    int testSpaceEnrichment = 3; //
    double width = 1.0, height = 1.0, depth = 1.0;
    
    vector<double> dimensions;
    dimensions.push_back(width);
    dimensions.push_back(height);
    dimensions.push_back(depth);
    
    vector<int> elementCounts;
    elementCounts.push_back(horizontalCells);
    elementCounts.push_back(verticalCells);
    elementCounts.push_back(depthCells);
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, testSpaceEnrichment);
    
    // rhs = f * v, where f = \Delta phi
    RHSPtr rhs = RHS::rhs();
    FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz();
    rhs->addTerm(f * q);
    
    IPPtr graphNorm = bf->graphNorm();
    
    BCPtr bc = BC::bc();
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    SolutionPtr solution;
    if (!usePenaltyBCs)
    {
      bc->addDirichlet(phi_hat, boundary, phi_exact);
      solution = Solution::solution(mesh, bc, rhs, graphNorm);
    }
    else
    {
      solution = Solution::solution(mesh, bc, rhs, graphNorm);
      SpatialFilterPtr entireBoundary = Teuchos::rcp( new GDAMinimumRuleTests_UnitHexahedronBoundary );
      
      Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
      pc->addConstraint(phi_hat==phi_exact,entireBoundary);
      
      solution->setFilter(pc);
    }
    
    return solution;
  }
  
  // ! copied from DPGTests GDAMinimumRuleTests
  SolutionPtr poissonExactSolution3DHangingNodes(int irregularity, FunctionPtr phi_exact, int H1Order)
  {
    // right now, we support 1-irregular and 2-irregular
    if ((irregularity > 2) || (irregularity < 0))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "only 1- and 2-irregularity supported");
    }
    int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 2, depthCellsInitialMesh = 1;
    
    bool useH1Traces = true; // "true" is the more thorough test
    
    SolutionPtr soln = poissonExactSolution3D(horizontalCellsInitialMesh, verticalCellsInitialMesh, depthCellsInitialMesh, H1Order, phi_exact, useH1Traces);
    
    if (irregularity==0) return soln;
    
    MeshPtr mesh = soln->mesh();
    
    //  cout << "about to refine to make Poisson 3D hanging node mesh.\n";
    
    GlobalIndexTypeToCast cellToRefine = 1;
    set<GlobalIndexType> cellIDs = {(GlobalIndexType)cellToRefine};
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());
    
    if (irregularity==1) return soln;
    
    // now, repeat the above, but with a 2-irregular mesh.
    auto myCellIDs = &mesh->cellIDsInPartition();
    int localCellIDToRefine = INT_MAX;
    if (mesh->getTopology()->isValidCellIndex(cellToRefine))
    {
      CellPtr parentCell = mesh->getTopology()->getCell(cellToRefine);
      vector<IndexType> children = parentCell->getChildIndices(mesh->getTopology());
      
      // childrenForSides outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
      vector< vector< pair< unsigned, unsigned > > > childrenForSides = parentCell->refinementPattern()->childrenForSides();
      
      for (int sideOrdinal=0; sideOrdinal<childrenForSides.size(); sideOrdinal++)
      {
        vector< pair< unsigned, unsigned > > childrenForSide = childrenForSides[sideOrdinal];
        bool foundCellToRefine = false;
        for (int i=0; i<childrenForSide.size(); i++)
        {
          unsigned childOrdinal = childrenForSide[i].first;
          IndexType childID = children[childOrdinal];
          unsigned childSideOrdinal = childrenForSide[i].second;
          if (myCellIDs->find(childID) != myCellIDs->end())
          {
            CellPtr child = mesh->getTopology()->getCell(childID);
            pair<GlobalIndexType,unsigned> neighborInfo = child->getNeighborInfo(childSideOrdinal, mesh->getTopology());
            GlobalIndexType neighborCellID = neighborInfo.first;
            if (neighborCellID != -1)   // not boundary
            {
              CellPtr neighbor = mesh->getTopology()->getCell(neighborCellID);
              pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(neighborInfo.second, mesh->getTopology());
              bool neighborIsPeer = neighborNeighborInfo.first == child->cellIndex();
              if (!neighborIsPeer)   // then by refining this cell, we induce a 2-irregular mesh
              {
                localCellIDToRefine = child->cellIndex();
                foundCellToRefine = true;
                break;
              }
            }
          }
        }
        if (foundCellToRefine) break;
      }
    }
    int commonCellIDToRefine;
    mesh->Comm()->MinAll(&localCellIDToRefine, &commonCellIDToRefine, 1);
    TEUCHOS_TEST_FOR_EXCEPTION(commonCellIDToRefine == INT_MAX, std::invalid_argument, "did not find a cell to refine");
    
    cellIDs = {(GlobalIndexType)commonCellIDToRefine};
    // this refinement will result in a 2-irregular mesh.
    // to avoid exceptions, we *must* defer rebuilding lookups.
    bool rebuildLookups = false;
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron(), rebuildLookups);
    
    return soln;
  }
  
  MeshPtr poissonUniformMesh(vector<int> elementWidths, int H1Order, bool useConformingTraces)
  {
    int spaceDim = elementWidths.size();
    int testSpaceEnrichment = spaceDim; //
    double span = 1.0; // in each spatial dimension
    
    vector<double> dimensions(spaceDim,span);
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    MeshPtr mesh = MeshFactory::rectilinearMesh(poissonForm.bf(), dimensions, elementWidths, H1Order, testSpaceEnrichment);
    return mesh;
  }
  
  MeshPtr poissonUniformMesh(int spaceDim, int elementWidth, int H1Order, bool useConformingTraces)
  {
    vector<int> elementCounts(spaceDim,elementWidth);
    return poissonUniformMesh(elementCounts, H1Order, useConformingTraces);
  }
  
  MeshPtr poissonIrregularMesh(int spaceDim, int irregularity, int H1Order, bool useConformingTraces)
  {
    int elementWidth = 2;
    if ((irregularity >= 2) && useConformingTraces)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can only use conforming traces when irregularity <= 1");
    }
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    
    int meshIrregularity = 0;
    GlobalIndexType cellToRefine = 1;
    vector<GlobalIndexType> cellsToRefine = {cellToRefine};
    
    auto myCellIDs = &mesh->cellIDsInPartition();
    
    int localSharedSideOrdinal = 0;
    bool ownRefinedCell = false;
    if (myCellIDs->find(cellToRefine) != myCellIDs->end())
    {
      ownRefinedCell = true;
      CellPtr cell = mesh->getTopology()->getCell(cellToRefine);
      localSharedSideOrdinal = -1;
      for (int sideOrdinal=0; sideOrdinal<cell->getSideCount(); sideOrdinal++)
      {
        if (cell->getNeighbor(sideOrdinal, mesh->getTopology()) != Teuchos::null)
        {
          localSharedSideOrdinal = sideOrdinal;
          break;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(localSharedSideOrdinal == -1, std::invalid_argument, "sharedSideOrdinal not found");
    }
    // communicate side ordinal to all processors
    int globalSharedSideOrdinal;
    mesh->Comm()->SumAll(&localSharedSideOrdinal, &globalSharedSideOrdinal, 1);
    
    while (meshIrregularity < irregularity)
    {
//      print("refining cells", cellsToRefine);
      cellsToRefine = {cellToRefine};
      mesh->hRefine(cellsToRefine);
      meshIrregularity++;
      
      // setup for the next refinement, if any:
      GlobalIndexTypeToCast localCellID = 0;
      localSharedSideOrdinal = 0;
      globalSharedSideOrdinal = 0;
      GlobalIndexTypeToCast commonCellID = 0;
      if (ownRefinedCell)
      {
        CellPtr cell = mesh->getTopology()->getCell(cellToRefine);
        auto childEntry = cell->childrenForSide(globalSharedSideOrdinal)[0];
        GlobalIndexType childWithNeighborCellID = childEntry.first;
        localSharedSideOrdinal = childEntry.second;
        localCellID = childWithNeighborCellID;
      }
      mesh->Comm()->SumAll(&localSharedSideOrdinal, &globalSharedSideOrdinal, 1);
      mesh->Comm()->SumAll(&localCellID, &commonCellID, 1);
      
      cellToRefine = commonCellID;
      myCellIDs = &mesh->cellIDsInPartition();
      ownRefinedCell = (myCellIDs->find(cellToRefine) != myCellIDs->end());
    }
    
//    mesh->getTopology()->baseMeshTopology()->printAllEntities();
    
    return mesh;
  }
  
  MeshPtr poisson3DUniformMesh()
  {
    return poissonUniformMesh(3, 2, 2, true);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson1D)
  {
    int spaceDim = 1;
    int elementWidth = 2;
    int H1Order = 2;
    bool useConformingTraces = true;
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson2DUniform)
  {
    int spaceDim = 2;
    int elementWidth = 2;
    int H1Order = 2;
    bool useConformingTraces = true;
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson2DHangingNode1Irregular_Slow)
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    int H1Order = 2;
    int irregularity = 1;
    bool useConformingTraces = true;
    MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order, useConformingTraces);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson2DTrianglesHangingNode1Irregular_Slow)
  {
    MPIWrapper::CommWorld()->Barrier();

    int spaceDim = 2;
    int H1Order = 2;
    bool useConformingTraces = true;
    
    int delta_k = spaceDim;

    vector<vector<double>> vertices = {{0,0},{1,0},{0.5,1}};
    vector<vector<IndexType>> elementVertices = {{0,1,2}};
    CellTopoPtr triangle = CellTopology::triangle();
    
    MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, {triangle}) );
    MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(geometry));
    
    out << "**** First mesh ****\n";
    
    // create a problematic mesh of a particular sort: refine once, then refine the interior element.  Then refine the interior element of the refined element.
    IndexType cellIDToRefine = 0, nextCellIndex = 1;
    int interiorChildOrdinal = 1; // interior child has index 1 in children
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(triangle);
    meshTopo->refineCell(cellIDToRefine, refPattern, nextCellIndex);
    nextCellIndex += refPattern->numChildren();
    
    vector<CellPtr> children = meshTopo->getCell(cellIDToRefine)->children();
    cellIDToRefine = children[interiorChildOrdinal]->cellIndex();
    meshTopo->refineCell(cellIDToRefine, refPattern, nextCellIndex);
    nextCellIndex += refPattern->numChildren();
    
    children = meshTopo->getCell(cellIDToRefine)->children();
    cellIDToRefine = children[interiorChildOrdinal]->cellIndex();
    meshTopo->refineCell(cellIDToRefine, refPattern, nextCellIndex);
    nextCellIndex += refPattern->numChildren();
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, poissonForm.bf(), H1Order, delta_k) );
    
    // The above mesh will cause some cascading constraints, which the new getBasisMap() can't
    // handle.  We have added logic to deal with this case to Mesh::enforceOneIrregularity().
    mesh->enforceOneIrregularity();

    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
    
    // now, try another, simpler (in some ways, anyway) mesh:
    // (this is one for which GMGOperator has had problems with an H^1-conforming quadratic trace basis)
    out << "**** Second mesh ****\n";
    bool useTriangles = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,2};
    
    mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                        elementCounts[0], elementCounts[1], useTriangles);
    
    set<GlobalIndexType> cellsToRefine = {2};
    mesh->hRefine(cellsToRefine);
    
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoissonContinuousGalerkin2DHangingNode_Slow)
  {
    int spaceDim = 2;
    int H1Order = 2;
    int delta_k = 0;
    MeshPtr mesh = minimalPoissonHangingNodeMesh(H1Order, delta_k, spaceDim, PoissonFormulation::CONTINUOUS_GALERKIN);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson2DHangingNode2Irregular_Slow)
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    int H1Order = 2;
    int irregularity = 2;

    bool useConformingTraces = false; // no longer support cascading constraints
    MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order, useConformingTraces);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
    
    // now, setup a simpler, but different, mesh, and test that:
    mesh = minimalPoissonHangingNodeMesh(H1Order, 2, spaceDim, PoissonFormulation::ULTRAWEAK);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson3DUniform_Slow)
  {
    int spaceDim = 3;
    int elementWidth = 1;
    int H1Order = 3;
    bool useConformingTraces = true;
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson3DHangingNode1Irregular_Slow)
  {
    int irregularity = 1;
    int spaceDim = 3;
    int H1Order = 2;
    // for 1-irregular meshes, can use conforming traces, and this is the harder test to pass.
    // for 2+-irregular meshes, conforming traces are not supported, so should use non-conforming.
    bool useConformingTraces = true;
    MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order, useConformingTraces);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoissonContinuousGalerkin3DHangingNode_Slow)
  {
    int spaceDim = 3;
    int H1Order = 2;
    int delta_k = 0;
    MeshPtr mesh = minimalPoissonHangingNodeMesh(H1Order, delta_k, spaceDim, PoissonFormulation::CONTINUOUS_GALERKIN);
    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson3DHangingNode2Irregular_Slow)
  {
    MPIWrapper::CommWorld()->Barrier();
    int irregularity = 2;
    int spaceDim = 3;
    int H1Order = 1;
    // for 1-irregular meshes, can use conforming traces, and this is the harder test to pass.
    // for 2+-irregular meshes, conforming traces are not supported, so should use non-conforming.
    bool useConformingTraces = false;
    MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order, useConformingTraces);

    testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, CheckConstraintsPoisson3DUniform )
  {
    MeshPtr mesh = poisson3DUniformMesh();
    testSubcellConstraintIsAncestor(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, CheckConstraintsPoisson3DHangingNode1Irregular )
  {
    int irregularity = 1;
    int spaceDim = 3;
    int H1Order = 2;
    // for 1-irregular meshes, can use conforming traces, and this is the harder test to pass.
    // for 2+-irregular meshes, conforming traces are not supported, so should use non-conforming.
    bool useConformingTraces = true;
    MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order, useConformingTraces);
    testSubcellConstraintIsAncestor(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, CheckConstraintsPoisson3DHangingNode2Irregular )
  {
    MPIWrapper::CommWorld()->Barrier();
    int irregularity = 2;
    int spaceDim = 3;
    int H1Order = 1;
    // for 1-irregular meshes, can use conforming traces, and this is the harder test to pass.
    // for 2+-irregular meshes, conforming traces are not supported, so should use non-conforming.
    bool useConformingTraces = false;
    MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order, useConformingTraces);
    testSubcellConstraintIsAncestor(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, ContiguousGlobalDofNumberingComplexSpaceTimeMesh )
  {
    bool useConformingTraces = true;
    testContiguousGlobalDofNumberingComplexSpaceTimeMesh(useConformingTraces, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, DistributeSubcellDofIndices )
  {
    // test the static method that does the distribution
    Epetra_CommPtr Comm = MPIWrapper::CommWorld();
    int rank = Comm->MyPID();
    int numProcs = Comm->NumProc();
    
    // set up a fake example in which all dof indices are local.
    // one hexahedral element per MPI rank
    CellTopoPtr hex = CellTopology::hexahedron();
    int H1Order = 3;
    BasisPtr basis = BasisFactory::basisFactory()->getBasis(H1Order, hex, Camellia::EFunctionSpace::FUNCTION_SPACE_HGRAD);
    
    auto setupSubcellDofIndices = [basis] (PartitionIndexType rank) -> SubcellDofIndices
    {
      int localDofCount = basis->getCardinality();
      GlobalIndexType globalDofOffset = localDofCount * rank;
      
      CellTopoPtr cellTopo = basis->domainTopology();
      int spaceDim = cellTopo->getDimension();

      int varID = 3;
      
      SubcellDofIndices scDofIndices;
      scDofIndices.subcellDofIndices.resize(spaceDim+1);
      for (int d=0; d<=spaceDim; d++)
      {
        int numSubcells = cellTopo->getSubcellCount(d);
        for (int scord=0; scord<numSubcells; scord++)
        {
          vector<int> localDofOrdinals = basis->dofOrdinalsForSubcell(d, scord);
          vector<GlobalIndexType> globalDofIndices;
          for (int localDofOrdinal : localDofOrdinals)
          {
            globalDofIndices.push_back(localDofOrdinal + globalDofOffset);
          }
          scDofIndices.subcellDofIndices[d][scord][varID] = globalDofIndices;
        }
      }
      return scDofIndices;
    };

    SubcellDofIndices scDofIndices = setupSubcellDofIndices(rank);
    
    GlobalIndexType myCellID = (GlobalIndexType) rank;
    map<GlobalIndexType,PartitionIndexType> myNeighbors;
    GlobalIndexType neighbor1 = (rank + 1) % numProcs;
    GlobalIndexType neighbor2 = (rank + 2) % numProcs;
    
    if (myCellID != neighbor1)
    {
      myNeighbors[neighbor1] = (PartitionIndexType) neighbor1;
    }
    if (myCellID != neighbor2)
    {
      myNeighbors[neighbor2] = (PartitionIndexType) neighbor2;
    }
    
    map<GlobalIndexType,SubcellDofIndices> mySubcellDofIndices;
    mySubcellDofIndices[myCellID] = scDofIndices;
    
    map<GlobalIndexType,SubcellDofIndices> neighborSubcellDofIndices;
    GDAMinimumRule::distributeSubcellDofIndices(Comm, mySubcellDofIndices, myNeighbors, neighborSubcellDofIndices);
    
    for (auto entry : myNeighbors)
    {
      GlobalIndexType neighborCellID = entry.first;
      SubcellDofIndices actualDofIndices = neighborSubcellDofIndices[neighborCellID];
      PartitionIndexType neighborRank = entry.second;
      SubcellDofIndices expectedDofIndices = setupSubcellDofIndices(neighborRank);
      TEST_ASSERT(actualDofIndices.subcellDofIndices == expectedDofIndices.subcellDofIndices);
    }
  }

  TEUCHOS_UNIT_TEST( GDAMinimumRule, DofCount_ContinuousTriangles )
  {
    // a couple tests to ensure that the dof counts are correct in linear, triangular Bubnov-Galerkin meshes
    int spaceDim = 2;
    bool useConformingTraces = true;
    bool useTriangles = true;
    PoissonFormulation form(spaceDim, useConformingTraces, PoissonFormulation::CONTINUOUS_GALERKIN);
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,1};
    
    int H1Order = 1, delta_k = 0;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    // to get accurate global entity counts, need to use a non-distributed MeshTopology
    MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                                             elementCounts[0], elementCounts[1], useTriangles);
    int vertexDim = 0;
    int numVertices = meshTopo->getEntityCount(vertexDim);
    
    int globalDofCount = mesh->numGlobalDofs();
    TEST_EQUALITY(numVertices, globalDofCount);
    
    elementCounts = {16,16};
    mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                        elementCounts[0], elementCounts[1], useTriangles);
    meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                             elementCounts[0], elementCounts[1], useTriangles);
    numVertices = meshTopo->getEntityCount(vertexDim);
    
    globalDofCount = mesh->numGlobalDofs();
    TEST_EQUALITY(numVertices, globalDofCount);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, DofCount_UltraweakConformingTriangles )
  {
    // a couple tests to ensure that the dof counts are correct in linear, triangular meshes
    int spaceDim = 2;
    bool useConformingTraces = true;
    bool useTriangles = true;
    PoissonFormulation form(spaceDim, useConformingTraces, PoissonFormulation::ULTRAWEAK);
  
    /*
     Ultraweak conforming Poisson formulation has the following variables:
     phi - scalar field
     psi - vector field
     phi_hat - H^1 conforming trace
     psi_n_hat - L^2 conforming flux
     
     The global degree of freedom count we expect for a uniform triangular linear H^1 mesh is as follows:
     phi - 1 dof/triangle (constant basis)
     psi - 2 dofs/triangle (constant basis)
     phi_hat - 1 dof/vertex (linear basis)
     psi_n_hat - 1 dof/edge (constant basis)
     
     */
    
    auto dofCount = [](int numTriangles, int numEdges, int numVertices) {
      return 3 * numTriangles + numVertices + numEdges;
    };
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,1};
    
    int H1Order = 1, delta_k = 0;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    // to get accurate global entity counts, need to use a non-distributed MeshTopology
    MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                                             elementCounts[0], elementCounts[1], useTriangles);
    int vertexDim = 0;
    int edgeDim = 1;
    int numVertices = meshTopo->getEntityCount(vertexDim);
    int numElements = meshTopo->activeCellCount();
    int numEdges = meshTopo->getEntityCount(edgeDim);
    
    int globalDofCount = mesh->numGlobalDofs();
    int expectedGlobalDofCount = dofCount(numElements,numEdges,numVertices);
    TEST_EQUALITY(expectedGlobalDofCount, globalDofCount);
    
    elementCounts = {16,16};
    mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                        elementCounts[0], elementCounts[1], useTriangles);
    meshTopo =  MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                              elementCounts[0], elementCounts[1], useTriangles);

    numVertices = meshTopo->getEntityCount(vertexDim);
    numElements = meshTopo->activeCellCount();
    numEdges = meshTopo->getEntityCount(edgeDim);
    
    globalDofCount = mesh->numGlobalDofs();
    expectedGlobalDofCount = dofCount(numElements,numEdges,numVertices);
    TEST_EQUALITY(expectedGlobalDofCount, globalDofCount);
  }

  TEUCHOS_UNIT_TEST( GDAMinimumRule, InterpretGlobalBasisCoefficientsUltraweakConforming_Triangles )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    bool useConformingTraces = true;
    bool useTriangles = true;
    PoissonFormulation form(spaceDim, useConformingTraces, PoissonFormulation::ULTRAWEAK);
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {4,4};
    
    int H1Order = 1, delta_k = 0;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    PartitionIndexType rank = mesh->Comm()->MyPID();
//    int numProcs = mesh->Comm()->NumProc();
//
//    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
//    for (int p=0; p<numProcs; p++)
//    {
//      if (rank == p)
//      {
//        cout << "rank " << p << " global dof info:\n";
//        minRule->printGlobalDofInfo();
//      }
//      mesh->Comm()->Barrier();
//    }
    
    double prescribedValue = 1.0;
    SolutionPtr soln = Solution::solution(form.bf(), mesh);
    BCPtr bc = BC::bc();
    VarPtr phi_hat = form.phi_hat();
    bc->addDirichlet(phi_hat, SpatialFilter::allSpace(), Function::constant(prescribedValue));
    soln->initializeLHSVector();
    
    Intrepid::FieldContainer<GlobalIndexType> bcGlobalIndices;
    Intrepid::FieldContainer<double> bcGlobalValues;
    
    set<GlobalIndexType> myGlobalIndicesSet = mesh->globalDofAssignment()->globalDofIndicesForPartition(rank);
    
    mesh->boundary().bcsToImpose(bcGlobalIndices,bcGlobalValues,*bc, myGlobalIndicesSet, mesh->globalDofAssignment().get());
    
    // we prescribe 1 at at all the bc indices, and 0 everywhere else.
    // then we check the values on the elements.
    // for now, we are interested in cell 0, side 2 in particular, because of a bug.
    
    auto lhsVector = soln->getLHSVector();
    
    for (int i=0; i<bcGlobalIndices.size(); i++)
    {
      GlobalIndexTypeToCast globalIndex = bcGlobalIndices[i];
      int LID = lhsVector->Map().LID(globalIndex);
      (*lhsVector)[0][LID] = bcGlobalValues[i];
    }
    
    soln->importSolution();
    // for each of my cells, find vertices on the mesh boundary.
    set<GlobalIndexType> myCellIDs = mesh->cellIDsInPartition();
    int numPoints = 4;
    FieldContainer<double> refLinePoints(numPoints,1);
    refLinePoints(0,0) = -1.0;
    refLinePoints(1,0) = -0.5;
    refLinePoints(2,0) =  0.5;
    refLinePoints(3,0) =  1.0;
    
    double tol = 1e-14;
    FunctionPtr phi_soln = Function::solution(phi_hat, soln);
    FieldContainer<double> phiValues(1,numPoints);
    for (GlobalIndexType cellID : myCellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      int sideCount = basisCache->cellTopology()->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
      {
        BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
        sideCache->setRefCellPoints(refLinePoints);
        phi_soln->values(phiValues, sideCache);
        
        // are there some physical points that lie on the boundary?
        // we expect to match prescribedValue at these
        const FieldContainer<double>* physicalPoints = &sideCache->getPhysicalCubaturePoints();
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          double x = (*physicalPoints)(0,pointOrdinal,0);
          double y = (*physicalPoints)(0,pointOrdinal,1);
          if ((abs(x-0) < tol) || (abs(x-1) < tol) || (abs(y-0) < tol) || (abs(y-1) < tol))
          {
            double actualValue = phiValues(0,pointOrdinal);
            out << "testing value for (" << x << "," << y << ") on cell " << cellID << ", side " << sideOrdinal << endl;
            TEST_FLOATING_EQUALITY(actualValue, prescribedValue, tol);
          }
        }
      }
    }
    HDF5Exporter exporter(mesh, "PoissonGlobalBCExample", "/tmp");
    int num1DPoints = 20;
    exporter.exportSolution(soln, 0, num1DPoints);
//    success = false;
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, InterpretLocalBasisCoefficientsHangingNode_Triangles )
  {
    /*
     This test replicates an issue that caused problems for GMGOperator.
     
     ____________
     |         /|
     |        / |
     |       /  |
     |      /   |
     |     /____|
     |    /|    |
     |   / |   /|
     |  /  |  / |
     | /   | /  |
     |/____|/___| <-- This vertex dof is one we've had trouble with
     |         /|
     |        / |
     |       /  |
     |      /   |
     |     /    |
     |    /     |
     |   /      |
     |  /       |
     | /        |
     |/_________|
     
     */
    int spaceDim = 2;
    bool useConformingTraces = true;
    bool useTriangles = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,2};
    
    int H1Order = 2, delta_k = 1;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
    
    set<GlobalIndexType> cellsToRefine = {2};
    mesh->hRefine(cellsToRefine);
    
    // would be better to get the cellID and vertex from the geometry rather than hardcoding, in case
    // the numbering of these things changes.
    GlobalIndexType cellID = 6; // cell 6 is the bottom-right guy in the refined cell
    unsigned cellVertex = 1;    // the problematic vertex pointed out above
    VarPtr phi_hat = form.phi_hat(); // H^1-conforming variable
    unsigned vertexDim = 0;
    unsigned cellSideOrdinal = 0; // side 0 is the one with the hanging node
    
    set<GlobalIndexType> myCells = mesh->cellIDsInPartition();
    // Since globalDofIndicesForVarOnSubcell() is only guaranteed to work for a local cell, we only run this test on the owning
    // MPI rank.
    if (myCells.find(cellID) != myCells.end())
    {
      // we start by representing the global dof for the problematic vertex in terms of the local basis on the refined cell.
      set<GlobalIndexType> globalIndices = minRule->globalDofIndicesForVarOnSubcell(phi_hat->ID(), cellID, vertexDim, cellVertex);
      // there should be exactly one globalIndex for the vertex
      TEST_EQUALITY(globalIndices.size(), 1);
      int vertexGlobalDofIndex = *globalIndices.begin();
     
      Epetra_SerialComm SerialComm; // rank-local map
      Epetra_Map    localXMap(1, 1, &vertexGlobalDofIndex, 0, SerialComm);
      Teuchos::RCP<Epetra_Vector> XLocal = Teuchos::rcp( new Epetra_Vector(localXMap) );
      (*XLocal)[0] = 1.0;
      
      FieldContainer<double> localCoefficients;
      minRule->interpretGlobalCoefficients(cellID, localCoefficients, *XLocal);
      
      DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
      BasisPtr basis = trialOrder->getBasis(phi_hat->ID(), cellSideOrdinal);
      FieldContainer<double> localBasisCoefficients(basis->getCardinality());
      
      for (int basisOrdinal=0; basisOrdinal<basis->getCardinality(); basisOrdinal++)
      {
        int localDofIndex = trialOrder->getDofIndex(phi_hat->ID(), basisOrdinal, cellSideOrdinal);
        localBasisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      
      FieldContainer<double> globalCoefficients;
      FieldContainer<GlobalIndexType> globalDofIndices;
      minRule->interpretLocalBasisCoefficients(cellID, phi_hat->ID(), cellSideOrdinal,
                                               localBasisCoefficients, globalCoefficients, globalDofIndices);
      
      // expect 1.0 weight for vertexGlobalDofIndex, 0 for any others
      bool foundVertexDofIndex = false;
      double tol = 1e-13;
      for (int i=0; i<globalDofIndices.size(); i++)
      {
        if (globalDofIndices[i] == vertexGlobalDofIndex)
        {
          foundVertexDofIndex = true;
          TEST_FLOATING_EQUALITY(globalCoefficients[i], (*XLocal)[0], tol);
        }
        else
        {
          TEST_COMPARE(abs(globalCoefficients[i]), <, tol);
        }
      }
      if (!foundVertexDofIndex)
      {
        success = false;
        out << "vertexGlobalDofIndex " << vertexGlobalDofIndex << " not mapped by interpretLocalBasisCoefficients(): FAILED\n";
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, OneIrregularityEnforcement_2D)
  {
    int spaceDim = 2;
    int irregularity = 2;
    int H1Order = 2;
    bool useConformingTraces = false;
    MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order, useConformingTraces);
    
    GlobalIndexType activeElementCount_initial = mesh->numActiveElements();
    
    mesh->enforceOneIrregularity();
    
    GlobalIndexType activeElementCount_final = mesh->numActiveElements();
    
    if (activeElementCount_final <= activeElementCount_initial)
    {
      out << "Failure: # of elements did not increase during 1-irregularity enforcement of 2D mesh, even though the mesh is 2-irregular.\n";
      success = false;
    }
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, OneIrregularityEnforcement_3D )
  {
    // very simple test: take a 2-irregular mesh, count # elements, enforce 1-irregularity.  Just check that there are more elements after enforcement.
    
    // important thing here is the irregularity is 2:
    int irregularity = 2;
    FunctionPtr phi_exact = Function::zero();
    int H1Order = 2;
    
    SolutionPtr soln = poissonExactSolution3DHangingNodes(irregularity,phi_exact,H1Order);
    MeshPtr mesh = soln->mesh();
    
    GlobalIndexType activeElementCount_initial = mesh->numActiveElements();
    
    mesh->enforceOneIrregularity();
    
    GlobalIndexType activeElementCount_final = mesh->numActiveElements();
    
    if (activeElementCount_final <= activeElementCount_initial)
    {
      out << "Failure in test1IrregularityEnforcement: # of elements did not increase during 1-irregularity enforcement of 3D mesh, even though the mesh is 2-irregular.\n";
      success = false;
    }
  }

  void testProjectionOntoTriangularContinuousGalerkinMesh(int meshWidth, int polyOrder, Teuchos::FancyOStream &out, bool &success)
  {
    bool useTriangles = true;
    int spaceDim = 2;
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces, PoissonFormulation::CONTINUOUS_GALERKIN);
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {meshWidth,meshWidth};
    
    int H1Order = polyOrder, delta_k = 0;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    SolutionPtr soln = Solution::solution(mesh);
    FunctionPtr phiExact = 2 * Function::xn(1) + Function::yn(1);
    map<int, FunctionPtr> projectionMap;
    projectionMap[form.phi()->ID()] = phiExact;
    soln->projectOntoMesh(projectionMap);
    soln->initializeLHSVector();
    soln->importSolution();
    
    FunctionPtr projectedFunction = Function::solution(form.phi(), soln);
    double err = (projectedFunction - phiExact)->l2norm(mesh);
    TEST_COMPARE(err, <, 1e-13);
    
    if (!success)
    {
      // DEBUGGING
      mesh->getTopology()->printAllEntitiesInBaseMeshTopology();

      GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
      minRule->printGlobalDofInfo();
      
      minRule->printConstraintInfo(0);
      minRule->getDofMapper(0)->printMappingReport();
      
      minRule->printConstraintInfo(1);
      minRule->getDofMapper(1)->printMappingReport();

      string outputSuperDir = "/tmp";
      string outputDir = "PoissonSoln";
      HDF5Exporter exporter(mesh, outputDir, outputSuperDir);
      cout << "Writing phi to " << outputSuperDir << "/" << outputDir << endl;

      exporter.exportSolution(soln);
    }
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, ProjectOntoTriangularUniformMesh_Linear )
  {
    int polyOrder = 1;
    int meshWidth = 2;
    testProjectionOntoTriangularContinuousGalerkinMesh(meshWidth, polyOrder, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, ProjectOntoTriangularUniformMesh_Quadratic )
  {
    int polyOrder = 2;
    int meshWidth = 1;
    testProjectionOntoTriangularContinuousGalerkinMesh(meshWidth, polyOrder, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, SolvePoisson2DContinuousGalerkinHangingNode_Slow )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    testSolvePoissonContinuousGalerkinHangingNode(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, SolvePoisson2DUltraweakHangingNode1Irregular_Slow)
  {
    int spaceDim = 2;
    testSolvePoissonUltraweakHangingNode(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GDAMinimumRule, SolvePoisson3DContinuousGalerkinHangingNode_Slow )
  {
    int spaceDim = 3;
    testSolvePoissonContinuousGalerkinHangingNode(spaceDim, out, success);
  }

  TEUCHOS_UNIT_TEST( GDAMinimumRule, SolvePoisson3DHangingNode_Slow )
  {
    // exact solution: for now, we just use a linear phi
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    //  FunctionPtr phi_exact = x + y;
    FunctionPtr phi_exact = -x + y + z;
    //  FunctionPtr phi_exact = Function::constant(3.14159);
    
    int H1Order = 2; // 1 higher than the order of phi_exact, to get an exactly recoverable solution with L^2 fields.
    int spaceDim = 3;
    bool useConformingTraces = true;
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);

    for (int irregularity = 1; irregularity<=1; irregularity++)
    {
      SolutionPtr soln = poissonExactSolution3DHangingNodes(irregularity,phi_exact,H1Order);
      
      MeshPtr mesh = soln->mesh();
      VarFactoryPtr vf = soln->mesh()->bilinearForm()->varFactory();
      
      if (! MeshTestUtility::checkLocalGlobalConsistency(mesh) )
      {
        cout << "FAILURE: " << irregularity << "-irregular Poisson 3D mesh fails local-to-global consistency check.\n";
        success = false;
        //    return success;
      }
      
      VarPtr phi = poissonForm.phi();
      VarPtr phi_hat = poissonForm.phi_hat();
      
      map<int, FunctionPtr> phi_exact_map;
      phi_exact_map[phi->ID()] = phi_exact;
      soln->projectOntoMesh(phi_exact_map);
      
      FunctionPtr phi_soln = Function::solution(phi, soln);
      FunctionPtr phi_err = phi_soln - phi_exact;
      
      FunctionPtr phi_hat_soln = Function::solution(phi_hat, soln);
      
      double tol = 1e-12;
      double phi_err_l2 = phi_err->l2norm(mesh);
      
      soln->clear();
      soln->solve();
      
      //    cout << irregularity << "-irregular 3D poisson w/hanging node solved.  About to check solution continuities.\n";
      
      Epetra_MultiVector *lhsVector = soln->getGlobalCoefficients();
      Epetra_SerialComm Comm;
      Epetra_Map partMap = soln->getPartitionMap();
      
      // Import solution onto current processor
      GlobalIndexTypeToCast numNodesGlobal = mesh->numGlobalDofs();
      GlobalIndexTypeToCast numMyNodes = numNodesGlobal;
      Epetra_Map     solnMap(numNodesGlobal, numMyNodes, 0, Comm);
      Epetra_Import  solnImporter(solnMap, partMap);
      Epetra_Vector  solnCoeff(solnMap);
      solnCoeff.Import(*lhsVector, solnImporter, Insert);
      
      if ( ! MeshTestUtility::neighborBasesAgreeOnSides(mesh, solnCoeff))
      {
        cout << "GDAMinimumRuleTests failure: for" << irregularity << "-irregular 3D Poisson mesh with hanging nodes (after solving), neighboring bases do not agree on sides." << endl;
        success = false;
      }
      
      //    cout << "...solution continuities checked.\n";
      
      phi_err_l2 = phi_err->l2norm(mesh);
      if (phi_err_l2 > tol)
      {
        success = false;
        cout << "GDAMinimumRuleTests failure: for " << irregularity << "-irregular 3D mesh and exactly recoverable solution, phi error is " << phi_err_l2 << endl;
        
        string outputSuperDir = ".";
        string outputDir = "poisson3DHangingNode";
        HDF5Exporter exporter(mesh, outputDir, outputSuperDir);
        cout << "Writing phi err to " << outputSuperDir << "/" << outputDir << endl;
        
        exporter.exportFunction(phi_err, "phi_err");
      }
    }
  }
} // namespace