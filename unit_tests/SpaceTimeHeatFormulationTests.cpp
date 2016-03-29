//
//  SpaceTimeHeatFormulation
//  Camellia
//
//  Created by Nate Roberts on 11/25/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Boundary.h"
#include "CamelliaDebugUtility.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SpaceTimeHeatFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
  void projectExactSolution(SpaceTimeHeatFormulation &form, SolutionPtr heatSolution, FunctionPtr u)
  {
    double epsilon = form.epsilon();
    
    FunctionPtr sigma1, sigma2, sigma3;
    int spaceTimeDim = heatSolution->mesh()->getDimension();
    int spaceDim = spaceTimeDim - 1;
    
    sigma1 = epsilon * u->dx();
    if (spaceDim > 1) sigma2 = epsilon * u->dy();
    if (spaceDim > 2) sigma3 = epsilon * u->dz();
    
    LinearTermPtr sigma_n_lt = form.sigma_n_hat()->termTraced();
    LinearTermPtr u_lt = form.u_hat()->termTraced();
    
    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u()->ID()] = u;
    exactMap[form.sigma(1)->ID()] = sigma1;
    if (spaceDim > 1) exactMap[form.sigma(2)->ID()] = sigma2;
    if (spaceDim > 2) exactMap[form.sigma(3)->ID()] = sigma3;
    
    // flux:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr sigma_n = sigma_n_lt->evaluate(exactMap);
    exactMap[form.sigma_n_hat()->ID()] = sigma_n;
    
    // traces:
    FunctionPtr u_hat = u_lt->evaluate(exactMap);
    exactMap[form.u_hat()->ID()] = u_hat;
    
    heatSolution->projectOntoMesh(exactMap);
  }
  
  void setupExactSolution(SpaceTimeHeatFormulation &form, FunctionPtr u,
                          MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k)
  {
    double epsilon = form.epsilon();
    
    FunctionPtr sigma1, sigma2, sigma3;
    int spaceTimeDim = meshTopo->getDimension();
    int spaceDim = spaceTimeDim - 1;
    
    FunctionPtr forcingFunction = SpaceTimeHeatFormulation::forcingFunction(spaceDim, epsilon, u);
    
    form.initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction);
  }
  
  void testForcingFunctionForConstantU(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    // Forcing function should be zero for constant u
    FunctionPtr f_expected = Function::zero();
    
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    
    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);
    
    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;
    
    FunctionPtr u = Function::constant(0.5);
    
    bool useConformingTraces = true;
    SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
    
    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);
    
    MeshPtr mesh = form.solution()->mesh();
    
    FunctionPtr f_actual = form.forcingFunction(spaceDim, epsilon, u);
    
    double l2_diff = (f_expected-f_actual)->l2norm(mesh);
    TEST_COMPARE(l2_diff, <, 1e-14);
  }
  
  void testSpaceTimeHeatConsistency(int spaceDim, bool useConformingTraces, bool useHangingNodes, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // 1^d mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    
    int numElementsTime = useHangingNodes ? 2 : 1;
    
    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1, numElementsTime);
    
    if (useHangingNodes)
    {
      int numElements = spaceTimeMeshTopo->cellCount();
      GlobalIndexType cellToRefine = 0;
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(spaceTimeMeshTopo->getCell(cellToRefine)->topology());
      spaceTimeMeshTopo->refineCell(0, refPattern, numElements);
    }
    
    double epsilon = 0.1;
    int fieldPolyOrder = 2, delta_k = 1;
    
    FunctionPtr u;
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    FunctionPtr t = Function::tn(1);
    
    if (spaceDim == 1)
    {
      u = x * t;
    }
    else if (spaceDim == 2)
    {
      u = x * t + y;
    }
    else if (spaceDim == 3)
    {
      u = x * t + y - z;
    }
    
    SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
    
    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);
    
    bool DEBUGGING = false; // set to true to print some information to console...
    if (DEBUGGING)
    {
      // DEBUGGING
      if (spaceDim == 2)
      {
        GlobalIndexType cellID = 0;
        
        GlobalDofAssignmentPtr gda = form.solution()->mesh()->globalDofAssignment();
        GDAMinimumRule* gdaMinRule = dynamic_cast<GDAMinimumRule*>(gda.get());
        gdaMinRule->printConstraintInfo(cellID);
        
        DofOrderingPtr trialOrder = form.solution()->mesh()->getElementType(cellID)->trialOrderPtr;
        Intrepid::FieldContainer<double> dofCoefficients(trialOrder->totalDofs());
        dofCoefficients[82] = 1.0;
        printLabeledDofCoefficients(form.bf()->varFactory(), trialOrder, dofCoefficients);
        
        
        VarPtr uHat = form.u_hat();
        int sideOrdinal = 0;
        int basisOrdinal = 1; // the one we seek, corresponding to 82 above
        BasisPtr uHatBasis = trialOrder->getBasis(uHat->ID(),sideOrdinal);
        
        int sideDim = uHatBasis->domainTopology()->getDimension();
        for (int subcdim=0; subcdim<=sideDim; subcdim++)
        {
          int subcCount = uHatBasis->domainTopology()->getSubcellCount(subcdim);
          for (int subcord=0; subcord<subcCount; subcord++)
          {
            const vector<int>* dofOrdinalsForSubcell = &uHatBasis->dofOrdinalsForSubcell(subcdim, subcord);
            if (std::find(dofOrdinalsForSubcell->begin(), dofOrdinalsForSubcell->end(), basisOrdinal) != dofOrdinalsForSubcell->end())
            {
              cout << "basisOrdinal " << basisOrdinal << " belongs to subcell " << subcord << " of dimension " << subcdim << endl;
            }
          }
        }
        
      }
    }
    
    projectExactSolution(form, form.solution(), u);
    
    form.solution()->clearComputedResiduals();
    
    double energyError = form.solution()->energyErrorTotal();
    
//    if (spaceDim != 3)
//    {
//      MeshPtr mesh = form.solution()->mesh();
//      string outputDir = "/tmp";
//      string solnName = (spaceDim == 1) ? "spaceTimeHeatSolution_1D" : "spaceTimeHeatSolution_2D";
//      cout << "\nDebugging: Outputting spaceTimeHeatSolution to " << outputDir << "/" << solnName << endl;
//      HDF5Exporter exporter(mesh, solnName, outputDir);
//      exporter.exportSolution(form.solution());
//    }
    
    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }
  
  void testSpaceTimeHeatConsistencyConstantSolution(int spaceDim, bool useHangingNodes, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    
    int numElementsTime = useHangingNodes ? 2 : 1;
    
    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1, numElementsTime);
    
    if (useHangingNodes)
    {
      int numElements = spaceTimeMeshTopo->cellCount();
      GlobalIndexType cellToRefine = 0;
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(spaceTimeMeshTopo->getCell(cellToRefine)->topology());
      spaceTimeMeshTopo->refineCell(0, refPattern, numElements);
    }
    
    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;
    
    FunctionPtr u = Function::constant(0.5);
    
    bool useConformingTraces = true;
    SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
    
    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);
    projectExactSolution(form, form.solution(), u);
    
    form.solution()->clearComputedResiduals();
    
    double energyError = form.solution()->energyErrorTotal();
    
    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }
  
  void testSpaceTimeHeatImposeConstantTraceBCs(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    
    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);
    
    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;
    
    static const double CONST_VALUE = 0.5;
    FunctionPtr u = Function::constant(CONST_VALUE);
    
    bool useConformingTraces = true;
    SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
    
    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);
    
    VarPtr u_hat = form.u_hat();
    BCPtr bc = form.solution()->bc();
    
    SpatialFilterPtr allTime = SpatialFilter::matchingT(t0) | SpatialFilter::matchingT(t1);
    bc->addDirichlet(u_hat, SpatialFilter::allSpace() | allTime, u);
    
    MeshPtr mesh = form.solution()->mesh();
    
    Boundary boundary = mesh->boundary();
    DofInterpreter* dofInterpreter = form.solution()->getDofInterpreter().get();
    std::map<GlobalIndexType, double> globalDofIndicesAndValues;
    GlobalIndexType cellID = 0;
    set<pair<int, unsigned>> singletons;
    boundary.bcsToImpose<double>(globalDofIndicesAndValues, *bc, cellID, singletons, dofInterpreter);
    
    // use our knowledge that we have a one-element mesh: every last dof for u_hat should be present, and have coefficient CONST_VALUE
    DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
    CellTopoPtr cellTopo = mesh->getElementType(cellID)->cellTopoPtr;
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    
    double tol = 1e-13;
    for (int sideOrdinal=0; sideOrdinal < cellTopo->getSideCount(); sideOrdinal++)
    {
      out << "******** SIDE " << sideOrdinal << " ********" << endl;
      BasisPtr basis = trialOrder->getBasis(u_hat->ID(),sideOrdinal);
      Intrepid::FieldContainer<double> uValues(basis->getCardinality());
      uValues.initialize(CONST_VALUE);
      Intrepid::FieldContainer<double> globalData;
      Intrepid::FieldContainer<GlobalIndexType> globalDofIndices;
      dofInterpreter->interpretLocalBasisCoefficients(cellID, u_hat->ID(), sideOrdinal, uValues, globalData, globalDofIndices);
      // sanity check on the interpreted global values
      for (int basisOrdinal=0; basisOrdinal<globalData.size(); basisOrdinal++)
      {
        TEST_FLOATING_EQUALITY(CONST_VALUE, globalData(basisOrdinal), tol);
      }
      
      for (int basisOrdinal=0; basisOrdinal<globalData.size(); basisOrdinal++)
      {
        GlobalIndexType globalDofIndex = globalDofIndices(basisOrdinal);
        if (globalDofIndicesAndValues.find(globalDofIndex) != globalDofIndicesAndValues.end())
        {
          double expectedValue = globalData(basisOrdinal);
          double actualValue = globalDofIndicesAndValues[globalDofIndex];
          TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
        }
        else
        {
          out << "On side " << sideOrdinal << ", did not find globalDofIndex " << globalDofIndex << endl;
          success = false;
        }
      }
    }
  }
  
  void testSpaceTimeHeatSolveConstantSolution(int spaceDim, bool useTraceBCsEverywhere, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    
    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);
    
    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;
    
    FunctionPtr u = Function::constant(0.5);
    
    bool useConformingTraces = true;
    SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
    
    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);
    
    if (!useTraceBCsEverywhere)
    {
      out << "useTraceBCsEverywhere = false not yet supported/implemented in test.\n";
      success = false;
    }
    else
    {
      VarPtr u_hat = form.u_hat();
      BCPtr bc = form.solution()->bc();
      SpatialFilterPtr allTime = SpatialFilter::matchingT(t0) | SpatialFilter::matchingT(t1);
      bc->addDirichlet(u_hat, SpatialFilter::allSpace() | allTime, u);
    }
    
    form.solution()->solve();
    
//    if (spaceDim != 3)
//    {
//      MeshPtr mesh = form.solution()->mesh();
//      string outputDir = "/tmp";
//      string solnName = (spaceDim == 1) ? "spaceTimeHeatSolution_1D" : "spaceTimeHeatSolution_2D";
//      cout << "\nDebugging: Outputting spaceTimeHeatSolution to " << outputDir << "/" << solnName << endl;
//      HDF5Exporter exporter(mesh, solnName, outputDir);
//      exporter.exportSolution(form.solution());
//    }
    
    double energyError = form.solution()->energyErrorTotal();
    
    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ConsistencyConstantSolution_1D )
  {
    // consistency test for space-time formulation with 1D space
    bool useHangingNodes = false;
    int spaceDim = 1;
    testSpaceTimeHeatConsistencyConstantSolution(spaceDim, useHangingNodes, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ConsistencyConstantSolution_2D )
  {
    // consistency test for space-time formulation with 2D space
    bool useHangingNodes = false;
    int spaceDim = 2;
    testSpaceTimeHeatConsistencyConstantSolution(spaceDim, useHangingNodes, out, success);
  }
  
//  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ConsistencyConstantSolution_3D )
//  {
//    // consistency test for space-time formulation with 3D space
//    bool useHangingNodes = false;
//    int spaceDim = 3;
//    testSpaceTimeHeatConsistencyConstantSolution(spaceDim, useHangingNodes, out, success);
//  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_Conforming_1D )
  {
    // consistency test for space-time formulation with 1D space
    bool useConformingTraces = true; // conforming and non conforming are the same for 1D
    bool useHangingNodes = false;
    testSpaceTimeHeatConsistency(1, useConformingTraces, useHangingNodes, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_Nonconforming_1D )
  {
    // consistency test for space-time formulation with 1D space
    bool useConformingTraces = false; // conforming and non conforming are the same for 1D
    bool useHangingNodes = false;
    testSpaceTimeHeatConsistency(1, useConformingTraces, useHangingNodes, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_Conforming_2D )
  {
    // consistency test for space-time formulation with 2D space
    bool useConformingTraces = true;
    bool useHangingNodes = false;
    testSpaceTimeHeatConsistency(2, useConformingTraces, useHangingNodes, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_Conforming_Irregular_2D )
  {
    // consistency test for space-time formulation with 2D space and hanging node
    bool useConformingTraces = true;
    bool useHangingNodes = true;
    testSpaceTimeHeatConsistency(2, useConformingTraces, useHangingNodes, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_Conforming_3D_Slow )
  {
    // consistency test for space-time formulation with 3D space
    bool useConformingTraces = true;
    bool useHangingNodes = false;
    testSpaceTimeHeatConsistency(3, useConformingTraces, useHangingNodes, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_Nonconforming_2D )
  {
    // consistency test for space-time formulation with 2D space
    bool useConformingTraces = false;
    bool useHangingNodes = false;
    testSpaceTimeHeatConsistency(2, useConformingTraces, useHangingNodes, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_Nonconforming_3D_Slow )
  {
    // consistency test for space-time formulation with 3D space
    bool useConformingTraces = false;
    bool useHangingNodes = false;
    testSpaceTimeHeatConsistency(3, useConformingTraces, useHangingNodes, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ForcingFunctionForConstantU_1D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(1, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ForcingFunctionForConstantU_2D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(2, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ForcingFunctionForConstantU_3D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(3, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ImposeConstantTraceBCs_1D )
  {
    // test BC imposition for space-time formulation with 1D space, exact solution with u constant
    testSpaceTimeHeatImposeConstantTraceBCs(1, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, SolveConstantSolution_1D )
  {
    // test solve for space-time formulation with 1D space, exact solution with u constant
    bool useTraceBCsEverywhere = true;
    testSpaceTimeHeatSolveConstantSolution(1, useTraceBCsEverywhere, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ImposeConstantTraceBCs_2D )
  {
    // test BC imposition for space-time formulation with 2D space, exact solution with u constant
    testSpaceTimeHeatImposeConstantTraceBCs(2, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, SolveConstantSolution_2D )
  {
    // test solve for space-time formulation with 1D space, exact solution with u constant
    bool useTraceBCsEverywhere = true;
    testSpaceTimeHeatSolveConstantSolution(2, useTraceBCsEverywhere, out, success);
  }
} // namespace
