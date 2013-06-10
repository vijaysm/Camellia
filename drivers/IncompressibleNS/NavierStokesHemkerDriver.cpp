//
//  NavierStokesCavityFlowDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "HConvergenceStudy.h"
#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"
#include <Teuchos_GlobalMPISession.hpp>

#include "NavierStokesFormulation.h"

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
//#include "LidDrivenFlowRefinementStrategy.h"
#include "RefinementPattern.h"
#include "RefinementHistory.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "MeshPolyOrderFunction.h"
#include "MeshTestUtility.h"
#include "NonlinearSolveStrategy.h"
#include "PenaltyConstraints.h"

#include "MeshFactory.h"

#include "SolutionExporter.h"

using namespace std;

static double Re = 5;

VarFactory varFactory; 
// test variables:
VarPtr tau1, tau2, v1, v2, q;
// traces and fluxes:
VarPtr u1hat, u2hat, t1n, t2n;
// field variables:
VarPtr u1, u2, sigma11, sigma12, sigma21, sigma22, p;

class LeftBoundary : public SpatialFilter {
  double _left;
public:
  LeftBoundary(double xLeft) {
    double tol = 1e-14;
    _left = xLeft + tol;
  }
  bool matchesPoint(double x, double y) {
    bool matches = x < _left;
//    cout << "Left boundary ";
//    if (matches) {
//      cout << "matches";
//    } else {
//      cout << "does not match";
//    }
//    cout << " point (" << x << ", " << y << ")\n";
    return matches;
  }
};

class RightBoundary : public SpatialFilter {
  double _right;
public:
  RightBoundary(double xRight) {
    double tol = 1e-14;
    _right = xRight - tol;
  }
  bool matchesPoint(double x, double y) {
    return x > _right;
  }
};


class TopBoundary : public SpatialFilter {
  double _top;
public:
  TopBoundary(double yTop) {
    double tol = 1e-14;
    _top = yTop - tol;
  }
  bool matchesPoint(double x, double y) {
    return y > _top;
  }
};

class BottomBoundary : public SpatialFilter {
  double _bottom;
public:
  BottomBoundary(double yBottom) {
    double tol = 1e-14;
    _bottom = yBottom + tol;
  }
  bool matchesPoint(double x, double y) {
    return y < _bottom;
  }
};

class NearCylinder : public SpatialFilter {
  double _enlarged_radius;
public:
  NearCylinder(double radius) {
    double enlargement_factor = 1.2;
    _enlarged_radius = radius * enlargement_factor;
  }
  bool matchesPoint(double x, double y) {
    if (x*x + y*y < _enlarged_radius * _enlarged_radius) {
      return true;
    } else {
      return false;
    }
  }
};

class BoundaryVelocity : public SimpleFunction {
  double _left, _right, _top, _bottom, _radius;
  int _comp;
public:
  BoundaryVelocity(double width, double height, double radius, int component) {
    double tol = 1e-14;
    _left = - width / 2.0 + tol;
    _right = width / 2.0 - tol;
    _top = height / 2.0 - tol;
    _bottom = - height / 2.0 + tol;
    _radius = radius;
    _comp = component; // 0 for x, 1 for y
  }
  
  double value(double x, double y) {
    // widen the radius to allow for some geometry error
    double enlarged_radius = _radius * 1.2;
    if (x*x + y*y < enlarged_radius * enlarged_radius) {
      // then we're on the cylinder
      return 0.0;
    }
    if (_comp == 0) { // u0 == 1 everywhere that we set u0
      return 1.0;
    } else {
      return 0.0;
    }
  }
};

FieldContainer<double> pointGrid(double xMin, double xMax, double yMin, double yMax, int numPoints) {
  vector<double> points1D_x, points1D_y;
  for (int i=0; i<numPoints; i++) {
    points1D_x.push_back( xMin + (xMax - xMin) * ((double) i) / (numPoints-1) );
    points1D_y.push_back( yMin + (yMax - yMin) * ((double) i) / (numPoints-1) );
  }
  int spaceDim = 2;
  FieldContainer<double> points(numPoints*numPoints,spaceDim);
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<numPoints; j++) {
      int pointIndex = i*numPoints + j;
      points(pointIndex,0) = points1D_x[i];
      points(pointIndex,1) = points1D_y[j];
    }
  }
  return points;
}

FieldContainer<double> solutionDataFromRefPoints(FieldContainer<double> &refPoints, SolutionPtr solution, VarPtr u) {
  int numPointsPerCell = refPoints.dimension(0);

  MeshPtr mesh = solution->mesh();
  int numCells = mesh->numActiveElements();
  int numPoints = numCells * numPointsPerCell;
  FieldContainer<double> xyzData(numPoints, 3);
  
  vector< ElementTypePtr > elementTypes = mesh->elementTypes(); // global element types list
  vector< ElementTypePtr >::iterator elemTypeIt;

  int globalPtIndex = 0;
  
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    //cout << "Solution: elementType loop, iteration: " << elemTypeNumber++ << endl;
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr, mesh));
    basisCache->setRefCellPoints(refPoints);
    
    vector<int> globalCellIDs = mesh->cellIDsOfTypeGlobal(elemTypePtr);
    
    FieldContainer<double> solutionValues(globalCellIDs.size(),numPointsPerCell);
    FieldContainer<double> physicalCellNodesForType = mesh->physicalCellNodesGlobal(elemTypePtr);

    basisCache->setPhysicalCellNodes(physicalCellNodesForType, globalCellIDs, false); // false: don't create side cache
    
    solution->solutionValues(solutionValues, u->ID(), basisCache);
    
    FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
    
    for (int cellIndex = 0; cellIndex < globalCellIDs.size(); cellIndex++) {
      for (int localPtIndex=0; localPtIndex<numPointsPerCell; localPtIndex++) {
        xyzData(globalPtIndex,0) = physicalPoints(cellIndex,localPtIndex,0);
        xyzData(globalPtIndex,1) = physicalPoints(cellIndex,localPtIndex,1);
        xyzData(globalPtIndex,2) = solutionValues(cellIndex,localPtIndex);
        globalPtIndex++;
      }
    }
  }
  
  return xyzData;
}

set<double> logContourLevels(double height, int numPointsTop=50) {
  set<double> levels;
  double level = height;
  for (int i=0; i<numPointsTop; i++) {
    levels.insert(level);
    levels.insert(-level);
    level /= 2.0;
  }
  return levels;
}

int main(int argc, char *argv[]) {
  int rank = 0;
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();

  bool useLineSearch = false;
  
  int pToAdd = 2; // for optimal test function approximation
  double dt = 0.5;
  bool enforceOneIrregularity = true;
  bool reportPerCellErrors  = true;

  bool startWithZeroSolutionAfterRefinement = false;
  
  bool artificialTimeStepping = false;
  
  bool useCondensedSolve = true;
  
  bool useScaleCompliantGraphNorm = false;
  bool enrichVelocity = useScaleCompliantGraphNorm;
  
  string solnSaveFile = "nsHemker.solution";
  string saveFile = "nsHemkerRefinements.replay";
  
  int maxIters = 50; // for nonlinear steps
  
  if (useScaleCompliantGraphNorm) {
    cout << "WARNING: useScaleCompliantGraphNorm = true, but support for this is not yet implemented in Hemker driver.\n";
  }
  
  // usage: polyOrder [numRefinements]
  // parse args:
  if ((argc != 4) && (argc != 3) && (argc != 2) && (argc != 5)) {
    cout << "Usage: NavierStokesHemkerDriver fieldPolyOrder [numRefinements=10 [Reyn=5]]\n";
    return -1;
  }
  int polyOrder = atoi(argv[1]);
  int numRefs = 10;
  if ( argc == 3) {
    numRefs = atoi(argv[2]);
  }
  if ( argc == 4) {
    numRefs = atoi(argv[2]);
    Re = atof(argv[3]);
  }
  if (rank == 0) {
    cout << "numRefinements = " << numRefs << endl;
    cout << "Re = " << Re << endl;
    if (artificialTimeStepping) cout << "dt = " << dt << endl;
    if (!startWithZeroSolutionAfterRefinement) {
      cout << "NOTE: experimentally, NOT starting with 0 solution after refinement...\n";
    }
    if (useCondensedSolve) {
      cout << "using condensed solve.\n";
    } else {
      cout << "not using condensed solve.\n";
    }
  }

  // define meshes:
  int H1Order = polyOrder + 1;
  double minL2Increment = 1e-8;

  // get variable definitions:
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  u1 = varFactory.fieldVar(VGP_U1_S);
  u2 = varFactory.fieldVar(VGP_U2_S);
  sigma11 = varFactory.fieldVar(VGP_SIGMA11_S);
  sigma12 = varFactory.fieldVar(VGP_SIGMA12_S);
  sigma21 = varFactory.fieldVar(VGP_SIGMA21_S);
  sigma22 = varFactory.fieldVar(VGP_SIGMA22_S);
  p = varFactory.fieldVar(VGP_P_S);
  
  u1hat = varFactory.traceVar(VGP_U1HAT_S);
  u2hat = varFactory.traceVar(VGP_U2HAT_S);
  t1n = varFactory.fluxVar(VGP_T1HAT_S);
  t2n = varFactory.fluxVar(VGP_T2HAT_S);
  
  v1 = varFactory.testVar(VGP_V1_S, HGRAD);
  v2 = varFactory.testVar(VGP_V2_S, HGRAD);
  tau1 = varFactory.testVar(VGP_TAU1_S, HDIV);
  tau2 = varFactory.testVar(VGP_TAU2_S, HDIV);
  q = varFactory.testVar(VGP_Q_S, HGRAD);
  
//  double width = 60, height = 20;
  double radius = 0.5;
  FunctionPtr zero = Function::zero();
  FunctionPtr inflowSpeed = Function::constant(1.0);
  
  double xLeft = -7.5;
  double xRight = 22.5;
  double meshHeight = 15;
  
  MeshGeometryPtr geometry = MeshFactory::shiftedHemkerGeometry(xLeft, xRight, meshHeight, radius); //MeshFactory::hemkerGeometry(width,height,radius);
  
  VGPNavierStokesProblem problem = VGPNavierStokesProblem(Function::constant(Re),geometry,
                                                          H1Order, pToAdd,
                                                          Function::zero(), Function::zero(), // zero forcing function
                                                          useScaleCompliantGraphNorm); // enrich velocity if using compliant graph norm
  SolutionPtr solution = problem.backgroundFlow();
  SolutionPtr solnIncrement = problem.solutionIncrement();
  
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr nearCylinder = Teuchos::rcp( new NearCylinder(radius) );
  SpatialFilterPtr top          = Teuchos::rcp( new TopBoundary(meshHeight/2.0) );
  SpatialFilterPtr bottom       = Teuchos::rcp( new BottomBoundary(-meshHeight/2.0) );
  SpatialFilterPtr left         = Teuchos::rcp( new LeftBoundary(xLeft) );
  SpatialFilterPtr right        = Teuchos::rcp( new RightBoundary(xRight) );
  // could simplify the below using SpatialFilter's Or-ing capability
  bc->addDirichlet(u1hat,nearCylinder,zero);
  bc->addDirichlet(u2hat,nearCylinder,zero);
  bc->addDirichlet(u1hat,left,inflowSpeed);
  bc->addDirichlet(u2hat,left,zero);
  bc->addDirichlet(u1hat,top,inflowSpeed);
  bc->addDirichlet(u2hat,top,zero);
  bc->addDirichlet(u1hat,bottom,inflowSpeed);
  bc->addDirichlet(u2hat,bottom,zero);
  // finally, no-traction conditions
  bc->addDirichlet(t1n,right,zero);
  bc->addDirichlet(t2n,right,zero);
  
  cout << "NOT imposing constraint on the pressure\n";
  // we used a problem constructor that neglects accumulated fluxes ==> we need to set BCs on each NR step
  problem.backgroundFlow()->setBC(bc);
  problem.solutionIncrement()->setBC(bc);
  
  Teuchos::RCP<Mesh> mesh = problem.mesh();
  mesh->registerSolution(solution);
  mesh->registerSolution(solnIncrement);
  
  Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
  mesh->registerObserver(refHistory);
  
  if (useScaleCompliantGraphNorm) {
    problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm());
  }
  
  // define bilinear form for stream function:
  VarFactory streamVarFactory;
  VarPtr phi_hat = streamVarFactory.traceVar("\\widehat{\\phi}");
  VarPtr psin_hat = streamVarFactory.fluxVar("\\widehat{\\psi}_n");
  VarPtr psi_1 = streamVarFactory.fieldVar("\\psi_1");
  VarPtr psi_2 = streamVarFactory.fieldVar("\\psi_2");
  VarPtr phi = streamVarFactory.fieldVar("\\phi");
  VarPtr q_s = streamVarFactory.testVar("q_s", HGRAD);
  VarPtr v_s = streamVarFactory.testVar("v_s", HDIV);
  BFPtr streamBF = Teuchos::rcp( new BF(streamVarFactory) );
  streamBF->addTerm(psi_1, q_s->dx());
  streamBF->addTerm(psi_2, q_s->dy());
  streamBF->addTerm(-psin_hat, q_s);
  
  streamBF->addTerm(psi_1, v_s->x());
  streamBF->addTerm(psi_2, v_s->y());
  streamBF->addTerm(phi, v_s->div());
  streamBF->addTerm(-phi_hat, v_s->dot_normal());
  
  Teuchos::RCP<Mesh> streamMesh;
  
  bool useConformingTraces = true;
  map<int, int> trialOrderEnhancements;
  if (enrichVelocity) {
    trialOrderEnhancements[u1->ID()] = 1;
    trialOrderEnhancements[u2->ID()] = 1;
  }
  streamMesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(),
                                 streamBF, H1Order, pToAdd,
                                 useConformingTraces, trialOrderEnhancements) );
  streamMesh->setEdgeToCurveMap(geometry->edgeToCurveMap());
  
  mesh->registerObserver(streamMesh); // will refine streamMesh in the same way as mesh.
  
//  bc->addZeroMeanConstraint(p);
  
  if (rank == 0) {
    cout << "Starting mesh has " << problem.mesh()->numElements() << " elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl; 
    cout << "pToAdd = " << pToAdd << endl;
    
    if (enforceOneIrregularity) {
      cout << "Enforcing 1-irregularity.\n";
    } else {
      cout << "NOT enforcing 1-irregularity.\n";
    }
  }
  
  map< int, FunctionPtr > initialGuess;
  initialGuess[u1->ID()] = Function::constant(1.0);
  initialGuess[u1hat->ID()] = Function::constant(1.0);
  // all other variables: use zero initial guess (the implicit one)
  
  ////////////////////   CREATE BCs   ///////////////////////
  FunctionPtr u1_prev = Function::solution(u1,solution);
  FunctionPtr u2_prev = Function::solution(u2,solution);
  
  FunctionPtr u1hat_prev = Function::solution(u1hat,solution);
  FunctionPtr u2hat_prev = Function::solution(u2hat,solution);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  
  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - u1->dy() + u2->dx() ) );

  FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
  
  double energyThreshold = 0.20; // for mesh refinements
  Teuchos::RCP<RefinementStrategy> refinementStrategy;
//  if (rank==0) cout << "NOTE: using solution, not solnIncrement, for refinement strategy.\n";
//  refinementStrategy = Teuchos::rcp( new RefinementStrategy( solution, energyThreshold ));
  refinementStrategy = Teuchos::rcp( new RefinementStrategy( solnIncrement, energyThreshold ));
  
  refinementStrategy->setEnforceOneIrregularity(enforceOneIrregularity);
  refinementStrategy->setReportPerCellErrors(reportPerCellErrors);
  
  if (true) { // do regular refinement strategy...
    bool printToConsole = rank==0;
    FunctionPtr u1_incr = Function::solution(u1, solnIncrement);
    FunctionPtr u2_incr = Function::solution(u2, solnIncrement);
    FunctionPtr sigma11_incr = Function::solution(sigma11, solnIncrement);
    FunctionPtr sigma12_incr = Function::solution(sigma12, solnIncrement);
    FunctionPtr sigma21_incr = Function::solution(sigma21, solnIncrement);
    FunctionPtr sigma22_incr = Function::solution(sigma22, solnIncrement);
    FunctionPtr p_incr = Function::solution(p, solnIncrement);
    
    FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
    + sigma11_incr * sigma11_incr + sigma12_incr * sigma12_incr
    + sigma21_incr * sigma21_incr + sigma22_incr * sigma22_incr;

    for (int refIndex=0; refIndex<numRefs; refIndex++){
      if (startWithZeroSolutionAfterRefinement) {
        // start with a fresh initial guess for each adaptive mesh:
        solution->clear();
        cout << "using zero initial guess for now...\n";
//        solution->projectOntoMesh(initialGuess);
        problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement)
      }
      
      double incr_norm;
      do {
        problem.iterate(useLineSearch,useCondensedSolve);
        incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
        if (rank==0) {
          cout << "\x1B[2K"; // Erase the entire current line.
          cout << "\x1B[0E"; // Move to the beginning of the current line.
          cout << "Refinement # " << refIndex << ", iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
          flush(cout);
        }
      } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));

      if (rank==0) {
        cout << "\nFor refinement " << refIndex << ", num iterations: " << problem.iterationCount() << endl;
      }
      
      // reset iteration count to 1 (for the background flow):
      problem.setIterationCount(1);
      
      if (rank==0) {
        if (solnSaveFile.length() > 0) {
          solution->writeToFile(solnSaveFile);
        }
      }
      
      refinementStrategy->refine(false); // don't print to console // (rank==0); // print to console on rank 0
      if (rank==0) {
        cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
      }
      
      if (saveFile.length() > 0) {
        if (rank == 0) {
          refHistory->saveToFile(saveFile);
        }
      }
      
    }
    // one more solve on the final refined mesh:
    if (rank==0) cout << "Final solve:\n";
    if (startWithZeroSolutionAfterRefinement) {
      // start with a fresh (zero) initial guess for each adaptive mesh:
      solution->clear();
      problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement)
    }
    double incr_norm;
    do {
      problem.iterate(useLineSearch,useCondensedSolve);
      incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
      if (rank==0) {
        cout << "\x1B[2K"; // Erase the entire current line.
        cout << "\x1B[0E"; // Move to the beginning of the current line.
        cout << "Iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
        flush(cout);
      }
    } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));
    if (rank==0) cout << endl;
  }

  if (rank==0) {
    if (solnSaveFile.length() > 0) {
      solution->writeToFile(solnSaveFile);
    }
  }
  
  double energyErrorTotal = solution->energyErrorTotal();
  double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
  if (rank == 0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
    cout << "Final incremental energy error: " << incrementalEnergyErrorTotal << ".)\n";
  }
  
  FunctionPtr u1_sq = u1_prev * u1_prev;
  FunctionPtr u_dot_u = u1_sq + (u2_prev * u2_prev);
  FunctionPtr u_div = Teuchos::rcp( new PreviousSolutionFunction(solution, u1->dx() + u2->dy() ) );
  FunctionPtr massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat->times_normal_x() + u2hat->times_normal_y()) );
  
  // check that the zero mean pressure is being correctly imposed:
  FunctionPtr p_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,p) );
  double p_avg = p_prev->integrate(mesh);
  if (rank==0)
    cout << "Integral of pressure: " << p_avg << endl;
  
  // integrate massFlux over each element (a test):
  // fake a new bilinear form so we can integrate against 1 
  VarPtr testOne = varFactory.testVar("1",CONSTANT_SCALAR);
  BFPtr fakeBF = Teuchos::rcp( new BF(varFactory) );
  LinearTermPtr massFluxTerm = massFlux * testOne;
  
  CellTopoPtr quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  DofOrderingFactory dofOrderingFactory(fakeBF);
  int fakeTestOrder = H1Order;
  DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(fakeTestOrder, *quadTopoPtr);
  
  int testOneIndex = testOrdering->getDofIndex(testOne->ID(),0);
  vector< ElementTypePtr > elemTypes = mesh->elementTypes(); // global element types
  map<int, double> massFluxIntegral; // cellID -> integral
  double maxMassFluxIntegral = 0.0;
  double totalMassFlux = 0.0;
  double totalAbsMassFlux = 0.0;
  double maxCellMeasure = 0;
  double minCellMeasure = 1;
  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemType = *elemTypeIt;
    vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
    vector<int> cellIDs;
    for (int i=0; i<elems.size(); i++) {
      cellIDs.push_back(elems[i]->cellID());
    }
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh,polyOrder) ); // enrich by trial space order
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
    FieldContainer<double> fakeRHSIntegrals(elems.size(),testOrdering->totalDofs());
    massFluxTerm->integrate(fakeRHSIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      // pick out the ones for testOne:
      massFluxIntegral[cellID] = fakeRHSIntegrals(i,testOneIndex);
    }
    // find the largest:
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
    }
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
      minCellMeasure = min(minCellMeasure,cellMeasures(i));
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
      totalMassFlux += massFluxIntegral[cellID];
      totalAbsMassFlux += abs( massFluxIntegral[cellID] );
    }
  }
  if (rank==0) {
    cout << "largest mass flux: " << maxMassFluxIntegral << endl;
    cout << "total mass flux: " << totalMassFlux << endl;
    cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;
    cout << "largest h: " << sqrt(maxCellMeasure) << endl;
    cout << "smallest h: " << sqrt(minCellMeasure) << endl;
    cout << "ratio of largest / smallest h: " << sqrt(maxCellMeasure) / sqrt(minCellMeasure) << endl;
  }
  
  if (rank==0){
    GnuPlotUtil::writeComputationalMeshSkeleton("finalHemkerMesh", mesh);
    
    VTKExporter exporter(solution, mesh, varFactory);
    exporter.exportSolution("nsHemkerSoln", H1Order*2);
    
    exporter.exportFunction(vorticity, "HemkerVorticity");
    
//    massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
//    u_div->writeValuesToMATLABFile(solution->mesh(), "u_div.m");
//    solution->writeFieldsToFile(u1->ID(), "u1.m");
//    solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
//    solution->writeFieldsToFile(u2->ID(), "u2.m");
//    solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
//    solution->writeFieldsToFile(p->ID(), "p.m");
    
//    polyOrderFunction->writeValuesToMATLABFile(mesh, "hemkerPolyOrders.m");
  }
  
  Teuchos::RCP<RHSEasy> streamRHS = Teuchos::rcp( new RHSEasy );
  streamRHS->addTerm(vorticity * q_s);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);
  ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(true);
  ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(true);
  
  Teuchos::RCP<BCEasy> streamBC = Teuchos::rcp( new BCEasy );
  // wherever we enforce velocity BCs, enforce BCs on phi, too
  // phi, the streamfunction, can be used to measure mass flux between two points
  // reverse engineering that fact, we can use y as the BC for phi
  FunctionPtr y = Function::yn();
  // experiment: try no BC around cylinder...
//  streamBC->addDirichlet(phi_hat, nearCylinder, y);
  streamBC->addDirichlet(phi_hat, left, y);
  streamBC->addDirichlet(phi_hat, top, y);
  streamBC->addDirichlet(phi_hat, bottom, y);
  
  IPPtr streamIP = Teuchos::rcp( new IP );
  streamIP->addTerm(q_s);
  streamIP->addTerm(q_s->grad());
  streamIP->addTerm(v_s);
  streamIP->addTerm(v_s->div());
  SolutionPtr streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC, streamRHS, streamIP ) );
  
  cout << "streamMesh has " << streamMesh->numActiveElements() << " elements.\n";
  cout << "solving for approximate stream function...\n";

  streamSolution->solve();
  energyErrorTotal = streamSolution->energyErrorTotal();
  if (rank == 0) {
    cout << "...solved.\n";
    cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
  }
  
  if (rank==0){
    VTKExporter streamExporter(streamSolution, streamMesh, streamVarFactory);
    streamExporter.exportSolution("hemkerStreamSoln", H1Order*2);

    // the commented-out code below doesn't really work because gnuplot requires a "point grid" in physical space...
//    FieldContainer<double> refPoints = pointGrid(-1, 1, -1, 1, H1Order);
//    FieldContainer<double> pointData = solutionDataFromRefPoints(refPoints, streamSolution, phi);
//    string patchDataPath = "phi_navierStokes_hemker.dat";
//    GnuPlotUtil::writeXYPoints(patchDataPath, pointData);
//    set<double> patchContourLevels = logContourLevels(height);
//    vector<string> patchDataPaths;
//    patchDataPaths.push_back(patchDataPath);
//    GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPaths, "hemkerNavierStokes.p");
  }
  
  return 0;
}
