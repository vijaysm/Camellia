//
//  NavierStokesCavityFlowAdaptive.cpp
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

#include "ParameterFunction.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "NavierStokesFormulation.h"

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
//#include "LidDrivenFlowRefinementStrategy.h"
#include "RefinementPattern.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "MeshPolyOrderFunction.h"
#include "MeshTestUtility.h"
#include "NonlinearSolveStrategy.h"
#include "PenaltyConstraints.h"

#include "MeshFactory.h"

#include "GnuPlotUtil.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

// static double REYN = 100;

VarFactory varFactory;
// test variables:
VarPtr tau1, tau2, v1, v2, q;
// traces and fluxes:
VarPtr u1hat, u2hat, t1n, t2n;
// field variables:
VarPtr u1, u2, sigma11, sigma12, sigma21, sigma22, p;


class U1_0 : public SimpleFunction
{
  double _eps;
public:
  U1_0(double eps)
  {
    _eps = eps;
  }
  double value(double x, double y)
  {
    double tol = 1e-14;
    if (abs(y-1.0) < tol)   // top boundary
    {
      if ( (abs(x) < _eps) )   // top left
      {
        return x / _eps;
      }
      else if ( abs(1.0-x) < _eps)     // top right
      {
        return (1.0-x) / _eps;
      }
      else     // top middle
      {
        return 1;
      }
    }
    else     // not top boundary: 0.0
    {
      return 0.0;
    }
  }
};

class U2_0 : public SimpleFunction
{
public:
  double value(double x, double y)
  {
    return 0.0;
  }
};

class Un_0 : public ScalarFunctionOfNormal
{
  SimpleFunctionPtr _u1, _u2;
public:
  Un_0(double eps)
  {
    _u1 = Teuchos::rcp(new U1_0(eps));
    _u2 = Teuchos::rcp(new U2_0);
  }
  double value(double x, double y, double n1, double n2)
  {
    double u1 = _u1->value(x,y);
    double u2 = _u2->value(x,y);
    return u1 * n1 + u2 * n2;
  }
};

class U0_cross_n : public ScalarFunctionOfNormal
{
  SimpleFunctionPtr _u1, _u2;
public:
  U0_cross_n(double eps)
  {
    _u1 = Teuchos::rcp(new U1_0(eps));
    _u2 = Teuchos::rcp(new U2_0);
  }
  double value(double x, double y, double n1, double n2)
  {
    double u1 = _u1->value(x,y);
    double u2 = _u2->value(x,y);
    return u1 * n2 - u2 * n1;
  }
};

class SqrtFunction : public Function
{
  FunctionPtr _f;
public:
  SqrtFunction(FunctionPtr f) : Function(0)
  {
    _f = f;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    CHECK_VALUES_RANK(values);
    _f->values(values,basisCache);

    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        double value = values(cellIndex,ptIndex);
        values(cellIndex,ptIndex) = sqrt(value);
      }
    }
  }
};

FieldContainer<double> pointGrid(double xMin, double xMax, double yMin, double yMax, int numPoints)
{
  vector<double> points1D_x, points1D_y;
  for (int i=0; i<numPoints; i++)
  {
    points1D_x.push_back( xMin + (xMax - xMin) * ((double) i) / (numPoints-1) );
    points1D_y.push_back( yMin + (yMax - yMin) * ((double) i) / (numPoints-1) );
  }
  int spaceDim = 2;
  FieldContainer<double> points(numPoints*numPoints,spaceDim);
  for (int i=0; i<numPoints; i++)
  {
    for (int j=0; j<numPoints; j++)
    {
      int pointIndex = i*numPoints + j;
      points(pointIndex,0) = points1D_x[i];
      points(pointIndex,1) = points1D_y[j];
    }
  }
  return points;
}

FieldContainer<double> solutionData(FieldContainer<double> &points, SolutionPtr solution, VarPtr u1)
{
  int numPoints = points.dimension(0);
  FieldContainer<double> values(numPoints);
  solution->solutionValues(values, u1->ID(), points);

  FieldContainer<double> xyzData(numPoints, 3);
  for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
  {
    xyzData(ptIndex,0) = points(ptIndex,0);
    xyzData(ptIndex,1) = points(ptIndex,1);
    xyzData(ptIndex,2) = values(ptIndex);
  }
  return xyzData;
}

set<double> diagonalContourLevels(FieldContainer<double> &pointData, int pointsPerLevel=1)
{
  // traverse diagonal of (i*numPoints + j) data from solutionData()
  int numPoints = sqrt(pointData.dimension(0));
  set<double> levels;
  for (int i=0; i<numPoints; i++)
  {
    levels.insert(pointData(i*numPoints + i,2)); // format for pointData has values at (ptIndex, 2)
  }
  // traverse the counter-diagonal
  for (int i=0; i<numPoints; i++)
  {
    levels.insert(pointData(i*numPoints + numPoints-1-i,2)); // format for pointData has values at (ptIndex, 2)
  }
  set<double> filteredLevels;
  int i=0;
  pointsPerLevel *= 2;
  for (set<double>::iterator levelIt = levels.begin(); levelIt != levels.end(); levelIt++)
  {
    if (i%pointsPerLevel==0)
    {
      filteredLevels.insert(*levelIt);
    }
    i++;
  }
  return filteredLevels;
}

void writePatchValues(double xMin, double xMax, double yMin, double yMax,
                      SolutionPtr solution, VarPtr u1, string filename,
                      int numPoints=100)
{
  FieldContainer<double> points = pointGrid(xMin,xMax,yMin,yMax,numPoints);
  FieldContainer<double> values(numPoints*numPoints);
  solution->solutionValues(values, u1->ID(), points);
  ofstream fout(filename.c_str());
  fout << setprecision(15);

  fout << "X = zeros(" << numPoints << ",1);\n";
  //    fout << "Y = zeros(numPoints);\n";
  fout << "U = zeros(" << numPoints << "," << numPoints << ");\n";
  for (int i=0; i<numPoints; i++)
  {
    fout << "X(" << i+1 << ")=" << points(i,0) << ";\n";
  }
  for (int i=0; i<numPoints; i++)
  {
    fout << "Y(" << i+1 << ")=" << points(i,1) << ";\n";
  }

  for (int i=0; i<numPoints; i++)
  {
    for (int j=0; j<numPoints; j++)
    {
      int pointIndex = i*numPoints + j;
      fout << "U("<<i+1<<","<<j+1<<")=" << values(pointIndex) << ";" << endl;
    }
  }
  fout.close();
}

int main(int argc, char *argv[])
{
  int rank = 0;
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();
#else
#endif
  bool useLineSearch = false;

  int pToAdd = 2; // for optimal test function approximation
  int pToAddForStreamFunction = 2;
  double nonlinearStepSize = 1.0;
  double dt = 0.5;
  double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  //  double nonlinearRelativeEnergyTolerance = 0.15; // used to determine convergence of the nonlinear solution
  double eps = 1.0/64.0; // width of ramp up to 1.0 for top BC;  eps == 0 ==> soln not in H1
  // epsilon above is chosen to match our initial 16x16 mesh, to avoid quadrature errors.
  //  double eps = 0.0; // John Evans's problem: not in H^1
  bool enforceLocalConservation = false;
  bool enforceOneIrregularity = true;
  bool reportPerCellErrors  = true;
  bool useMumps = true;

  int horizontalCells, verticalCells;

  int maxIters = 50; // for nonlinear steps

  vector<double> ReValues;

  // usage: polyOrder [numRefinements]
  // parse args:
  if (argc < 6)
  {
    cout << "Usage: NavierStokesCavityFlowContinuationFixedMesh fieldPolyOrder hCells vCells energyErrorGoal Re0 [Re1 ...]\n";
    return -1;
  }
  int polyOrder = atoi(argv[1]);
  horizontalCells = atoi(argv[2]);
  verticalCells = atoi(argv[3]);
  double energyErrorGoal = atof(argv[4]);
  for (int i=5; i<argc; i++)
  {
    ReValues.push_back(atof(argv[i]));
  }
  if (rank == 0)
  {
    cout << "L^2 order: " << polyOrder << endl;
    cout << "initial mesh size: " << horizontalCells << " x " << verticalCells << endl;
    cout << "energy error goal: " << energyErrorGoal << endl;
    cout << "Reynolds number values for continuation:\n";
    for (int i=0; i<ReValues.size(); i++)
    {
      cout << ReValues[i] << ", ";
    }
    cout << endl;
  }

  FieldContainer<double> quadPoints(4,2);

  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;

  // define meshes:
  int H1Order = polyOrder + 1;
  bool useTriangles = false;
  bool meshHasTriangles = useTriangles;

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

  FunctionPtr u1_0 = Teuchos::rcp( new U1_0(eps) );
  FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
  FunctionPtr zero = Function::zero();
  ParameterFunctionPtr Re_param = ParameterFunction::parameterFunction(1);
  VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re_param,quadPoints,
                                   horizontalCells,verticalCells,
                                   H1Order, pToAdd,
                                   u1_0, u2_0,  // BC for u
                                   zero, zero); // zero forcing function
  SolutionPtr solution = problem.backgroundFlow();
  SolutionPtr solnIncrement = problem.solutionIncrement();

  Teuchos::RCP<Mesh> mesh = problem.mesh();
  mesh->registerSolution(solution);
  mesh->registerSolution(solnIncrement);

  ///////////////////////////////////////////////////////////////////////////

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

  Teuchos::RCP<Mesh> streamMesh, overkillMesh;

  streamMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                          streamBF, H1Order+pToAddForStreamFunction,
                                          H1Order+pToAdd+pToAddForStreamFunction, useTriangles);

  mesh->registerObserver(streamMesh); // will refine streamMesh in the same way as mesh.

  map<int, double> dofsToL2error; // key: numGlobalDofs, value: total L2error compared with overkill
  vector< VarPtr > fields;
  fields.push_back(u1);
  fields.push_back(u2);
  fields.push_back(sigma11);
  fields.push_back(sigma12);
  fields.push_back(sigma21);
  fields.push_back(sigma22);
  fields.push_back(p);

  if (rank == 0)
  {
    cout << "Starting mesh has " << horizontalCells << " x " << verticalCells << " elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl;
    cout << "pToAdd = " << pToAdd << endl;
    cout << "eps for top BC = " << eps << endl;

    if (useTriangles)
    {
      cout << "Using triangles.\n";
    }
    if (enforceLocalConservation)
    {
      cout << "Enforcing local conservation.\n";
    }
    else
    {
      cout << "NOT enforcing local conservation.\n";
    }
    if (enforceOneIrregularity)
    {
      cout << "Enforcing 1-irregularity.\n";
    }
    else
    {
      cout << "NOT enforcing 1-irregularity.\n";
    }
  }

  ////////////////////   CREATE BCs   ///////////////////////
  SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered );

  FunctionPtr u1_prev = Function::solution(u1,solution);
  FunctionPtr u2_prev = Function::solution(u2,solution);

  FunctionPtr u1hat_prev = Function::solution(u1hat,solution);
  FunctionPtr u2hat_prev = Function::solution(u2hat,solution);


  ////////////////////   SOLVE & REFINE   ///////////////////////

  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - u1->dy() + u2->dx() ) );
  //  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution,sigma12 - sigma21) );
  RHSPtr streamRHS = RHS::rhs();
  streamRHS->addTerm(vorticity * q_s);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);
  ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(true);
  ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(true);

  BCPtr streamBC = BC::bc();
  //  streamBC->addDirichlet(psin_hat, entireBoundary, u0_cross_n);
  streamBC->addDirichlet(phi_hat, entireBoundary, zero);
  //  streamBC->addZeroMeanConstraint(phi);

  IPPtr streamIP = Teuchos::rcp( new IP );
  streamIP->addTerm(q_s);
  streamIP->addTerm(q_s->grad());
  streamIP->addTerm(v_s);
  streamIP->addTerm(v_s->div());
  SolutionPtr streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC, streamRHS, streamIP ) );

  if (enforceLocalConservation)
  {
    FunctionPtr zero = Function::zero();
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
    solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }

  if (true)
  {
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

    double energyThreshold = 0.20;
    Teuchos::RCP< RefinementStrategy > refinementStrategy = Teuchos::rcp( new RefinementStrategy( solnIncrement, energyThreshold ));

    for (int i=0; i<ReValues.size(); i++)
    {
      double Re = ReValues[i];
      Re_param->setValue(Re);
      if (rank==0) cout << "Solving with Re = " << Re << ":\n";
      double energyErrorTotal;
      do
      {
        double incr_norm;
        do
        {
          problem.iterate(useLineSearch);
          incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
          if (rank==0)
          {
            cout << "\x1B[2K"; // Erase the entire current line.
            cout << "\x1B[0E"; // Move to the beginning of the current line.
            cout << "Iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
            flush(cout);
          }
        }
        while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));
        if (rank==0) cout << endl;
        problem.setIterationCount(1); // 1 means reuse background flow (which we must, given that we want continuation in Re...)
        energyErrorTotal = solnIncrement->energyErrorTotal(); //solution->energyErrorTotal();
        if (energyErrorTotal > energyErrorGoal)
        {
          refinementStrategy->refine(false);
        }
        if (rank==0)
        {
          cout << "Energy error: " << energyErrorTotal << endl;
        }
      }
      while (energyErrorTotal > energyErrorGoal);
    }
  }

  double energyErrorTotal = solution->energyErrorTotal();
  double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
  if (rank == 0)
  {
    cout << "final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
    cout << "energy error: " << energyErrorTotal << endl;
    cout << "  (Incremental solution's energy error is " << incrementalEnergyErrorTotal << ".)\n";
  }

  FunctionPtr u1_sq = u1_prev * u1_prev;
  FunctionPtr u_dot_u = u1_sq + (u2_prev * u2_prev);
  FunctionPtr u_mag = Teuchos::rcp( new SqrtFunction( u_dot_u ) );
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

  CellTopoPtrLegacy quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
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
  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++)
  {
    ElementTypePtr elemType = *elemTypeIt;
    vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
    vector<GlobalIndexType> cellIDs;
    for (int i=0; i<elems.size(); i++)
    {
      cellIDs.push_back(elems[i]->cellID());
    }
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh,polyOrder) ); // enrich by trial space order
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
    FieldContainer<double> fakeRHSIntegrals(elems.size(),testOrdering->totalDofs());
    massFluxTerm->integrate(fakeRHSIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    //      cout << "fakeRHSIntegrals:\n" << fakeRHSIntegrals;
    for (int i=0; i<elems.size(); i++)
    {
      int cellID = cellIDs[i];
      // pick out the ones for testOne:
      massFluxIntegral[cellID] = fakeRHSIntegrals(i,testOneIndex);
    }
    // find the largest:
    for (int i=0; i<elems.size(); i++)
    {
      int cellID = cellIDs[i];
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
    }
    for (int i=0; i<elems.size(); i++)
    {
      int cellID = cellIDs[i];
      maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
      minCellMeasure = min(minCellMeasure,cellMeasures(i));
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
      totalMassFlux += massFluxIntegral[cellID];
      totalAbsMassFlux += abs( massFluxIntegral[cellID] );
    }
  }
  if (rank==0)
  {
    cout << "largest mass flux: " << maxMassFluxIntegral << endl;
    cout << "total mass flux: " << totalMassFlux << endl;
    cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;
    cout << "largest h: " << sqrt(maxCellMeasure) << endl;
    cout << "smallest h: " << sqrt(minCellMeasure) << endl;
    cout << "ratio of largest / smallest h: " << sqrt(maxCellMeasure) / sqrt(minCellMeasure) << endl;
  }
  if (rank == 0)
  {
    cout << "phi ID: " << phi->ID() << endl;
    cout << "psi1 ID: " << psi_1->ID() << endl;
    cout << "psi2 ID: " << psi_2->ID() << endl;

    cout << "streamMesh has " << streamMesh->numActiveElements() << " elements.\n";
    cout << "solving for approximate stream function...\n";
  }

  streamSolution->solve(useMumps);
  energyErrorTotal = streamSolution->energyErrorTotal();
  if (rank == 0)
  {
    cout << "...solved.\n";
    cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
  }

  if (rank==0)
  {
    FieldContainer<double> points = pointGrid(0, 1, 0, 1, 100);
    FieldContainer<double> pointData = solutionData(points, streamSolution, phi);
    GnuPlotUtil::writeXYPoints("phi_patch_navierStokes_cavity.dat", pointData);
    set<double> patchContourLevels = diagonalContourLevels(pointData,1);
    vector<string> patchDataPath;
    patchDataPath.push_back("phi_patch_navierStokes_cavity.dat");
    GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPath, "lidCavityNavierStokes.p");

    GnuPlotUtil::writeExactMeshSkeleton("lid_navierStokes_continuation_adaptive", mesh, 2);

    writePatchValues(0, 1, 0, 1, streamSolution, phi, "phi_patch.m");
    writePatchValues(0, .1, 0, .1, streamSolution, phi, "phi_patch_detail.m");
    writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
    writePatchValues(0, .001, 0, .001, streamSolution, phi, "phi_patch_minute_minute_detail.m");
  }

  return 0;
}
