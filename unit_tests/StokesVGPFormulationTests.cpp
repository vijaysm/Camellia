//
//  StokesVGPFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/5/15.
//
//

#include "EpetraExt_RowMatrixOut.h"
#include "Teuchos_UnitTestHarness.hpp"

#include "StokesVGPFormulation.h"

#include "GDAMinimumRule.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "Solution.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
void projectExactSolution(StokesVGPFormulation &form, SolutionPtr stokesSolution, FunctionPtr u, FunctionPtr p)
{
  double mu = form.mu();

  int spaceDim = form.spaceDim();

  LinearTermPtr t1_n_lt, t2_n_lt, t3_n_lt;
  t1_n_lt = form.tn_hat(1)->termTraced();
  t2_n_lt = form.tn_hat(2)->termTraced();
  if (spaceDim==3)
  {
    t3_n_lt = form.tn_hat(3)->termTraced();
  }

  map<int, FunctionPtr> exactMap;
  
  // fields:
  exactMap[form.p()->ID() ] =  p;
  for (int comp_i=1; comp_i<=spaceDim; comp_i++)
  {
    FunctionPtr ui = u->spatialComponent(comp_i);
    exactMap[form.u(comp_i)->ID()] = ui;
  
    for (int comp_j=1; comp_j<=spaceDim; comp_j++)
    {
      exactMap[form.sigma(comp_i,comp_j)->ID()] = mu * ui->grad()->spatialComponent(comp_j);
    }
  }
  
  // fluxes and traces:
  // use the exact field variable solution together with the termTraced to determine the flux traced

  for (int comp_i=1; comp_i<=spaceDim; comp_i++)
  {
    // fluxes:
    LinearTermPtr tn_i_lt = form.tn_hat(comp_i)->termTraced();
    FunctionPtr tn_i = tn_i_lt->evaluate(exactMap);
    exactMap[form.tn_hat(comp_i)->ID()] = tn_i;
    
    // traces:
    LinearTermPtr ui_lt = form.u_hat(comp_i)->termTraced();
    FunctionPtr ui = ui_lt->evaluate(exactMap);
    exactMap[form.u_hat(comp_i)->ID()] = ui;
  }
  
  stokesSolution->projectOntoMesh(exactMap);
}

void setupExactSolution(StokesVGPFormulation &form, FunctionPtr u, FunctionPtr p,
                        MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k)
{
  FunctionPtr forcingFunction = form.forcingFunction(u, p);

  form.initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction);

  if (form.isSpaceTime())
  {
    form.addPointPressureCondition();
  }
  else
  {
    form.addZeroMeanPressureCondition();
  }
  form.addInflowCondition(SpatialFilter::allSpace(), u);
}

void testStokesConsistencySteady(int spaceDim, Teuchos::FancyOStream &out, bool &success)
{
  vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
  vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
  vector<double> x0(spaceDim,-1.0);
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
  double Re = 1.0;
  int fieldPolyOrder = 2, delta_k = 1;

  FunctionPtr u, p;
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  if (spaceDim == 2)
  {
    FunctionPtr u1 = x;
    FunctionPtr u2 = -y; // divergence 0
    u = Function::vectorize(u1,u2);
    p = x + 2. * y; // zero average
  }
  else if (spaceDim == 3)
  {
    FunctionPtr u1 = 2. * x;
    FunctionPtr u2 = -y; // divergence 0
    FunctionPtr u3 = -z;
    u = Function::vectorize(u1,u2,u3);
    p = x + 2. * y + 3. * z; // zero average
  }

  bool useConformingTraces = true;
  double mu = 1.0 / Re;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim,mu,useConformingTraces);
  
  setupExactSolution(form, u, p, meshTopo, fieldPolyOrder, delta_k);
  projectExactSolution(form, form.solution(), u, p);

  FunctionPtr pSoln = Function::solution(form.p(), form.solution());

  form.solution()->clearComputedResiduals();

  double energyError = form.solution()->energyErrorTotal();

  double tol = 1e-13;
  TEST_COMPARE(energyError, <, tol);
}

TEUCHOS_UNIT_TEST( StokesVGPFormulation, Consistency_2D_Steady )
{
  int spaceDim = 2;
  testStokesConsistencySteady(spaceDim,out,success);
}

TEUCHOS_UNIT_TEST( StokesVGPFormulation, Consistency_3D_Steady_Slow )
{
  int spaceDim = 3;
  testStokesConsistencySteady(spaceDim,out,success);
}

TEUCHOS_UNIT_TEST( StokesVGPFormulation, StreamFormulationConsistency )
{
  /*
    The stream formulation's psi function should be (-u2, u1).  Here, we project that solution
    onto the stream formulation, and test that the residual is 0.
   */
  int spaceDim = 2;
  vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
  vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
  vector<double> x0(spaceDim,-1.0);
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
  double Re = 1.0;
  int fieldPolyOrder = 3, delta_k = 1;

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr u1 = x;
  FunctionPtr u2 = -y; // divergence 0
  FunctionPtr u = Function::vectorize(u1,u2);
  FunctionPtr p = y * y * y; // zero average

  bool useConformingTraces = true;
  double mu = 1.0 / Re;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim,mu,useConformingTraces);

  
  setupExactSolution(form, u, p, meshTopo, fieldPolyOrder, delta_k);

  SolutionPtr streamSoln = form.streamSolution();

  // to determine phi_exact, we solve the problem:
  //   d/dx phi = -u2
  //   d/dy phi = +u1
  // subject to the constraint that its integral on the domain is zero.
  // Here, phi = xy solves it.

  FunctionPtr phi_exact = x * y;
  FunctionPtr psi_exact = Function::vectorize(-u2, u1);

  map<int, FunctionPtr> exactMap;
  // fields:
  exactMap[form.streamFormulation().phi()->ID()] = phi_exact;
  exactMap[form.streamFormulation().psi()->ID()] = psi_exact;

  VarPtr phi_hat = form.streamFormulation().phi_hat();
  VarPtr psi_n_hat = form.streamFormulation().psi_n_hat();

  // traces and fluxes:
  // use the exact field variable solution together with the termTraced to determine the flux traced
  FunctionPtr phi_hat_exact = phi_hat->termTraced()->evaluate(exactMap);
  FunctionPtr psi_n_hat_exact = psi_n_hat->termTraced()->evaluate(exactMap);
  exactMap[phi_hat->ID()] = phi_hat_exact;
  exactMap[psi_n_hat->ID()] = psi_n_hat_exact;

  streamSoln->projectOntoMesh(exactMap);

  double energyError = streamSoln->energyErrorTotal();

  double tol = 1e-13;
  TEST_COMPARE(energyError, <, tol);
}

  TEUCHOS_UNIT_TEST( StokesVGPFormulation, Consistency_2D_SpaceTime_Slow )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double t0 = 0.0, t1 = 0.1;
    int numTimeElements = 1;
    MeshTopologyPtr meshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1, numTimeElements);
    double Re = 1.0;
    int fieldPolyOrder = 3, delta_k = 1;
    
    // testing space-time formulation consistency goes much as with the steady state;
    // if we project a steady solution onto the space-time mesh, we should have a zero residual
    // (would also be worth checking that an exactly-recoverable transient solution has zero residual)
    
    bool useConformingTraces = true;
    double mu = 1.0 / Re;
    StokesVGPFormulation form = StokesVGPFormulation::spaceTimeFormulation(spaceDim, mu, useConformingTraces);
    
    vector<pair<FunctionPtr, FunctionPtr>> exactSolutions; // (u,p) pairs
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr t = Function::tn(1);
    
//    FunctionPtr u1 = x;
//    FunctionPtr u2 = -y; // divergence 0
//    FunctionPtr u = Function::vectorize(u1,u2);
//    FunctionPtr p = Function::zero(); // y * y * y; // zero average
//    exactSolutions.push_back({u,p});
//    
//    u1 = 2 * x * y;
//    u2 = -y * y; // divergence 0
//    u = Function::vectorize(u1,u2);
//    p = Function::zero(); // zero average
//    exactSolutions.push_back({u,p});
    
    FunctionPtr u1 = x * t;
    FunctionPtr u2 = -y * t; // divergence 0
    FunctionPtr u = Function::vectorize(u1,u2);
    FunctionPtr p = (y * y * y + 1.0) * t; // zero at (-1,-1), which is where the point constraint happens to be imposed...
    exactSolutions.push_back({u,p});
    
    for (auto exactSolution : exactSolutions)
    {
      u = exactSolution.first;
      p = exactSolution.second;

      setupExactSolution(form, u, p, meshTopo, fieldPolyOrder, delta_k);
      projectExactSolution(form, form.solution(), u, p);
//      form.addPointPressureCondition();
      
      double energyError = form.solution()->energyErrorTotal();
      
      double tol = 1e-13;
      TEST_COMPARE(energyError, <, tol);
      
//      {         // DEBUGGING:
//        form.bf()->printTrialTestInteractions();
//        FunctionPtr f = form.forcingFunction(u, p);
//        cout << "forcing function: " << f->displayString() << endl;
//        HDF5Exporter exporter(form.solution()->mesh(),"StokesSpaceTimeForcingFunction","/tmp");
//        FunctionPtr f_padded = Function::vectorize(f->x(), f->y(), Function::zero());
//        exporter.exportFunction(f_padded, "forcing function", 0.0, 5);
//        
//        HDF5Exporter solutionExporter(form.solution()->mesh(),"StokesSpaceTimeSolution","/tmp");
//        // export the projected solution at "time" 0
//        solutionExporter.exportSolution(form.solution(), 0.0, 10);
//        
//        // solve, and export the solution at "time" 1
//        form.solve();
//        solutionExporter.exportSolution(form.solution(), 1.0, 10);
//        cout << "Exported solution.\n";
//      }
    }
  }
  
  TEUCHOS_UNIT_TEST( StokesVGPFormulation, Consistency_2D_TimeStepping )
{
  int spaceDim = 2;
  vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
  vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
  vector<double> x0(spaceDim,-1.0);
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
  double Re = 1.0;
  int fieldPolyOrder = 3, delta_k = 1;

  // testing transient formulation consistency goes much as with the steady state;
  // if we project a steady solution onto the previous solution as well as the current solution,
  // we should have a zero residual

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr u1 = x;
  FunctionPtr u2 = -y; // divergence 0
  FunctionPtr u = Function::vectorize(u1,u2);
  FunctionPtr p = y * y * y; // zero average

  bool useConformingTraces = true;
  double mu = 1.0 / Re;
  double dt = 1.0;
  StokesVGPFormulation form = StokesVGPFormulation::timeSteppingFormulation(spaceDim, mu, dt, useConformingTraces, BACKWARD_EULER);
  
  setupExactSolution(form, u, p, meshTopo, fieldPolyOrder, delta_k);
  projectExactSolution(form, form.solution(), u, p);
  projectExactSolution(form, form.solutionPreviousTimeStep(), u, p);

  double energyError = form.solution()->energyErrorTotal();

  double tol = 1e-13;
  TEST_COMPARE(energyError, <, tol);
}

TEUCHOS_UNIT_TEST( StokesVGPFormulation, ForcingFunction_2D)
{
  double Re = 10.0;

  int spaceDim = 2;
  vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
  vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
  vector<double> x0(spaceDim,-1.0);
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

  bool useConformingTraces = true;
  double mu = 1.0 / Re;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, useConformingTraces);
  
  int fieldPolyOrder = 1;
  int delta_k = 1;
  MeshPtr stokesMesh = Teuchos::rcp( new Mesh(meshTopo,form.bf(),fieldPolyOrder+1, delta_k) );

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr u1 = x;
  FunctionPtr u2 = -y; // divergence 0
  //    FunctionPtr p = y * y * y; // zero average
  //    FunctionPtr u1 = Function::constant(1.0);
  //    FunctionPtr u2 = Function::constant(1.0);
  FunctionPtr p = x + y;

  FunctionPtr forcingFunction_x = p->dx() - (1.0/Re) * (u1->dx()->dx() + u1->dy()->dy());
  FunctionPtr forcingFunction_y = p->dy() - (1.0/Re) * (u2->dx()->dx() + u2->dy()->dy());
  FunctionPtr forcingFunctionExpected = Function::vectorize(forcingFunction_x, forcingFunction_y);

  FunctionPtr forcingFunctionActual = form.forcingFunction(Function::vectorize(u1, u2), p);

  double tol = 1e-13;
  double err = (forcingFunctionExpected - forcingFunctionActual)->l2norm(stokesMesh);
  TEST_COMPARE(err, <, tol);
}

  TEUCHOS_UNIT_TEST( StokesVGPFormulation, ForcingFunction_2D_SpaceTime )
  {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr meshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);
    double Re = 1.0;
    int fieldPolyOrder = 3, delta_k = 1;
    
    bool useConformingTraces = true;
    double mu = 1.0 / Re;
    StokesVGPFormulation form = StokesVGPFormulation::spaceTimeFormulation(spaceDim, mu, useConformingTraces);
    
    // testing space-time formulation consistency goes much as with the steady state;
    // if we project a steady solution onto the space-time mesh, we should have a zero residual
    // We also check that an exactly-recoverable transient solution has zero residual.
    
    vector<pair<FunctionPtr, FunctionPtr>> exactSolutions; // (u,p) pairs
    vector<FunctionPtr> analyticForcingFunctions; // hand-computed
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr u1 = x;
    FunctionPtr u2 = -y; // divergence 0
    FunctionPtr u = Function::vectorize(u1,u2);
    FunctionPtr p = y * y * y; // zero average
    FunctionPtr f_x = Function::zero();
    FunctionPtr f_y = 3 * y * y; // p->dy()
    
    analyticForcingFunctions.push_back(Function::vectorize(f_x, f_y));
    exactSolutions.push_back({u,p});
    
    FunctionPtr t = Function::tn(1);
    u1 = x * t;
    u2 = -y * t; // divergence 0
    u = Function::vectorize(u1,u2);
    p = y * y * y * t; // zero average
    f_x = x;                 // p->dx() + u1->dt()
    f_y = 3 * y * y * t - y; // p->dy() + u2->dt()
    
    analyticForcingFunctions.push_back(Function::vectorize(f_x, f_y));
    exactSolutions.push_back({u,p});
    
    u1 = Function::zero();
    u2 = -t; // divergence 0
    u = Function::vectorize(u1,u2);
    p = Function::zero(); // zero average
    f_x = Function::zero();  // u1->dt()
    f_y = Function::constant(-1.0); // u2->dt()
    
    analyticForcingFunctions.push_back(Function::vectorize(f_x, f_y));
    exactSolutions.push_back({u,p});
    
    for (int i=0; i<exactSolutions.size(); i++)
    {
      auto exactSolution = exactSolutions[i];
      auto f_analytic = analyticForcingFunctions[i];
      u = exactSolution.first;
      p = exactSolution.second;
      setupExactSolution(form, u, p, meshTopo, fieldPolyOrder, delta_k);
      
      FunctionPtr f_actual = form.forcingFunction(u, p);
      FunctionPtr f_expected_x = p->dx() - mu * (u->x()->dx()->dx() + u->x()->dy()->dy()) + u->x()->dt();
      FunctionPtr f_expected_y = p->dy() - mu * (u->y()->dx()->dx() + u->y()->dy()->dy()) + u->y()->dt();
      
      MeshPtr mesh = form.solution()->mesh();
      
      double tol = 1e-14;
      double diff_x = (f_expected_x - f_actual->x())->l2norm(mesh);
      double diff_y = (f_expected_y - f_actual->y())->l2norm(mesh);
      TEST_COMPARE(diff_x, <, tol);
      TEST_COMPARE(diff_y, <, tol);
      
      double diff_x_analytic = (f_expected_x - f_analytic->x())->l2norm(mesh);
      double diff_y_analytic = (f_expected_y - f_analytic->y())->l2norm(mesh);
      TEST_COMPARE(diff_x_analytic, <, tol);
      TEST_COMPARE(diff_y_analytic, <, tol);
    }
  }

  
TEUCHOS_UNIT_TEST( StokesVGPFormulation, Projection_2D_Slow )
{
  int spaceDim = 2;
  vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
  vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
  vector<double> x0(spaceDim,-1.0);
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
  double Re = 1.0;
  int fieldPolyOrder = 1, delta_k = 1;

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr u1 = x;
  FunctionPtr u2 = -y; // divergence 0
  FunctionPtr u = Function::vectorize(u1,u2);

//    FunctionPtr p = y * y * y; // zero average
//    FunctionPtr u1 = Function::constant(1.0);
//    FunctionPtr u2 = Function::constant(1.0);
  FunctionPtr p = x + y;

  bool useConformingTraces = true;
  double mu = 1.0 / Re;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, useConformingTraces);
  setupExactSolution(form,u,p,meshTopo,fieldPolyOrder,delta_k);

  MeshPtr stokesMesh = form.solution()->mesh();

  BFPtr stokesBF = form.bf();

  //uniform h-refinement:
  stokesMesh->hRefine(stokesMesh->getActiveCellIDsGlobal());

  form.solve();

  SolutionPtr stokesProjection = Solution::solution(stokesMesh);
  projectExactSolution(form, stokesProjection, u, p);

  SolutionPtr stokesSolution = form.solution();
  stokesSolution->addSolution(stokesProjection, -1);

  FunctionPtr u1_diff = Function::solution(form.u(1), stokesSolution);
  FunctionPtr u2_diff = Function::solution(form.u(2), stokesSolution);
  FunctionPtr sigma11_diff = Function::solution(form.sigma(1,1), stokesSolution);
  FunctionPtr sigma12_diff = Function::solution(form.sigma(1,2), stokesSolution);
  FunctionPtr sigma21_diff = Function::solution(form.sigma(2,1), stokesSolution);
  FunctionPtr sigma22_diff = Function::solution(form.sigma(2,2), stokesSolution);
  FunctionPtr p_diff = Function::solution(form.p(), stokesSolution);

  double p_diff_l2 = p_diff->l2norm(stokesMesh);
  double u1_diff_l2 = u1_diff->l2norm(stokesMesh);
  double u2_diff_l2 = u2_diff->l2norm(stokesMesh);
  double sigma11_diff_l2 = sigma11_diff->l2norm(stokesMesh);
  double sigma12_diff_l2 = sigma12_diff->l2norm(stokesMesh);
  double sigma21_diff_l2 = sigma21_diff->l2norm(stokesMesh);
  double sigma22_diff_l2 = sigma22_diff->l2norm(stokesMesh);

  double tol = 1e-13;
  TEST_COMPARE(p_diff_l2, <, tol);
  TEST_COMPARE(u1_diff_l2, <, tol);
  TEST_COMPARE(u2_diff_l2, <, tol);
  TEST_COMPARE(sigma11_diff_l2, <, tol);
  TEST_COMPARE(sigma12_diff_l2, <, tol);
  TEST_COMPARE(sigma21_diff_l2, <, tol);
  TEST_COMPARE(sigma22_diff_l2, <, tol);
}
  
  TEUCHOS_UNIT_TEST( StokesVGPFormulation, RefineMesh_Slow )
  {
    MPIWrapper::CommWorld()->Barrier();
    
    int spaceDim = 2;
    bool useConformingTraces = true;
    int meshWidth = 2;
    int polyOrder = 2;
    int delta_k = 2;
    double mu = 1.0;
    
    StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, useConformingTraces);
    
    vector<double> dims(spaceDim,1.0);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,0.0);
    
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    form.initializeSolution(meshTopo, polyOrder, delta_k);
    form.addPointPressureCondition();
    
    MeshPtr mesh = form.solution()->mesh();
    
    VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);
    VarPtr u1 = form.u(1), u2 = form.u(2);
    VarPtr tn1_hat = form.tn_hat(1), tn2_hat = form.tn_hat(2);
    VarPtr sigma12 = form.sigma(1, 2);
    
    TFunctionPtr<double> n = TFunction<double>::normal();
    TFunctionPtr<double> n_parity = n * TFunction<double>::sideParity();
    FunctionPtr minus_n_parity = - n_parity;
    FunctionPtr sigma12_exact = Function::constant(-1.0);
    
    map<int, FunctionPtr> solnToProject;
    FunctionPtr u1_exact = Function::yn(1);
    FunctionPtr u2_exact = Function::constant(2.0);

    solnToProject[u1->ID()] = u1_exact;
    solnToProject[u2->ID()] = u2_exact;
    solnToProject[sigma12->ID()] = sigma12_exact;
    
    // set up fluxes and traces to agree with the fields, according to termTraced():
    solnToProject[u1_hat->ID()] = u1_hat->termTraced()->evaluate(solnToProject);
    solnToProject[u2_hat->ID()] = u2_hat->termTraced()->evaluate(solnToProject);
    solnToProject[tn1_hat->ID()] = tn1_hat->termTraced()->evaluate(solnToProject);
    
    FunctionPtr tn1_exact = solnToProject[tn1_hat->ID()];
    
    form.solution()->projectOntoMesh(solnToProject);
    
    // sanity check: did the projection work?
    FunctionPtr u1_soln = Function::solution(u1, form.solution());
    FunctionPtr u2_soln = Function::solution(u2, form.solution());
    FunctionPtr u1_hat_soln = Function::solution(u1_hat, form.solution());
    FunctionPtr u2_hat_soln = Function::solution(u2_hat, form.solution());
    bool weightByParity = false; // don't weight by parity so that we know that the expected value is tn_exact everywhere
    FunctionPtr tn1_hat_soln = Function::solution(tn1_hat, form.solution(), weightByParity);
    
    double tol = 1e-11;
    double err;
    err = (u1_soln - u1_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u1_hat_soln - u1_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u2_soln - u2_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u2_hat_soln - u2_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (tn1_hat_soln - tn1_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    
    bool repartitionAndRebuild = false;
    
    auto outputMesh = [mesh] (int ordinal) -> void
    {
      MeshTopologyPtr meshCopy = mesh->getTopology()->getGatheredCopy();
      int rank = mesh->Comm()->MyPID();
      if (rank == 0)
      {
        bool labelCells = true;
        int numPointsPerEdge = 2;
        ostringstream name;
        name << "meshSequence-" << ordinal;
        GnuPlotUtil::writeExactMeshSkeleton(name.str(), meshCopy.get(), numPointsPerEdge, labelCells);
      }
    };
    
    int meshOrdinal = 0;
//    outputMesh(meshOrdinal++);
    
    {
      vector<GlobalIndexType> cellIDs = {1};
      mesh->hRefine(cellIDs, repartitionAndRebuild);
      mesh->enforceOneIrregularity();
      mesh->repartitionAndRebuild();
//      outputMesh(meshOrdinal++);
      cellIDs = {4};
      mesh->hRefine(cellIDs, repartitionAndRebuild);
      mesh->enforceOneIrregularity();
      mesh->repartitionAndRebuild();
    }
//    outputMesh(meshOrdinal++);
    
    int rank = mesh->Comm()->MyPID();

    int activeElements = mesh->numActiveElements();
//    cout << "active elements seen by rank " << rank << ": " << activeElements << endl;
    
    err = (u1_soln - u1_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u1_hat_soln - u1_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u2_soln - u2_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u2_hat_soln - u2_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (tn1_hat_soln - tn1_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    
    if (!success)
    {
      HDF5Exporter exporter(mesh, "stokesTestFailure, prior to solve");
      exporter.exportSolution(form.solution(),0);
      
      HDF5Exporter exporter2(mesh, "stokesTestFailure, prior to solve, difference in tn1");
      exporter2.exportFunction({tn1_hat_soln-tn1_exact,tn1_hat_soln, tn1_exact},{"tn1 error", "tn1_hat solution", "tn1_hat exact"},0);
    }
    
    form.addInflowCondition(SpatialFilter::allSpace(), Function::vectorize(u1_exact, u2_exact));
    form.addPointPressureCondition();
    form.solve();

    err = (u1_soln - u1_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u1_hat_soln - u1_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u2_soln - u2_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    err = (u2_hat_soln - u2_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    
    if (!success)
    {
      HDF5Exporter exporter(mesh, "stokesTestFailure");
      exporter.exportSolution(form.solution(),0);
    }
//    else
//    {
//      // DEBUGGING: output even on success
//      HDF5Exporter exporter(mesh, "stokesTestFailure, correct solution");
//      exporter.exportSolution(form.solution(),0);
//    }
  }
  
//  TEUCHOS_UNIT_TEST( StokesVGPFormulation, RefinedMesh )
//  {
//    // this test imitates an issue found in the "wild" -- expected to fail on 4 MPI ranks, but succeed on 3
//    // the only test here is whether we can run to completion without generating an exception...
//    
//    MPIWrapper::CommWorld()->Barrier();
//    
//    int spaceDim = 2;
//    bool useConformingTraces = true;
//    int meshWidth = 2;
//    int polyOrder = 0;
//    int delta_k = 0;
//    double mu = 1.0;
//    
//    StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, useConformingTraces);
//    
//    vector<double> dims(spaceDim,1.0);
//    vector<int> numElements(spaceDim,meshWidth);
//    vector<double> x0(spaceDim,0.0);
//    
//    int maxIters = 1000;
//    double cgTol = 1e-6;
//    
//    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
//    form.initializeSolution(meshTopo, polyOrder, delta_k);
//    form.addPointPressureCondition();
//    
//    VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);
//    
//    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1);
//    SpatialFilterPtr notTopBoundary = SpatialFilter::negatedFilter(topBoundary);
//    form.addWallCondition(notTopBoundary);
//    
//    FunctionPtr u1_topRamp = Function::zero();
//    FunctionPtr u_topRamp;
//    FunctionPtr zero = Function::zero();
//    if (spaceDim == 2)
//    {
//      u_topRamp = Function::vectorize(u1_topRamp,zero);
//    }
//    else
//    {
//      u_topRamp = Function::vectorize(u1_topRamp,zero,zero);
//    }
//    form.addInflowCondition(topBoundary, u_topRamp);
//    
//    form.solveIteratively(maxIters, cgTol);
//    
//    MeshPtr mesh = form.solution()->mesh();
//    {
//      // DEBUGGING
//      bool repartitionAndRebuild = false;
//      
//      vector<GlobalIndexType> cellIDs = {3,1};
//      mesh->hRefine(cellIDs, repartitionAndRebuild);
//      mesh->enforceOneIrregularity();
//      mesh->repartitionAndRebuild();
//      cellIDs = {10, 7};
//      mesh->hRefine(cellIDs, repartitionAndRebuild);
//      mesh->enforceOneIrregularity();
//      mesh->repartitionAndRebuild();
//      cellIDs = {18, 15};
//      mesh->hRefine(cellIDs, repartitionAndRebuild);
//      mesh->enforceOneIrregularity();
//      mesh->repartitionAndRebuild();
//      cellIDs = {26, 23};
//      mesh->hRefine(cellIDs, repartitionAndRebuild);
//      mesh->enforceOneIrregularity();
//      mesh->repartitionAndRebuild();
//      cellIDs = {34, 31};
//      mesh->hRefine(cellIDs, repartitionAndRebuild);
//      mesh->enforceOneIrregularity();
//      mesh->repartitionAndRebuild();
//      cellIDs = {42, 43, 38, 39};
//      mesh->hRefine(cellIDs, repartitionAndRebuild);
//      mesh->enforceOneIrregularity();
//      mesh->repartitionAndRebuild();
//    }
//  }

TEUCHOS_UNIT_TEST( StokesVGPFormulation, SaveAndLoad )
{
  vector<double> dimensions = {1.0, 2.0}; // 1 x 2 domain
  vector<int> elementCounts = {3, 2}; // 3 x 2 mesh
  vector<double> x0 = {0.0, 0.0};
  int spaceDim = dimensions.size();

  double mu = 1.0;
  bool useConformingTraces = true;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, useConformingTraces);

  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

  int fieldPolyOrder = 1, delta_k = 1;

  form.initializeSolution(meshTopo, fieldPolyOrder, delta_k);

  string savePrefix = "StokesVGPTest";
  form.save(savePrefix);

  StokesVGPFormulation loadedForm = StokesVGPFormulation::steadyFormulation(meshTopo->getDimension(), mu, useConformingTraces);

  loadedForm.initializeSolution(savePrefix,fieldPolyOrder,delta_k);

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  Comm.Barrier(); // Barrier to make sure that everyone is done with the files before we delete them
  
  int rank = Teuchos::GlobalMPISession::getRank();
  
  if (rank == 0)
  {
    // delete the files we created
    remove((savePrefix+".soln").c_str());
    remove((savePrefix+".mesh").c_str());
    remove((savePrefix+"_stream.soln").c_str());
    remove((savePrefix+"_stream.mesh").c_str());
  }

//    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*> (loadedForm.solution()->mesh()->globalDofAssignment().get());

//    set<GlobalIndexType> cellsToRefine = {0};
//    loadedForm.solution()->mesh()->pRefine(cellsToRefine);
}
  
  TEUCHOS_UNIT_TEST( StokesVGPFormulation, SpaceTime2D_PureDirichletSolve_Slow )
  {
    int spaceDim = 2;
    double mu = 1.0;
    bool useConformingTraces = true;
    StokesVGPFormulation spaceTimeForm = StokesVGPFormulation::spaceTimeFormulation(spaceDim, mu, useConformingTraces);
    
    // build a 1 x 1 x T box, single element
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology({1.0,1.0}, {1,1});
    double t_0 = 0.0;
    double t_final = 1.0;
    int numTimeElements = 1;
    MeshTopologyPtr meshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t_0, t_final, numTimeElements);
    
    // choose exact solution that matches 0 at time 0
    FunctionPtr t = Function::tn(1);
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr u1 = t * t;
    FunctionPtr u2 = t;
    FunctionPtr p = t * x * y;
    
    //    FunctionPtr u1 = t;
    //    FunctionPtr u2 = Function::zero();
    //    FunctionPtr p = Function::zero();
    
    FunctionPtr u = Function::vectorize(u1, u2);
    
    FunctionPtr f = spaceTimeForm.forcingFunction(u, p);
    
    int polyOrder = 2, delta_k = 1;
    int temporalPolyOrder = 2;
    spaceTimeForm.initializeSolution(meshTopo, polyOrder, delta_k, f, temporalPolyOrder);
    
    SpatialFilterPtr spatialBoundary = SpatialFilter::allSpace();
    
    spaceTimeForm.addInflowCondition(spatialBoundary, u);
    spaceTimeForm.addZeroInitialCondition(t_0);
    spaceTimeForm.addPointPressureCondition({0.0,0.0});
    
    spaceTimeForm.solve();
    
    FunctionPtr pSoln = spaceTimeForm.getPressureSolution();
    FunctionPtr uSoln = spaceTimeForm.getVelocitySolution();
    
    double pSoln_mean = pSoln->integrate(spaceTimeForm.solution()->mesh());
    // the true pSoln is one with zero average
    pSoln = pSoln - pSoln_mean;
    
    double p_mean = p->integrate(spaceTimeForm.solution()->mesh()); // zero, for our present p
    p = p - p_mean;
    
    double tol = 1e-10;
    double pErr = (pSoln - p)->l2norm(spaceTimeForm.solution()->mesh());
    double uErr = (uSoln - u)->l2norm(spaceTimeForm.solution()->mesh());
    
    TEUCHOS_TEST_COMPARE(pErr, <, tol, out, success);
    TEUCHOS_TEST_COMPARE(uErr, <, tol, out, success);
    
    double energyError = spaceTimeForm.solution()->energyErrorTotal();
    TEUCHOS_TEST_COMPARE(energyError, <, tol, out, success);
    
    if (!success)
    {         // DEBUGGING:
      {
        SolutionPtr solution = spaceTimeForm.solution();
        solution->initializeLHSVector();
        solution->initializeStiffnessAndLoad();
        solution->populateStiffnessAndLoad();
        
        Teuchos::RCP<Epetra_CrsMatrix> A = solution->getStiffnessMatrix();
        string fileName = "/tmp/A.dat";
        EpetraExt::RowMatrixToMatrixMarketFile(fileName.c_str(),*A, NULL, NULL, false);
      }
      cout << "bf:\n";
      spaceTimeForm.bf()->printTrialTestInteractions();
      
      cout << "forcing function: " << f->displayString() << endl;
      HDF5Exporter exporter(spaceTimeForm.solution()->mesh(),"StokesSpaceTimeForcingFunction","/tmp");
      FunctionPtr f_padded = Function::vectorize(f->x(), f->y(), Function::zero());
      exporter.exportFunction(f_padded, "forcing function", 0.0, 5);
      
      HDF5Exporter solutionExporter(spaceTimeForm.solution()->mesh(),"StokesSpaceTimeSolution","/tmp");
      // export the projected solution at "time" 0
      solutionExporter.exportSolution(spaceTimeForm.solution(), 0.0, 5);
      
      // solve, and export the solution at "time" 1
      spaceTimeForm.solve();
      solutionExporter.exportSolution(spaceTimeForm.solution(), 1.0, 5);
      cout << "Exported solution.\n";
    }
  }

//  TEUCHOS_UNIT_TEST( StokesVGPFormulation, SpaceTime2D_PureDirichletRefinedSolve_Slow )
//  {
//    int spaceDim = 2;
//    double mu = 1.0;
//    bool useConformingTraces = true;
//    StokesVGPFormulation spaceTimeForm = StokesVGPFormulation::spaceTimeFormulation(spaceDim, mu, useConformingTraces);
//    
//    // build a 1 x 1 x T box, single element
//    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology({1.0,1.0}, {1,1});
//    double t_0 = 0.0;
//    double t_final = 1.0;
//    int numTimeElements = 1;
//    MeshTopologyPtr meshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t_0, t_final, numTimeElements);
//
//    // choose exact solution that matches 0 at time 0
//    FunctionPtr t = Function::tn(1);
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//    FunctionPtr u1 = t * t;
//    FunctionPtr u2 = t;
//    FunctionPtr p = t * x * y;
//    
////    FunctionPtr u1 = t;
////    FunctionPtr u2 = Function::zero();
////    FunctionPtr p = Function::zero();
//    
//    FunctionPtr u = Function::vectorize(u1, u2);
//    
//    FunctionPtr f = spaceTimeForm.forcingFunction(u, p);
//    
//    int polyOrder = 3, delta_k = 1;
//    int temporalPolyOrder = 2;
//    spaceTimeForm.initializeSolution(meshTopo, polyOrder, delta_k, f, temporalPolyOrder);
//    
//    // refine here:
//    spaceTimeForm.solution()->mesh()->hRefine(set<GlobalIndexType>{0});
//    
//    SpatialFilterPtr spatialBoundary = SpatialFilter::allSpace();
//    
//    // just checking something
////    cout << "Experimentally, using spatialBoundary = allSpace in test, just to see if that fixes things.\n";
////    spatialBoundary = SpatialFilter::allSpace();
//    
//    spaceTimeForm.addInflowCondition(spatialBoundary, u);
//    spaceTimeForm.addZeroInitialCondition(t_0);
//    spaceTimeForm.addPointPressureCondition({0.0,0.0});
//    
//    spaceTimeForm.solve();
//    
//    FunctionPtr pSoln = spaceTimeForm.getPressureSolution();
//    FunctionPtr uSoln = spaceTimeForm.getVelocitySolution();
//    
//    double pSoln_mean = pSoln->integrate(spaceTimeForm.solution()->mesh());
//    // the true pSoln is one with zero average
//    pSoln = pSoln - pSoln_mean;
//    
//    double p_mean = p->integrate(spaceTimeForm.solution()->mesh()); // zero, for our present p
//    p = p - p_mean;
//    
//    double tol = 1e-10;
//    double pErr = (pSoln - p)->l2norm(spaceTimeForm.solution()->mesh());
//    double uErr = (uSoln - u)->l2norm(spaceTimeForm.solution()->mesh());
//    
//    TEUCHOS_TEST_COMPARE(pErr, <, tol, out, success);
//    TEUCHOS_TEST_COMPARE(uErr, <, tol, out, success);
//    
//    double energyError = spaceTimeForm.solution()->energyErrorTotal();
//    TEUCHOS_TEST_COMPARE(energyError, <, tol, out, success);
//    
//    if (!success)
//    {         // DEBUGGING:
//      {
//        SolutionPtr solution = spaceTimeForm.solution();
//        solution->initializeLHSVector();
//        solution->initializeStiffnessAndLoad();
//        solution->populateStiffnessAndLoad();
//        
//        Teuchos::RCP<Epetra_CrsMatrix> A = solution->getStiffnessMatrix();
//        string fileName = "/tmp/A.dat";
//        EpetraExt::RowMatrixToMatrixMarketFile(fileName.c_str(),*A, NULL, NULL, false);
//      }
//      cout << "bf:\n";
//      spaceTimeForm.bf()->printTrialTestInteractions();
//      
//      cout << "forcing function: " << f->displayString() << endl;
//      HDF5Exporter exporter(spaceTimeForm.solution()->mesh(),"StokesSpaceTimeForcingFunction","/tmp");
//      FunctionPtr f_padded = Function::vectorize(f->x(), f->y(), Function::zero());
//      exporter.exportFunction(f_padded, "forcing function", 0.0, 5);
//
//      HDF5Exporter solutionExporter(spaceTimeForm.solution()->mesh(),"StokesSpaceTimeSolution","/tmp");
//      // export the projected solution at "time" 0
//      solutionExporter.exportSolution(spaceTimeForm.solution(), 0.0, 5);
//
//      // solve, and export the solution at "time" 1
//      spaceTimeForm.solve();
//      solutionExporter.exportSolution(spaceTimeForm.solution(), 1.0, 5);
//      cout << "Exported solution.\n";
//    }
//  }
  
  TEUCHOS_UNIT_TEST( StokesVGPFormulation, SpaceTime2D_FreeStreamSolve_Slow )
  {
    // try a constant-velocity horizontal flow, with inflow conditions on the left, outflow conditions on right and top/bottom
    
    int spaceDim = 2;
    double mu = 1.0;
    bool useConformingTraces = true;
    StokesVGPFormulation spaceTimeForm = StokesVGPFormulation::spaceTimeFormulation(spaceDim, mu, useConformingTraces);
    
    // build a 1 x 1 x T box, single element
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology({1.0,1.0}, {1,1});
    double t_0 = 0.0;
    double t_final = 1.0;
    int numTimeElements = 1;
    MeshTopologyPtr meshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t_0, t_final, numTimeElements);
    
    // choose exact solution that matches 0 at time 0
    FunctionPtr t = Function::tn(1);
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr u1 = t * t;
    FunctionPtr u2 = t;
    FunctionPtr p = t * (1-x) * (1-y) * y; // 0 at t=0, and at all three outflow sides
    
//    FunctionPtr u1 = Function::constant(1.0);
//    FunctionPtr u2 = Function::zero();
//    FunctionPtr p = Function::zero();
    
    FunctionPtr u = Function::vectorize(u1, u2);
    
    FunctionPtr f = spaceTimeForm.forcingFunction(u, p);
    
    int polyOrder = 2, delta_k = 1, temporalPolyOrder = 2;
    spaceTimeForm.initializeSolution(meshTopo, polyOrder, delta_k, f, temporalPolyOrder);
    
    SpatialFilterPtr outflowBoundary = SpatialFilter::matchingX(1.0) | SpatialFilter::matchingY(0) | SpatialFilter::matchingY(1);
    
    spaceTimeForm.addOutflowCondition(outflowBoundary, false);
    spaceTimeForm.addInflowCondition(SpatialFilter::matchingX(0), u);
    spaceTimeForm.addInitialCondition(t_0, {u1,u2}); // no pressure condition
    
    spaceTimeForm.solve();
    
    FunctionPtr pSoln = spaceTimeForm.getPressureSolution();
    FunctionPtr uSoln = spaceTimeForm.getVelocitySolution();
    
    double pSoln_mean = pSoln->integrate(spaceTimeForm.solution()->mesh());
    // the true pSoln is one with zero average
    pSoln = pSoln - pSoln_mean;
    
    double p_mean = p->integrate(spaceTimeForm.solution()->mesh()); // zero, for our present p
    p = p - p_mean;
    
    double tol = 1e-11;
    double pErr = (pSoln - p)->l2norm(spaceTimeForm.solution()->mesh());
    double uErr = (uSoln - u)->l2norm(spaceTimeForm.solution()->mesh());
    
    TEUCHOS_TEST_COMPARE(pErr, <, tol, out, success);
    TEUCHOS_TEST_COMPARE(uErr, <, tol, out, success);
    
    double energyError = spaceTimeForm.solution()->energyErrorTotal();
    TEUCHOS_TEST_COMPARE(energyError, <, tol, out, success);
    
//    {         // DEBUGGING:
//      {
//        SolutionPtr solution = spaceTimeForm.solution();
//        solution->initializeLHSVector();
//        solution->initializeStiffnessAndLoad();
//        solution->populateStiffnessAndLoad();
//        
//        Teuchos::RCP<Epetra_CrsMatrix> A = solution->getStiffnessMatrix();
//        string fileName = "/tmp/A.dat";
//        EpetraExt::RowMatrixToMatrixMarketFile(fileName.c_str(),*A, NULL, NULL, false);
//      }
//      cout << "bf:\n";
//      spaceTimeForm.bf()->printTrialTestInteractions();
//      
//      cout << "forcing function: " << f->displayString() << endl;
//      HDF5Exporter exporter(spaceTimeForm.solution()->mesh(),"StokesSpaceTimeForcingFunction","/tmp");
//      FunctionPtr f_padded = Function::vectorize(f->x(), f->y(), Function::zero());
//      exporter.exportFunction(f_padded, "forcing function", 0.0, 5);
//      
//      HDF5Exporter solutionExporter(spaceTimeForm.solution()->mesh(),"StokesSpaceTimeSolution","/tmp");
//      // export the projected solution at "time" 0
//      solutionExporter.exportSolution(spaceTimeForm.solution(), 0.0, 5);
//      
//      // solve, and export the solution at "time" 1
//      spaceTimeForm.solve();
//      solutionExporter.exportSolution(spaceTimeForm.solution(), 1.0, 5);
//      cout << "Exported solution.\n";
//    }
  }
} // namespace
