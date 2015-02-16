//
//  StokesVGPFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "StokesVGPFormulation.h"
#include "PenaltyConstraints.h"

const string StokesVGPFormulation::S_U1 = "u_1";
const string StokesVGPFormulation::S_U2 = "u_2";
const string StokesVGPFormulation::S_U3 = "u_3";
const string StokesVGPFormulation::S_P = "p";
const string StokesVGPFormulation::S_SIGMA1 = "\\sigma_{1}";
const string StokesVGPFormulation::S_SIGMA2 = "\\sigma_{2}";
const string StokesVGPFormulation::S_SIGMA3 = "\\sigma_{3}";

const string StokesVGPFormulation::S_U1_HAT = "\\widehat{u}_1";
const string StokesVGPFormulation::S_U2_HAT = "\\widehat{u}_2";
const string StokesVGPFormulation::S_U3_HAT = "\\widehat{u}_3";
const string StokesVGPFormulation::S_TN1_HAT = "\\widehat{t}_{1n}";
const string StokesVGPFormulation::S_TN2_HAT = "\\widehat{t}_{2n}";
const string StokesVGPFormulation::S_TN3_HAT = "\\widehat{t}_{3n}";

const string StokesVGPFormulation::S_V1 = "v_1";
const string StokesVGPFormulation::S_V2 = "v_2";
const string StokesVGPFormulation::S_V3 = "v_3";
const string StokesVGPFormulation::S_TAU1 = "\\tau_{1}";
const string StokesVGPFormulation::S_TAU2 = "\\tau_{2}";
const string StokesVGPFormulation::S_TAU3 = "\\tau_{3}";
const string StokesVGPFormulation::S_Q = "q";

StokesVGPFormulation::StokesVGPFormulation(int spaceDim, bool useConformingTraces, double mu,
                                           bool transient, double dt) {
  _spaceDim = spaceDim;
  _useConformingTraces = useConformingTraces;
  _mu = mu;
  _dt = ParameterFunction::parameterFunction(dt);
  
  _theta = ParameterFunction::parameterFunction(0.5); // Crank-Nicolson
  _transient = transient;
  
  if ((spaceDim != 2) && (spaceDim != 3)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 2 or 3");
  }
  
  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u1, u2, u3;
  VarPtr p;
  VarPtr sigma1, sigma2, sigma3;
  
  // traces
  VarPtr u1_hat, u2_hat, u3_hat;
  VarPtr t1n, t2n, t3n;
  
  // tests
  VarPtr v1, v2, v3;
  VarPtr tau1, tau2, tau3;
  VarPtr q;
  
  u1 = _vf.fieldVar(S_U1);
  u2 = _vf.fieldVar(S_U2);
  if (spaceDim==3) u3 = _vf.fieldVar(S_U3);
  
  p = _vf.fieldVar(S_P);
  
  sigma1 = _vf.fieldVar(S_SIGMA1, VECTOR_L2);
  sigma2 = _vf.fieldVar(S_SIGMA2, VECTOR_L2);
  if (spaceDim==3) {
    sigma3 = _vf.fieldVar(S_SIGMA3, VECTOR_L2);
  }
  
  Space uHatSpace = useConformingTraces ? HGRAD : L2;
  
  u1_hat = _vf.traceVar(S_U1_HAT, 1.0 * u1, uHatSpace);
  u2_hat = _vf.traceVar(S_U2_HAT, 1.0 * u2, uHatSpace);
  if (spaceDim==3) u3_hat = _vf.traceVar(S_U3_HAT, 1.0 * u3, uHatSpace);
  
  FunctionPtr n = Function::normal();
  LinearTermPtr t1n_lt, t2n_lt, t3n_lt;
  t1n_lt = p * n->x() - sigma1 * n ;
  t2n_lt = p * n->y() - sigma2 * n;
  if (spaceDim==3) {
    t3n_lt = p * n->z() - sigma3 * n;
  }
  t1n = _vf.fluxVar(S_TN1_HAT, t1n_lt);
  t2n = _vf.fluxVar(S_TN2_HAT, t2n_lt);
  if (spaceDim==3) t3n = _vf.fluxVar(S_TN3_HAT, t3n_lt);
  
  v1 = _vf.testVar(S_V1, HGRAD);
  v2 = _vf.testVar(S_V2, HGRAD);
  if (spaceDim==3) v3 = _vf.testVar(S_V3, HGRAD);
  
  tau1 = _vf.testVar(S_TAU1, HDIV);
  tau2 = _vf.testVar(S_TAU2, HDIV);
  if (spaceDim==3) {
    tau3 = _vf.testVar(S_TAU3, HDIV);
  }
  
  q = _vf.testVar(S_Q, HGRAD);
  
  _steadyStokesBF = Teuchos::rcp( new BF(_vf) );
  // v1
  // tau1 terms:
  _steadyStokesBF->addTerm(u1, tau1->div());
  _steadyStokesBF->addTerm((1.0/_mu) * sigma1, tau1); // (sigma1, tau1)
  _steadyStokesBF->addTerm(-u1_hat, tau1->dot_normal());
  
  // tau2 terms:
  _steadyStokesBF->addTerm(u2, tau2->div());
  _steadyStokesBF->addTerm((1.0/_mu) * sigma2, tau2);
  _steadyStokesBF->addTerm(-u2_hat, tau2->dot_normal());
  
  // tau3:
  if (spaceDim==3) {
    _steadyStokesBF->addTerm(u3, tau3->div());
    _steadyStokesBF->addTerm((1.0/_mu) * sigma3, tau3);
    _steadyStokesBF->addTerm(-u3_hat, tau3->dot_normal());
  }
  
  // v1:
  _steadyStokesBF->addTerm(sigma1, v1->grad()); // (mu sigma1, grad v1)
  _steadyStokesBF->addTerm( - p, v1->dx() );
  _steadyStokesBF->addTerm( t1n, v1);
  
  // v2:
  _steadyStokesBF->addTerm(sigma2, v2->grad()); // (mu sigma2, grad v2)
  _steadyStokesBF->addTerm( - p, v2->dy());
  _steadyStokesBF->addTerm( t2n, v2);
  
  // v3:
  if (spaceDim==3) {
    _steadyStokesBF->addTerm(sigma3, v3->grad()); // (mu sigma3, grad v3)
    _steadyStokesBF->addTerm( - p, v3->dz());
    _steadyStokesBF->addTerm( t3n, v3);
  }
  
  // q:
  _steadyStokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  _steadyStokesBF->addTerm(-u2,q->dy());
  if (spaceDim==3) _steadyStokesBF->addTerm(-u3, q->dz());
  
  if (spaceDim==2) {
    _steadyStokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
  } else if (spaceDim==3) {
    _steadyStokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y() + u3_hat * n->z(), q);
  }
  
  if (!_transient) {
    _stokesBF = _steadyStokesBF;
  } else {
    // v1
    // tau1 terms:
    _stokesBF->addTerm(_theta * u1, tau1->div());
    _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma1), tau1); // (sigma1, tau1)
    _stokesBF->addTerm(-u1_hat, tau1->dot_normal());
    
    // tau2 terms:
    _stokesBF->addTerm(_theta * u2, tau2->div());
    _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma2), tau2);
    _stokesBF->addTerm(-u2_hat, tau2->dot_normal());
    
    // tau3:
    if (spaceDim==3) {
      _stokesBF->addTerm(_theta * u3, tau3->div());
      _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma3), tau3);
      _stokesBF->addTerm(-u3_hat, tau3->dot_normal());
    }
    
    // v1:
    _stokesBF->addTerm(u1 / _dt, v1);
    _stokesBF->addTerm(_theta * sigma1, v1->grad()); // (mu sigma1, grad v1)
    _stokesBF->addTerm(_theta * (-p), v1->dx() );
    _stokesBF->addTerm( t1n, v1);
    
    // v2:
    _stokesBF->addTerm(u2 / _dt, v2);
    _stokesBF->addTerm(_theta * sigma2, v2->grad()); // (mu sigma2, grad v2)
    _stokesBF->addTerm(_theta * (-p), v2->dy());
    _stokesBF->addTerm( t2n, v2);
    
    // v3:
    if (spaceDim==3) {
      _stokesBF->addTerm(u3 / _dt, v3);
      _stokesBF->addTerm(_theta * sigma3, v3->grad()); // (mu sigma3, grad v3)
      _stokesBF->addTerm(_theta * (- p), v3->dz());
      _stokesBF->addTerm( t3n, v3);
    }
    
    // q:
    _stokesBF->addTerm(_theta * (-u1),q->dx()); // (-u, grad q)
    _stokesBF->addTerm(_theta * (-u2),q->dy());
    if (spaceDim==3) _stokesBF->addTerm(_theta * (-u3), q->dz());
    
    if (spaceDim==2) {
      _stokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
    } else if (spaceDim==3) {
      _stokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y() + u3_hat * n->z(), q);
    }
  }
  
  // define tractions (used in outflow conditions)
  // definition of traction: _mu * ( (\nabla u) + (\nabla u)^T ) n - p n
  //                      = (sigma + sigma^T) n - p n
  if (spaceDim == 2) {
    _t1 = n->x() * (2 * sigma1->x() - p)       + n->y() * (sigma1->x() + sigma2->x());
    _t2 = n->x() * (sigma1->y() + sigma2->x()) + n->y() * (2 * sigma2->y() - p);
  } else {
    _t1 = n->x() * (2 * sigma1->x() - p)       + n->y() * (sigma1->x() + sigma2->x()) + n->z() * (sigma1->z() + sigma3->x());
    _t2 = n->x() * (sigma1->y() + sigma2->x()) + n->y() * (2 * sigma2->y() - p)       + n->z() * (sigma2->z() + sigma3->y());
    _t3 = n->x() * (sigma1->z() + sigma3->x()) + n->y() * (sigma2->z() + sigma3->y()) + n->z() * (2 * sigma3->z() - p);
  }
}

void StokesVGPFormulation::addInflowCondition(SpatialFilterPtr inflowRegion, FunctionPtr u) {
  int spaceDim = _solution->mesh()->getTopology()->getSpaceDim();
  
  VarPtr u1_hat = this->u_hat(1), u2_hat = this->u_hat(2);
  VarPtr u3_hat;
  if (spaceDim==3) u3_hat = this->u_hat(3);
  
  _solution->bc()->addDirichlet(u1_hat, inflowRegion, u->x());
  _solution->bc()->addDirichlet(u2_hat, inflowRegion, u->y());
  if (spaceDim==3) _solution->bc()->addDirichlet(u3_hat, inflowRegion, u->z());
}

void StokesVGPFormulation::addOutflowCondition(SpatialFilterPtr outflowRegion) {
  int spaceDim = _solution->mesh()->getTopology()->getSpaceDim();
  
  // my favorite way to do outflow conditions is via penalty constraints imposing a zero traction
  Teuchos::RCP<LocalStiffnessMatrixFilter> filter_incr = _solution->filter();
  
  Teuchos::RCP< PenaltyConstraints > pcRCP;
  PenaltyConstraints* pc;
  
  if (filter_incr.get() != NULL) {
    pc = dynamic_cast<PenaltyConstraints*>(filter_incr.get());
    if (pc == NULL) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't add PenaltyConstraints when a non-PenaltyConstraints LocalStiffnessMatrixFilter already in place");
    }
  } else {
    pcRCP = Teuchos::rcp( new PenaltyConstraints );
    pc = pcRCP.get();
  }
  FunctionPtr zero = Function::zero();
  pc->addConstraint(_t1==zero, outflowRegion);
  pc->addConstraint(_t2==zero, outflowRegion);
  if (spaceDim==3) pc->addConstraint(_t3==zero, outflowRegion);
  
  if (pcRCP != Teuchos::null) { // i.e., we're not just adding to a prior PenaltyConstraints object
    _solution->setFilter(pcRCP);
  }
}

void StokesVGPFormulation::addPointPressureCondition() {
  VarPtr p = this->p();
  
  _solution->bc()->addSinglePointBC(p->ID(), 0.0);
  
  if (_solution->bc()->imposeZeroMeanConstraint(p->ID())) {
    _solution->bc()->removeZeroMeanConstraint(p->ID());
  }
}

void StokesVGPFormulation::addWallCondition(SpatialFilterPtr wall) {
  int spaceDim = _solution->mesh()->getTopology()->getSpaceDim();
  vector<double> zero(spaceDim, 0.0);
  addInflowCondition(wall, Function::constant(zero));
}

void StokesVGPFormulation::addZeroMeanPressureCondition() {
  VarPtr p = this->p();
  
  _solution->bc()->addZeroMeanConstraint(p);
  
  if (_solution->bc()->singlePointBC(p->ID())) {
    _solution->bc()->removeSinglePointBC(p->ID());
  }
}

BFPtr StokesVGPFormulation::bf() {
  return _stokesBF;
}

void StokesVGPFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
                                              FunctionPtr forcingFunction) {
  int H1Order = fieldPolyOrder + 1;
  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, _stokesBF, H1Order, delta_k) ) ;
 
  BCPtr bc = BC::bc();
  
  _solution = Solution::solution(mesh,bc);
  if (_transient) _previousSolution = Solution::solution(mesh,bc);
  
  RHSPtr rhs = this->rhs(forcingFunction); // in transient case, this will refer to _previousSolution
  IPPtr ip = _stokesBF->graphNorm();
  
  _solution->setRHS(rhs);
  _solution->setIP(ip);
  
  mesh->registerSolution(_solution); // will project both time steps during refinements...
  if (_transient) mesh->registerSolution(_previousSolution);
  
  LinearTermPtr residual = rhs->linearTerm() - _stokesBF->testFunctional(_solution,false); // false: don't exclude boundary terms
  
  _time = 0;
}

VarPtr StokesVGPFormulation::p() {
  return _vf.fieldVar(S_P);
}

RefinementStrategyPtr StokesVGPFormulation::getRefinementStrategy() {
  return _refinementStrategy;
}

void StokesVGPFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy) {
  _refinementStrategy = refStrategy;
}

void StokesVGPFormulation::refine() {
  _refinementStrategy->refine();
}

RHSPtr StokesVGPFormulation::rhs(FunctionPtr f) {
  int spaceDim = _solution->mesh()->getTopology()->getSpaceDim();
  
  RHSPtr rhs = RHS::rhs();
  
  VarPtr v1 = this->v(1);
  VarPtr v2 = this->v(2);
  VarPtr v3;
  if (spaceDim==3) v3 = this->v(3);
  
  if (f != Teuchos::null) {
    rhs->addTerm( f->x() * v1 );
    rhs->addTerm( f->y() * v2 );
    if (spaceDim == 3) rhs->addTerm( f->z() * v3 );
  }
  
  if (_transient) {
    FunctionPtr u1_prev, u2_prev, u3_prev;
    u1_prev = Function::solution(this->u(1), _previousSolution);
    u2_prev = Function::solution(this->u(2), _previousSolution);
    if (spaceDim==3) u3_prev = Function::solution(this->u(3), _previousSolution);
    rhs->addTerm(u1_prev / _dt * v1);
    rhs->addTerm(u2_prev / _dt * v2);
    if (spaceDim==3) rhs->addTerm(u3_prev / _dt * v3);
    
    bool excludeFluxesAndTraces = true;
    LinearTermPtr prevTimeStepFunctional = _steadyStokesBF->testFunctional(_previousSolution,excludeFluxesAndTraces);
    rhs->addTerm((_theta - 1.0) * prevTimeStepFunctional);
  }
  
  return rhs;
}

VarPtr StokesVGPFormulation::sigma(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i) {
    case 1:
      return _vf.fieldVar(S_SIGMA1);
    case 2:
      return _vf.fieldVar(S_SIGMA2);
    case 3:
      return _vf.fieldVar(S_SIGMA3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr StokesVGPFormulation::u(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i) {
    case 1:
      return _vf.fieldVar(S_U1);
    case 2:
      return _vf.fieldVar(S_U2);
    case 3:
      return _vf.fieldVar(S_U3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// traces:
VarPtr StokesVGPFormulation::tn_hat(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i) {
    case 1:
      return _vf.fluxVar(S_TN1_HAT);
    case 2:
      return _vf.fluxVar(S_TN2_HAT);
    case 3:
      return _vf.fluxVar(S_TN3_HAT);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr StokesVGPFormulation::u_hat(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i) {
    case 1:
      return _vf.traceVar(S_U1_HAT);
    case 2:
      return _vf.traceVar(S_U2_HAT);
    case 3:
      return _vf.traceVar(S_U3_HAT);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// test variables:
VarPtr StokesVGPFormulation::tau(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i) {
    case 1:
      return _vf.testVar(S_TAU1, HDIV);
    case 2:
      return _vf.testVar(S_TAU2, HDIV);
    case 3:
      return _vf.testVar(S_TAU3, HDIV);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// ! set current time step used for transient solve
void StokesVGPFormulation::setTimeStep(double dt) {
  _dt->setValue(dt);
}

// ! Returns the solution (at current time)
SolutionPtr StokesVGPFormulation::solution() {
  return _solution;
}

// ! Returns the solution (at previous time)
SolutionPtr StokesVGPFormulation::solutionPreviousTimeStep() {
  return _previousSolution;
}

// ! Solves
void StokesVGPFormulation::solve() {
  _solution->solve();
}

// ! Takes a time step (assumes you have called solve() first)
void StokesVGPFormulation::takeTimeStep() {
  SimpleFunction* dtValueFxn = dynamic_cast<SimpleFunction*>(_dt->getValue().get());
  
  double dt = dtValueFxn->value(0);
  _time += dt;
  
  // if we implemented some sort of value-replacement in Solution, that would be more efficient than this:
  _previousSolution->clear();
  _previousSolution->addSolution(_solution, 1.0);
}

// ! Returns the sum of the time steps taken thus far.
double StokesVGPFormulation::getTime() {
  return _time;
}

VarPtr StokesVGPFormulation::v(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i) {
    case 1:
      return _vf.testVar(S_V1, HGRAD);
    case 2:
      return _vf.testVar(S_V2, HGRAD);
    case 3:
      return _vf.testVar(S_V3, HGRAD);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}