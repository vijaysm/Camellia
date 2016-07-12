#include "AdditiveSchwarz.h"
#include "CamelliaDebugUtility.h"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GMGOperator.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "LinearElasticityFormulation.h"
#include "MeshFactory.h"
#include "NavierStokesVGPFormulation.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "Solver.h"
#include "TrigFunctions.h"
#include "TypeDefs.h"

#include "AztecOO.h"
#include "AztecOO_StatusTestResNorm.h"

#include "Epetra_Operator_to_Epetra_Matrix.h"
#include "Epetra_SerialComm.h"

#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_Amesos.h"
#include "Ifpack_IC.h"
#include "Ifpack_ILU.h"

#include <Teuchos_GlobalMPISession.hpp>

using namespace Camellia;
using namespace Intrepid;

bool runGMGOperatorInDebugMode;
int maxDofsForKLU;
double coarseCGTol;
int coarseMaxIterations;
bool narrateSolution;
bool narrateCoarseSolution;
bool saveVisualizationOutput = false;
FunctionPtr errToPlot;
string errFunctionName;

string getFactorizationTypeString(GMGOperator::FactorType factorizationType)
{
  switch (factorizationType)
  {
  case GMGOperator::Direct:
    return "Direct";
  case GMGOperator::IC:
    return "IC";
  case GMGOperator::ILU:
    return "ILU";
  default:
    return "Unknown";
  }
}

GMGOperator::FactorType getFactorizationType(string factorizationTypeString)
{
  if (factorizationTypeString == "Direct")
  {
    return GMGOperator::Direct;
  }
  if (factorizationTypeString == "IC") return GMGOperator::IC;
  if (factorizationTypeString == "ILU") return GMGOperator::ILU;

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "factorization type not recognized");
}

Teuchos::RCP<Epetra_Operator> CamelliaAdditiveSchwarzPreconditioner(::Epetra_RowMatrix* A, int overlapLevel, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter,
    GMGOperator::FactorType schwarzBlockFactorization,
    int levelOfFill, double fillRatio, bool hOnly)
{

  int sideDim = mesh->getTopology()->getDimension() - 1;
  Teuchos::RCP<Ifpack_Preconditioner> preconditioner;
  Teuchos::ParameterList List;
  switch (schwarzBlockFactorization)
  {
  case GMGOperator::Direct:
    preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_Amesos>(A, overlapLevel, mesh, dofInterpreter, hOnly, sideDim) );
    break;
  case GMGOperator::IC:
    preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_IC>(A, overlapLevel, mesh, dofInterpreter, hOnly, sideDim) );
    List.set("fact: ict level-of-fill", fillRatio);
    break;
  case GMGOperator::ILU:
    preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_ILU>(A, overlapLevel, mesh, dofInterpreter, hOnly, sideDim) );
    List.set("fact: level-of-fill", levelOfFill);
    break;
  default:
    break;
  }

  List.set("schwarz: combine mode", "Add"); // The PDF doc says to use "Insert" to maintain symmetry, but the HTML docs (which are more recent) say to use "Add".  http://trilinos.org/docs/r11.10/packages/ifpack/doc/html/index.html
  int err = preconditioner->SetParameters(List);
  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->SetParameters() returned with err " << err << endl;
  }

  err = preconditioner->Initialize();
  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Initialize() returned with err " << err << endl;
  }
  err = preconditioner->Compute();

  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Compute() returned with err = " << err << endl;
  }

  return preconditioner;
}

Teuchos::RCP<Epetra_Operator> IfPackAdditiveSchwarzPreconditioner(Epetra_RowMatrix* A, int overlapLevel,
    GMGOperator::FactorType schwarzBlockFactorization,
    int levelOfFill, double fillRatio)
{
  Teuchos::RCP<Ifpack_Preconditioner> preconditioner;
  Teuchos::ParameterList List;
  switch (schwarzBlockFactorization)
  {
  case GMGOperator::Direct:
    preconditioner = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_Amesos>(A, overlapLevel) );
    break;
  case GMGOperator::IC:
    preconditioner = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_IC>(A, overlapLevel) );
    List.set("fact: ict level-of-fill", fillRatio);
    break;
  case GMGOperator::ILU:
    preconditioner = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_ILU>(A, overlapLevel) );
    List.set("fact: level-of-fill", levelOfFill);
    break;
  default:
    break;
  }

  List.set("schwarz: combine mode", "Add"); // The PDF doc says to use "Insert" to maintain symmetry, but the HTML docs (which are more recent) say to use "Add".  http://trilinos.org/docs/r11.10/packages/ifpack/doc/html/index.html
  int err = preconditioner->SetParameters(List);
  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->SetParameters() returned with err " << err << endl;
  }


  err = preconditioner->Initialize();
  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Initialize() returned with err " << err << endl;
  }
  err = preconditioner->Compute();

  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Compute() returned with err = " << err << endl;
  }

  return preconditioner;
}

class AztecSolver : public Solver
{
  int _maxIters;
  double _tol;
  int _schwarzOverlap;
  bool _useSchwarzPreconditioner;
  bool _hierarchical; // for Camellia Schwarz preconditioners

  int _iterationCount;

  int _azOutputLevel;

  int _levelOfFill;
  double _fillRatio;

  MeshPtr _mesh;
  Teuchos::RCP<AztecOO_StatusTestResNorm> _statusTest;
  Teuchos::RCP<DofInterpreter> _dofInterpreter;
  GMGOperator::FactorType _schwarzBlockFactorization;
public:
  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner,
              GMGOperator::FactorType schwarzBlockFactorization, int levelOfFill, double fillRatio)
  {
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _azOutputLevel = 1;
    _schwarzBlockFactorization = schwarzBlockFactorization;
    _levelOfFill = levelOfFill;
    _fillRatio = fillRatio;
    _hierarchical = false;
  }

  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner,
              GMGOperator::FactorType schwarzBlockFactorization, int levelOfFill, double fillRatio,
              MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter, bool hierarchical)
  {
    _mesh = mesh;
    _dofInterpreter = dofInterpreter;
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _schwarzBlockFactorization = schwarzBlockFactorization;
    _azOutputLevel = 1;
    _levelOfFill = levelOfFill;
    _fillRatio = fillRatio;
    _hierarchical = hierarchical;
  }
  void setAztecOutputLevel(int AztecOutputLevel)
  {
    _azOutputLevel = AztecOutputLevel;
  }

  int solve()
  {
    AztecOO solver(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get());

    solver.SetAztecOption(AZ_solver, AZ_cg_condnum);

    solver.SetAztecOption(AZ_conv, AZ_rhs); // convergence is relative to the RHS
    
//    cout << "***Experiment: using one-norm of residual instead of two-norm for stopping criterion.***\n";
//    // I think we really want _tol * _tol (the status test really should take sqrt of one-norm), but doing _tol for now...
//    _statusTest = Teuchos::rcp( new AztecOO_StatusTestResNorm(*_stiffnessMatrix.get(), *(*_lhs.get())(0), *(*_rhs.get())(0), _tol));
//    solver.SetStatusTest(_statusTest.get());
//    _statusTest->DefineResForm(AztecOO_StatusTestResNorm::Explicit, AztecOO_StatusTestResNorm::OneNorm);
//    _statusTest->DefineScaleForm(AztecOO_StatusTestResNorm::NormOfRHS, AztecOO_StatusTestResNorm::OneNorm);
    
    Teuchos::RCP<Epetra_CrsMatrix> A = this->_stiffnessMatrix;

    Teuchos::RCP<Epetra_Operator> preconditioner;
    if (_mesh != Teuchos::null)
    {
      preconditioner = CamelliaAdditiveSchwarzPreconditioner(A.get(), _schwarzOverlap, _mesh, _dofInterpreter, _schwarzBlockFactorization,
                       _levelOfFill, _fillRatio, _hierarchical);

//      Teuchos::RCP< Epetra_CrsMatrix > M;
//      M = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, A->RowMatrixRowMap());
//
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "writing preconditioner to /tmp/preconditioner.dat.\n";
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/preconditioner.dat",*M, NULL, NULL, false);

    }
    else
    {
      preconditioner = IfPackAdditiveSchwarzPreconditioner(A.get(), _schwarzOverlap, _schwarzBlockFactorization, _levelOfFill, _fillRatio);
    }

    if (_useSchwarzPreconditioner)
    {
      solver.SetPrecOperator(preconditioner.get());
      solver.SetAztecOption(AZ_precond, AZ_user_precond);
    }
    else
    {
      solver.SetAztecOption(AZ_precond, AZ_none);
    }

    solver.SetAztecOption(AZ_output, _azOutputLevel);

    int solveResult = solver.Iterate(_maxIters,_tol);

    int remainingIters = _maxIters;

    const double* status = solver.GetAztecStatus();
    int whyTerminated = status[AZ_why];

    int rank = Teuchos::GlobalMPISession::getRank();

    int maxRestarts = 1;
    int numRestarts = 0;
    while ((whyTerminated==AZ_loss) && (numRestarts < maxRestarts))
    {
      remainingIters -= status[AZ_its];
      if (rank==0) cout << "Aztec warned that the recursive residual indicates convergence even though the true residual is too large.  Restarting with the new solution as initial guess, with maxIters = " << remainingIters << endl;
      solveResult = solver.Iterate(remainingIters,_tol);
      whyTerminated = status[AZ_why];
      numRestarts++;
    }
    remainingIters -= status[AZ_its];
    _iterationCount = _maxIters - remainingIters;

    switch (whyTerminated)
    {
    case AZ_normal:
//        cout << "whyTerminated: AZ_normal " << endl;
      break;
    case AZ_param:
      cout << "whyTerminated: AZ_param " << endl;
      break;
    case AZ_breakdown:
      cout << "whyTerminated: AZ_breakdown " << endl;
      break;
    case AZ_loss:
      cout << "whyTerminated: AZ_loss " << endl;
      break;
    case AZ_ill_cond:
      cout << "whyTerminated: AZ_ill_cond " << endl;
      break;
    case AZ_maxits:
      cout << "whyTerminated: AZ_maxits " << endl;
      break;
    default:
      break;
    }

    _iterationCount = status[AZ_its];

    return solveResult;
  }
  Teuchos::RCP< Epetra_CrsMatrix > getPreconditionerMatrix(const Epetra_Map &map)
  {
    Teuchos::RCP<Epetra_CrsMatrix> A = this->_stiffnessMatrix;
    Teuchos::RCP<Epetra_Operator> preconditioner;
    if (_mesh != Teuchos::null)
    {
      preconditioner = CamelliaAdditiveSchwarzPreconditioner(A.get(), _schwarzOverlap, _mesh, _dofInterpreter,
                       _schwarzBlockFactorization, _levelOfFill, _fillRatio,_hierarchical);

//      Teuchos::RCP< Epetra_CrsMatrix > M;
//      M = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, A->RowMatrixRowMap());
//
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "writing preconditioner to /tmp/preconditioner.dat.\n";
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/preconditioner.dat",*M, NULL, NULL, false);

    }
    else
    {
      preconditioner = IfPackAdditiveSchwarzPreconditioner(A.get(), _schwarzOverlap, _schwarzBlockFactorization,
                       _levelOfFill, _fillRatio);
    }

    return Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, map);
  }
  int iterationCount()
  {
    return _iterationCount;
  }
};

#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
#include <xmmintrin.h>
#endif

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "GlobalDofAssignment.h"

#include "CondensedDofInterpreter.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PoissonFormulation.h"
#include "ConvectionDiffusionFormulation.h"
#include "StokesVGPFormulation.h"

#include "GMGSolver.h"

enum ProblemChoice
{
  Poisson,
  ConvectionDiffusion,
  ConvectionDiffusionExperimental,
  Stokes,
  NavierStokes,
  LinearElasticity
};

enum RunManyPreconditionerChoices
{
  DontPrecondition,
  GMGAlgebraicSchwarz, // GMG with algebraic Schwarz smoother
  GMGGeometricSchwarz, // GMG with geometric Schwarz smoother
  GMGPointSymmetricGaussSeidel, // GMG with point symmetric Gauss Seidel smoother
  GMGBlockSymmetricGaussSeidel, // GMG with block symmetric Gauss Seidel smoother
  GMGPointJacobi, // GMG with point symmetric Jacobi smoother
  GMGBlockJacobi, // GMG with block symmetric Jacobi smoother
  AlgebraicSchwarz, // algebraic Schwarz preconditioner
  GeometricSchwarz, // geometric Schwarz preconditioner
  AllGMG,     // Schwarz smoother, multiple overlap values, both algebraic and geometric Schwarz
  AllSchwarz, // Schwarz as the only preconditioner; multiple overlap values, both algebraic and geometric Schwarz
  All         // All of the above, including the DontPrecondition option
};

Teuchos::RCP<GMGSolver> initializeGMGSolver(SolutionPtr solution, int AztecOutputLevel, GMGOperator::SmootherChoice smootherType,
                                            int schwarzOverlap, int dimensionForSchwarzNeighbors, GMGOperator::FactorType schwarzBlockFactorization, int schwarzLevelOfFill, double schwarzFillRatio,
                                            bool useStaticCondensation, bool hOnly, bool useHierarchicalNeighborsForSchwarz,
                                            MeshPtr coarseMesh,
                                            Solver::SolverChoice coarseSolverChoice, int cgMaxIterations, double cgTol,
                                            GMGOperator::SmootherApplicationType comboType, double smootherWeight, bool useWeightMatrixForSchwarz)
{
  int rank = Teuchos::GlobalMPISession::getRank();

  GMGSolver* gmgSolver = new GMGSolver(solution, coarseMesh, cgMaxIterations, cgTol, Teuchos::null, useStaticCondensation);
  gmgSolver->setNarrateOnRankZero(narrateSolution, "fine GMGSolver");
  gmgSolver->gmgOperator()->setNarrateOnRankZero(narrateSolution, "fine GMGOperator");
  gmgSolver->gmgOperator()->getCoarseSolution()->setNarrateOnRankZero(narrateCoarseSolution, "coarse solution");
  
  Teuchos::RCP<Solver> coarseSolver = Teuchos::null;
  
  bool saveFactorization = true;
  coarseSolver = Solver::getSolver(coarseSolverChoice, saveFactorization, coarseCGTol, coarseMaxIterations, gmgSolver->gmgOperator()->getCoarseSolution(),
                                   Solver::getDirectSolver(saveFactorization));
  
  gmgSolver->gmgOperator()->setCoarseSolver(coarseSolver);
  
  if ((smootherType == GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ) && hOnly)
  {
    // then use hierarchical neighbor relationship
    gmgSolver->gmgOperator()->setUseHierarchicalNeighborsForSchwarz(useHierarchicalNeighborsForSchwarz);
    
    if (rank==0)
    {
      if (useHierarchicalNeighborsForSchwarz)
        cout << "using hierarchical Schwarz neighbors option\n";
      else
        cout << "NOT using hierarchical Schwarz neighbors option\n";
    }

//    {
//      // DEBUGGING:
//      gmgSolver->gmgOperator()->setUseHierarchicalNeighborsForSchwarz(false);
//      if (rank==0) cout << "TEMPORARILY overriding NOT to use hierarchical neighbors for Schwarz.\n";
//    }
    
    //      if (conformingTraces)
    //      {
    //        // then vertex neighbors should count as neighbors
    //        gmgSolver->gmgOperator()->setDimensionForSchwarzNeighborRelationship(0);
    //        if (rank==0) cout << "conforming and h-multigrid: using vertex neighbors for Schwarz\n";
    //      }
  }
  
  gmgSolver->setAztecOutput(AztecOutputLevel);
  gmgSolver->setUseConjugateGradient(true);
  gmgSolver->setComputeConditionNumberEstimate(true);
  gmgSolver->gmgOperator()->setSchwarzFactorizationType(schwarzBlockFactorization);
  gmgSolver->gmgOperator()->setLevelOfFill(schwarzLevelOfFill);
  gmgSolver->gmgOperator()->setFillRatio(schwarzFillRatio);
  //    cout << "Set GMGOperator level of fill to " << schwarzLevelOfFill << endl;
  //    cout << "Set GMGOperator fill ratio to " << schwarzFillRatio << endl;
  gmgSolver->gmgOperator()->setSmootherType(smootherType);
  gmgSolver->gmgOperator()->setSmootherOverlap(schwarzOverlap);
  gmgSolver->gmgOperator()->setSmootherApplicationType(comboType);
  if (smootherWeight != -1)
  {
    gmgSolver->gmgOperator()->setSmootherWeight(smootherWeight);
  }
  else
    gmgSolver->gmgOperator()->setUseSchwarzScalingWeight(true);
  if (dimensionForSchwarzNeighbors != -1)
  {
    gmgSolver->gmgOperator()->setDimensionForSchwarzNeighborRelationship(dimensionForSchwarzNeighbors);
  }
  
  gmgSolver->gmgOperator()->setUseSchwarzDiagonalWeight(useWeightMatrixForSchwarz);
  
  gmgSolver->gmgOperator()->setDebugMode(runGMGOperatorInDebugMode);
  
  if (saveVisualizationOutput)
  {
    Teuchos::RCP<HDF5Exporter> exporter = Teuchos::rcp(new HDF5Exporter(solution->mesh(), "preconditionerError", "./") );
    gmgSolver->gmgOperator()->setFunctionExporter(exporter, errToPlot, errFunctionName);
  }
  
  return Teuchos::rcp(gmgSolver);
}

string smootherString(GMGOperator::SmootherChoice smoother)
{
  switch (smoother)
  {
  case GMGOperator::IFPACK_ADDITIVE_SCHWARZ:
    return "IfPack-Schwarz";
  case GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ:
    return "Camellia-Schwarz";
  case GMGOperator::NONE:
    return "None";
  case GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL:
    return "Block-Gauss-Seidel";
  case GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL:
    return "Point-Gauss-Seidel";
  case GMGOperator::BLOCK_JACOBI:
    return "Block-Jacobi";
  case GMGOperator::POINT_JACOBI:
    return "Point-Jacobi";
  default:
    return "Unknown";
  }
}

void initializeSolutionAndCoarseMesh(SolutionPtr &solution, MeshPtr &coarseMesh, IPPtr &graphNorm, double graphNormBeta, ProblemChoice problemChoice,
                                     int spaceDim, bool conformingTraces, bool useStaticCondensation, int numCells, int k, int delta_k, int k_coarse,
                                     int rootMeshNumCells, bool hOnly, bool useHierarchicalNeighborsForSchwarz, bool useZeroMeanConstraints,
                                     bool enhanceFieldsForH1TracesWhenConforming,
                                     Teuchos::RCP<NavierStokesVGPFormulation> &navierStokesFormulationFine,
                                     Teuchos::RCP<NavierStokesVGPFormulation> &navierStokesFormulationCoarse)
{
  BFPtr bf;
  BCPtr bc;
  RHSPtr rhs;
  MeshPtr mesh;
  map<int,int> coarseMeshTrialSpaceEnhancements;

  int rank = Teuchos::GlobalMPISession::getRank();
  
  double width = 1.0; // in each dimension
  vector<double> x0(spaceDim,0); // origin is the default

  VarPtr p; // pressure

  map<int,int> trialOrderEnhancements;
  FunctionPtr exactSolution;
  VarPtr varExact; // the variable of which we have the exact solution in exactSolution

  if (problemChoice == Poisson)
  {
    PoissonFormulation formulation(spaceDim, conformingTraces);

    bf = formulation.bf();
    
    rhs = RHS::rhs();
    FunctionPtr f = Function::constant(1.0);

    VarPtr q = formulation.q();
    rhs->addTerm( f * q );

    bc = BC::bc();
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    VarPtr phi_hat = formulation.phi_hat();
    bc->addDirichlet(phi_hat, boundary, Function::zero());
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      trialOrderEnhancements[formulation.phi()->ID()] = 1;
    }
    
    if (saveVisualizationOutput)
    {
      // we don't have an exact solution, so let's just plot phi instead...
      varExact = formulation.phi();
      exactSolution = Function::zero();
    }
  }
  else if (problemChoice == ConvectionDiffusion)
  {
    double epsilon = 1e-2;
    x0 = vector<double>(spaceDim,-1.0);

    FunctionPtr beta;
    FunctionPtr beta_x = Function::constant(1);
    FunctionPtr beta_y = Function::constant(2);
    FunctionPtr beta_z = Function::constant(3);
    if (spaceDim == 1)
      beta = beta_x;
    else if (spaceDim == 2)
      beta = Function::vectorize(beta_x, beta_y);
    else if (spaceDim == 3)
      beta = Function::vectorize(beta_x, beta_y, beta_z);
    ConvectionDiffusionFormulation formulation(spaceDim, conformingTraces, beta, epsilon);

    bf = formulation.bf();
    
    rhs = RHS::rhs();
    FunctionPtr f = Function::constant(1.0);

    VarPtr v = formulation.v();
    rhs->addTerm( f * v );

    bc = BC::bc();
    VarPtr uhat = formulation.uhat();
    VarPtr tc = formulation.tc();
    SpatialFilterPtr inflowX = SpatialFilter::matchingX(-1);
    SpatialFilterPtr inflowY = SpatialFilter::matchingY(-1);
    SpatialFilterPtr inflowZ = SpatialFilter::matchingZ(-1);
    SpatialFilterPtr outflowX = SpatialFilter::matchingX(1);
    SpatialFilterPtr outflowY = SpatialFilter::matchingY(1);
    SpatialFilterPtr outflowZ = SpatialFilter::matchingZ(1);
    FunctionPtr zero = Function::zero();
    FunctionPtr one = Function::constant(1);
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    if (spaceDim == 1)
    {
      bc->addDirichlet(tc, inflowX, -one);
      bc->addDirichlet(uhat, outflowX, zero);
    }
    if (spaceDim == 2)
    {
      bc->addDirichlet(tc, inflowX, -1*.5*(one-y));
      bc->addDirichlet(uhat, outflowX, zero);
      bc->addDirichlet(tc, inflowY, -2*.5*(one-x));
      bc->addDirichlet(uhat, outflowY, zero);
    }
    if (spaceDim == 3)
    {
      bc->addDirichlet(tc, inflowX, -1*.25*(one-y)*(one-z));
      bc->addDirichlet(uhat, outflowX, zero);
      bc->addDirichlet(tc, inflowY, -2*.25*(one-x)*(one-z));
      bc->addDirichlet(uhat, outflowY, zero);
      bc->addDirichlet(tc, inflowZ, -3*.25*(one-x)*(one-y));
      bc->addDirichlet(uhat, outflowZ, zero);
    }
    
//    if (rank==0) cout << "NOTE: TRYING SOMETHING IN ConvectionDiffusion's IP.\n";
//    
//    VarPtr tau = formulation.tau();
//    map<int,double> trialWeights; // leave empty for unit weights (default)
//    map<int,double> testL2Weights;
//    testL2Weights[v->ID()] = 1.0;
//    testL2Weights[tau->ID()] = 1.0 / epsilon;
//    graphNorm = bf->graphNorm(trialWeights,testL2Weights);
//    if (rank==0)
//    {
//      bf->printTrialTestInteractions();
//      cout << "RHS: " << rhs->linearTerm()->displayString() << endl;
//    }
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      trialOrderEnhancements[formulation.u()->ID()] = 1;
    }
  }
  else if (problemChoice == ConvectionDiffusionExperimental)
  {
    double epsilon = 1e-2;
    x0 = vector<double>(spaceDim,-1.0);

    FunctionPtr beta;
    FunctionPtr beta_x = Function::constant(1);
    FunctionPtr beta_y = Function::constant(2);
    FunctionPtr beta_z = Function::constant(3);
    if (spaceDim == 1)
      beta = beta_x;
    else if (spaceDim == 2)
      beta = Function::vectorize(beta_x, beta_y);
    else if (spaceDim == 3)
      beta = Function::vectorize(beta_x, beta_y, beta_z);

    FunctionPtr zero = Function::zero();
    FunctionPtr one = Function::constant(1);
    
    Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
    Space uhat_space = conformingTraces ? HGRAD : L2;
    Space vSpace = (spaceDim > 1) ? VECTOR_L2 : L2;
    
    // fields
    VarPtr u;
    VarPtr sigma;
    
    // traces
    VarPtr uhat, tc;
    
    // tests
    VarPtr v;
    VarPtr tau;
    
    VarFactoryPtr vf = VarFactory::varFactory();
    u = vf->fieldVar("u");
    sigma = vf->fieldVar("sigma", vSpace);
    
    TFunctionPtr<double> n = TFunction<double>::normal();
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    
    if (spaceDim > 1)
      uhat = vf->traceVar("uhat", u, uhat_space);
    else
      uhat = vf->fluxVar("uhat", u * (parity * Function::normal_1D()), uhat_space); // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
    
    
    if (spaceDim > 1)
      tc = vf->fluxVar("tc", (beta*u-sigma) * (n * parity));
    else
      tc = vf->fluxVar("tc", (beta*u-sigma) * (parity * Function::normal_1D()));
    
    v = vf->testVar("v", HGRAD);
    tau = vf->testVar("tau", tauSpace);
    
    bf = Teuchos::rcp( new BF(vf) );
    
    if (spaceDim==1)
    {
      // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
      bf->addTerm(sigma, tau);
      bf->addTerm(u * epsilon, tau->dx());
      bf->addTerm(-epsilon * uhat, tau);
      
      bf->addTerm(-beta*u + sigma, v->dx());
      bf->addTerm(tc, v);
    }
    else
    {
      // compared with the standard formulation, we multiply all tau terms by epsilon (tau doesn't enter RHS)
      bf->addTerm(sigma, tau);
      bf->addTerm(u * epsilon, tau->div());
      bf->addTerm(-epsilon * uhat, tau->dot_normal());
      
      bf->addTerm(-beta*u + sigma, v->grad());
      bf->addTerm(tc, v);
    }
    
    rhs = RHS::rhs();
    FunctionPtr f = Function::constant(1.0);
    
    rhs->addTerm( f * v );
    
    bc = BC::bc();
    SpatialFilterPtr inflowX = SpatialFilter::matchingX(-1);
    SpatialFilterPtr inflowY = SpatialFilter::matchingY(-1);
    SpatialFilterPtr inflowZ = SpatialFilter::matchingZ(-1);
    SpatialFilterPtr outflowX = SpatialFilter::matchingX(1);
    SpatialFilterPtr outflowY = SpatialFilter::matchingY(1);
    SpatialFilterPtr outflowZ = SpatialFilter::matchingZ(1);
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    if (spaceDim == 1)
    {
      bc->addDirichlet(tc, inflowX, -one);
      bc->addDirichlet(uhat, outflowX, zero);
    }
    if (spaceDim == 2)
    {
      bc->addDirichlet(tc, inflowX, -1*.5*(one-y));
      bc->addDirichlet(uhat, outflowX, zero);
      bc->addDirichlet(tc, inflowY, -2*.5*(one-x));
      bc->addDirichlet(uhat, outflowY, zero);
    }
    if (spaceDim == 3)
    {
      bc->addDirichlet(tc, inflowX, -1*.25*(one-y)*(one-z));
      bc->addDirichlet(uhat, outflowX, zero);
      bc->addDirichlet(tc, inflowY, -2*.25*(one-x)*(one-z));
      bc->addDirichlet(uhat, outflowY, zero);
      bc->addDirichlet(tc, inflowZ, -3*.25*(one-x)*(one-y));
      bc->addDirichlet(uhat, outflowZ, zero);
    }
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      trialOrderEnhancements[u->ID()] = 1;
    }
    
//    if (rank==0)
//    {
//      bf->printTrialTestInteractions();
//      cout << "RHS: " << rhs->linearTerm()->displayString() << endl;
//    }
  }
  else if (problemChoice == Stokes)
  {
    double mu = 1.0;
    StokesVGPFormulation formulation = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);

    p = formulation.p();

    bf = formulation.bf();
    graphNorm = bf->graphNorm();
    
    rhs = RHS::rhs();

    FunctionPtr cos_y = Teuchos::rcp( new Cos_y );
    FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
    FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
    FunctionPtr exp_z = Teuchos::rcp( new Exp_z );

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);

    FunctionPtr u1_exact, u2_exact, u3_exact, p_exact;

    if (spaceDim == 2)
    {
      // this one was in the Cockburn Kanschat LDG Stokes paper
      u1_exact = - exp_x * ( y * cos_y + sin_y );
      u2_exact = exp_x * y * sin_y;
      p_exact = 2.0 * exp_x * sin_y;
    }
    else
    {
      // this one is inspired by the 2D one
      u1_exact = - exp_x * ( y * cos_y + sin_y );
      u2_exact = exp_x * y * sin_y + exp_z * y * cos_y;
      u3_exact = - exp_z * (cos_y - y * sin_y);
      p_exact = 2.0 * exp_x * sin_y + 2.0 * exp_z * cos_y;
      // DEBUGGING:
//      u1_exact = Function::zero();
//      u2_exact = Function::zero();
//      u3_exact = x;
//      p_exact = Function::zero();
    }
    if (saveVisualizationOutput)
    {
      exactSolution = u1_exact;
      varExact = formulation.u(1);
      errFunctionName = "u1_error";
    }

    // to ensure zero mean for p, need the domain carefully defined:
    x0 = vector<double>(spaceDim,-1.0);

    width = 2.0;

    bc = BC::bc();
    // our usual way of adding in the zero mean constraint results in a negative eigenvalue
    // therefore, for now, we use a single-point BC
//    bc->addZeroMeanConstraint(formulation.p());
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    bc->addDirichlet(formulation.u_hat(1), boundary, u1_exact);
    bc->addDirichlet(formulation.u_hat(2), boundary, u2_exact);
    if (spaceDim==3) bc->addDirichlet(formulation.u_hat(3), boundary, u3_exact);

    FunctionPtr f1, f2, f3;
    if (spaceDim==2)
    {
      f1 = -p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
      f2 = -p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    }
    else
    {
      f1 = -p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy() + u1_exact->dz()->dz());
      f2 = -p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy() + u2_exact->dz()->dz());
      f3 = -p_exact->dz() + mu * (u3_exact->dx()->dx() + u3_exact->dy()->dy() + u3_exact->dz()->dz());
    }

    VarPtr v1 = formulation.v(1);
    VarPtr v2 = formulation.v(2);

    VarPtr v3;
    if (spaceDim==3) v3 = formulation.v(3);

    RHSPtr rhs = RHS::rhs();
    if (spaceDim==2)
      rhs->addTerm(f1 * v1 + f2 * v2);
    else
      rhs->addTerm(f1 * v1 + f2 * v2 + f3 * v3);
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      for (int d=0; d<spaceDim; d++)
        trialOrderEnhancements[formulation.u(d+1)->ID()] = 1;
    }
  }
  else if (problemChoice == LinearElasticity)
  {
//    cout << "LinearElasticity not yet supported!\n";
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "LinearElasticity not yet supported!");
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "only spaceDim = 2 is supported");
    
    double mu = 1.0, lambda = 1.0;
    LinearElasticityFormulation formulation = LinearElasticityFormulation::steadyFormulation(spaceDim, lambda, mu, conformingTraces);
    
    bf = formulation.bf();
    graphNorm = bf->graphNorm();
    
    const static double PI  = 3.141592653589793238462;
    
    FunctionPtr sin_pi_x = Teuchos::rcp( new Sin_ax(PI) );
    FunctionPtr sin_pi_y = Teuchos::rcp( new Sin_ay(PI) );
        
    FunctionPtr u1_exact, u2_exact, u3_exact;
    
    if (spaceDim == 2)
    {
      // u_x = sin (pi x) sin (pi y)
      // u_y = sin (pi x) sin (pi y)
      
      u1_exact = sin_pi_x * sin_pi_y;
      u2_exact = sin_pi_x * sin_pi_y;
    }
    else
    {
      // TODO: define an exact solution for 3D
    }
    
    x0 = vector<double>(spaceDim,0.0); // origin
    
    width = 1.0;
    
    bc = BC::bc();

    FunctionPtr zero = Function::zero();
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    bc->addDirichlet(formulation.u_hat(1), boundary, zero);
    bc->addDirichlet(formulation.u_hat(2), boundary, zero);
    if (spaceDim==3) bc->addDirichlet(formulation.u_hat(3), boundary, zero);
    
//    vector<FunctionPtr> f_vector(spaceDim, zero);
//    for (int i=1; i<= spaceDim; i++)
//    {
//      for (int j=1; j<= spaceDim; j++)
//      {
//        for (int k=1; k<= spaceDim; k++)
//        {
//          FunctionPtr u_k;
//          switch (k) {
//            case 1:
//              u_k = u1_exact;
//              break;
//            case 2:
//              u_k = u2_exact;
//              break;
//            case 3:
//              u_k = u3_exact;
//            default:
//              break;
//          }
//          for (int l=1; l<= spaceDim; l++)
//          {
//            FunctionPtr u_k_lj = u_k->grad()->spatialComponent(l)->grad()->spatialComponent(j);
//            double E_ijkl = formulation.E(i, j, k, l);
////            cout << i << ", " << j << ", " << k << ", " << l << ": ";
////            cout << -C_ijkl << " * " << u_k_lj->displayString() << endl;
//            if (E_ijkl == 0) f_vector[i-1] = f_vector[i-1] + zero;
//            else f_vector[i-1] = f_vector[i-1] -E_ijkl * u_k_lj;
////            cout << f_vector[i-1]->displayString() << endl;
//          }
//        }
//      }
//      
////      cout << "f[" << i << "]: " << f_vector[i-1]->displayString() << endl;
//    }
//    
//    FunctionPtr f = Function::vectorize(f_vector);

    FunctionPtr u_exact = (spaceDim == 2) ? Function::vectorize(u1_exact, u2_exact) : Function::vectorize(u1_exact, u2_exact, u3_exact);
    FunctionPtr f = formulation.forcingFunction(u_exact);
    
    if (saveVisualizationOutput)
    {
      exactSolution = u1_exact;
      varExact = formulation.u(1);
      errFunctionName = "u1_error";
    }
    
    VarPtr v1 = formulation.v(1);
    VarPtr v2 = formulation.v(2);
    
    VarPtr v3;
    if (spaceDim==3) v3 = formulation.v(3);
    
    rhs = formulation.rhs(f);

    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      for (int d=0; d<spaceDim; d++)
        trialOrderEnhancements[formulation.u(d+1)->ID()] = 1;
    }
  }
  else if (problemChoice == NavierStokes)
  {
    if (spaceDim != 2)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option for Navier-Stokes"); // we can add support for a 3D exact solution later...
    }
    
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({2.0,2.0}, {rootMeshNumCells, rootMeshNumCells}, {-0.5,0.0});
    MeshTopologyViewPtr coarseMeshTopo;
    if (!hOnly)
    {
      coarseMeshTopo = meshTopo;
    }
    int meshWidthCells = rootMeshNumCells;
    while (meshWidthCells < numCells)
    {
      vector<IndexType> activeCellIDs = meshTopo->getActiveCellIndicesGlobal(); // should match between coarseMesh and mesh
      IndexType nextCellIndexFine = meshTopo->cellCount();
      for (IndexType cellIndex : activeCellIDs)
      {
        CellTopoPtr cellTopo = meshTopo->getCell(cellIndex)->topology();
        RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
        meshTopo->refineCell(cellIndex, refPattern, nextCellIndexFine);
        nextCellIndexFine += refPattern->numChildren();
      }
      meshWidthCells *= 2;
      
      if (hOnly && (meshWidthCells == numCells))
      {
        // final iteration for h-multigrid: coarseMesh should be the parents of the most recently refined guys
        set<IndexType> cellsForCoarseMesh;
        cellsForCoarseMesh.insert(activeCellIDs.begin(),activeCellIDs.end());
        coarseMeshTopo = meshTopo->getView(cellsForCoarseMesh);
      }
    }
    
    // set up classical Kovasznay flow solution:
    double Re = 40.0;
    navierStokesFormulationFine = Teuchos::rcp(new NavierStokesVGPFormulation(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, meshTopo, k, delta_k)));
    navierStokesFormulationCoarse = Teuchos::rcp(new NavierStokesVGPFormulation(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, coarseMeshTopo, k_coarse, delta_k)));

    int k_low_order = 1;
    NavierStokesVGPFormulation lowestOrderForm(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, meshTopo, k_low_order, delta_k));
    
    FunctionPtr u1, u2, p;
    NavierStokesVGPFormulation::getKovasznaySolution(Re, u1, u2, p);
    
    mesh = navierStokesFormulationFine->solutionIncrement()->mesh();
    coarseMesh = navierStokesFormulationCoarse->solutionIncrement()->mesh();
    
    if (useZeroMeanConstraints)
    {
      navierStokesFormulationFine->addZeroMeanPressureCondition();
      lowestOrderForm.addZeroMeanPressureCondition();
      double p_mean = p->integrate(mesh);
      p = p - p_mean;
    }
    else
    {
      navierStokesFormulationFine->addPointPressureCondition({0.5,1.0});
      lowestOrderForm.addPointPressureCondition({0.5,1.0});
      double p_center = p->evaluate(0.5, 1.0);
      p = p - p_center;
    }
    
    FunctionPtr u = Function::vectorize({u1, u2});
    FunctionPtr forcingFunction = NavierStokesVGPFormulation::forcingFunctionSteady(spaceDim, Re, u, p);
    
    int kovasznayCubatureEnrichment = 20; // 20 is better than 10 for accurately measuring error on the coarser meshes.

    navierStokesFormulationFine->addInflowCondition(SpatialFilter::allSpace(), u);
    navierStokesFormulationFine->setForcingFunction(forcingFunction);
    
    lowestOrderForm.addInflowCondition(SpatialFilter::allSpace(), u);
    lowestOrderForm.setForcingFunction(forcingFunction);

    lowestOrderForm.solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    
    if (rank == 0) cout << "Navier-Stokes: taking three Newton steps on first-order mesh; using this as background flow.\n";
    for (int i=0; i<3; i++)
    {
      lowestOrderForm.solveAndAccumulate();
    }
    lowestOrderForm.solution()->projectFieldVariablesOntoOtherSolution(navierStokesFormulationFine->solution());
    
    navierStokesFormulationFine->solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    navierStokesFormulationFine->solutionIncrement()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option for Navier-Stokes"); // if it seems important, we can add support for this later...
    }
    
    graphNorm = navierStokesFormulationFine->bf()->graphNorm();
    
    solution = navierStokesFormulationFine->solutionIncrement();
    solution->setUseCondensedSolve(useStaticCondensation);
    
    if (saveVisualizationOutput)
    {
      exactSolution = u1;
      varExact = navierStokesFormulationFine->u(1);
      errFunctionName = "u1_error";
      FunctionPtr u1_background = Function::solution(varExact, navierStokesFormulationFine->solution());
      FunctionPtr u1_increment = Function::solution(varExact, navierStokesFormulationFine->solutionIncrement());
      // exact increment is (u1 - u1_background)
      errToPlot = u1_increment - (u1 - u1_background);
    }
    return; // return early for Navier-Stokes; we've already set up the Solution objects, etc.
  }

  int H1Order = k + 1;

  vector<double> dimensions;
  vector<int> elementCounts;
  
  for (int d=0; d<spaceDim; d++)
  {
    dimensions.push_back(width);
    elementCounts.push_back(rootMeshNumCells);
  }
  mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0, trialOrderEnhancements);
  
  int H1Order_coarse = k_coarse + 1;

  // now that we have mesh, add pressure constraint for Stokes (imposing zero at origin--want to aim for center of mesh)
  if (problemChoice == Stokes)
  {
    if (!useZeroMeanConstraints)
    {
      vector<double> origin(spaceDim,0);
      IndexType vertexIndex;
      if (!mesh->getTopology()->getVertexIndex(origin, vertexIndex))
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "origin vertex not found");
      }
      bc->addSpatialPointBC(p->ID(), 0.0, origin);
    }
    else
    {
      bc->addZeroMeanConstraint(p);
    }
  }

  int meshWidthCells = rootMeshNumCells;
  while (meshWidthCells < numCells)
  {
    set<IndexType> activeCellIDs = mesh->getActiveCellIDsGlobal(); // should match between coarseMesh and mesh
    mesh->hRefine(activeCellIDs);
    meshWidthCells *= 2;
    
    if (hOnly && (meshWidthCells == numCells))
    {
      // final iteration for h-multigrid: coarseMesh should be the parents of the most recently refined guys
      set<IndexType> cellsForCoarseMesh;
      cellsForCoarseMesh.insert(activeCellIDs.begin(),activeCellIDs.end());
      MeshTopologyViewPtr coarseMeshTopo = mesh->getTopology()->getView(cellsForCoarseMesh);
      MeshPartitionPolicyPtr inducedPartitionPolicy = MeshPartitionPolicy::inducedPartitionPolicyFromRefinedMesh(coarseMeshTopo, mesh);
      coarseMesh = Teuchos::rcp( new Mesh(coarseMeshTopo, mesh->bilinearForm(), H1Order_coarse, delta_k, map<int,int>(), map<int,int>(), inducedPartitionPolicy) ) ;
    }
  }

  if (meshWidthCells != numCells)
  {
    int rank = Teuchos::GlobalMPISession::getRank();
    if (rank == 0)
    {
      cout << "Warning: may have overrefined mesh; mesh has width " << meshWidthCells << ", not " << numCells << endl;
    }
  }

  // coarse and fine mesh share a MeshTopology.  This means that they should not be further refined (they won't be, here)
  if (!hOnly)
  {
    // ensure that coarse and fine mesh share a common partitioning
    MeshPartitionPolicyPtr inducedPartitionPolicy = MeshPartitionPolicy::inducedPartitionPolicy(mesh);
    coarseMesh = Teuchos::rcp(new Mesh(mesh->getTopology(), bf, H1Order_coarse, delta_k, trialOrderEnhancements,
                                       map<int,int>(), inducedPartitionPolicy));
  }
  
  if (graphNorm == Teuchos::null) // if set previously, honor that...
    graphNorm = bf->graphNorm();

  solution = Solution::solution(mesh, bc, rhs, graphNorm);
  solution->setUseCondensedSolve(useStaticCondensation);
  solution->setZMCsAsGlobalLagrange(false); // fine grid solution shouldn't impose ZMCs (should be handled in coarse grid solve)
  
  if ((saveVisualizationOutput) && (varExact != Teuchos::null))
  {
    FunctionPtr var_soln = Function::solution(varExact, solution);
    // exact increment is (u1 - u1_background)
    errToPlot = exactSolution - var_soln;
  }
  
//  {
//    cout << "DEBUGGING: outputting sin_pi_x_sin_pi_y to disk at /tmp/.\n";
//    const static double PI  = 3.141592653589793238462;
//    
//    FunctionPtr sin_pi_x = Teuchos::rcp( new Sin_ax(PI) );
//    FunctionPtr sin_pi_y = Teuchos::rcp( new Sin_ay(PI) );
//    
//    HDF5Exporter::exportFunction("/tmp/", "sin_pi_x_sin_pi_y", sin_pi_x * sin_pi_y, mesh);
//  }
}

void run(ProblemChoice problemChoice, int &iterationCount, int spaceDim, int numCells, int k, int delta_k, int k_coarse, bool conformingTraces,
         bool useStaticCondensation, bool precondition, bool schwarzOnly, GMGOperator::SmootherChoice smootherType,
         int schwarzOverlap, int dimensionForSchwarzNeighbors, GMGOperator::FactorType schwarzBlockFactorization, int schwarzLevelOfFill, double schwarzFillRatio,
         Solver::SolverChoice coarseSolverChoice, double cgTol, int cgMaxIterations, int AztecOutputLevel,
         bool reportTimings, double &solveTime, double &error, bool reportEnergyError, int numCellsRootMesh,
         bool hOnly, bool useHierarchicalNeighborsForSchwarz, bool useZeroMeanConstraints,
         bool writeAndExit, GMGOperator::SmootherApplicationType comboType, double smootherWeight, bool useWeightMatrixForSchwarz,
         bool enhanceFieldsForH1TracesWhenConforming, double graphNormBeta)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (k_coarse == -1)
  {
    if (hOnly)
      k_coarse = k;
    else
    {
      // then set k_coarse to be ceil(k/2) unless k == 1, in which case do k=0
      if (k == 1)
        k_coarse = 0;
      else
        k_coarse = ceil(k / 2.0);
    }
  }

//  if ((numCellsRootMesh == -1) && hOnly)
//  {
//    // then use a single level of h-coarsening as the root mesh.
//    numCellsRootMesh = numCells / 2;
//    if (numCellsRootMesh == 0)
//    {
//      cout << "Too few cells in root mesh.  Aborting.\n";
//      exit(1);
//    }
//    int rank = Teuchos::GlobalMPISession::getRank();
//    if (rank == 0)
//    {
//      cout << "Setting numCellsRootMesh = " << numCellsRootMesh << endl;
//    }
//  }
//  else if (numCellsRootMesh == -1)
  if (numCellsRootMesh == -1)
  {
    int evenDivisor = numCells;
    
    while ((evenDivisor/2) * 2 == evenDivisor)
    {
      evenDivisor /= 2;
    }
    
    numCellsRootMesh = numCells;
    if (problemChoice == Stokes)
    {
      // need origin to be a vertex in root mesh; this is true if the number of root cells is even
      numCellsRootMesh = (evenDivisor % 2 == 0) ? evenDivisor : evenDivisor * 2;
    }
    else
    {
      numCellsRootMesh = max(evenDivisor,1);
    }
  }
  if ((numCellsRootMesh == numCells) && hOnly)
  {
    cout << "Too few cells in root mesh.  Aborting.\n";
    exit(1);
  }
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time initializationTimer(Comm);

  SolutionPtr solution;
  MeshPtr k0Mesh;
  IPPtr graphNorm;
  Teuchos::RCP<NavierStokesVGPFormulation> navierStokesFormFine, navierStokesFormCoarse; // NULL if not doing Navier-Stokes
  initializeSolutionAndCoarseMesh(solution, k0Mesh, graphNorm, graphNormBeta, problemChoice, spaceDim, conformingTraces, useStaticCondensation,
                                  numCells, k, delta_k, k_coarse, numCellsRootMesh, hOnly, useHierarchicalNeighborsForSchwarz,
                                  useZeroMeanConstraints, enhanceFieldsForH1TracesWhenConforming,
                                  navierStokesFormFine, navierStokesFormCoarse);
  
  MeshPtr mesh = solution->mesh();
  BCPtr bc = solution->bc();

  int coarseElements = k0Mesh->numActiveElements(), fineElements = mesh->numActiveElements();
  int fineDofs = mesh->numGlobalDofs(), coarseDofs = k0Mesh->numGlobalDofs();
  if (rank == 0)
  {
    cout << "fine mesh has " << fineElements << " active elements and " << fineDofs << " degrees of freedom.\n";
    if (!schwarzOnly && precondition)
      cout << "coarse mesh has " << coarseElements << " active elements and " << coarseDofs << " degrees of freedom.\n";
  }
  
  
//  {
//    // DEBUGGING
//    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
//    if (rank==0)
//      minRule->printGlobalDofInfo();
//  }
  
//  if (hOnly)
//  {
//    // then replace the k0Mesh with the h-coarsened mesh:
//    Teuchos::ParameterList pl;
//    pl.set("kCoarse", 0);
//    pl.set("delta_k", 1); // this should not really matter in this context
//    pl.set("jumpToCoarsePolyOrder", false);
//    vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, pl); // not the most efficient way to do this, but it should work...
//
//    k0Mesh = meshesCoarseToFine[meshesCoarseToFine.size()-2];
////    MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
////    MeshTopologyPtr coarseMeshTopo = meshTopo->getRootMeshTopology();
////    int H1OrderP0 = k + 1;
////    k0Mesh = Teuchos::rcp( new Mesh(coarseMeshTopo, k0Mesh->bilinearForm(), H1OrderP0, delta_k) );
//  }

  double initializationTime = initializationTimer.ElapsedTime();
  int numCoarseGlobalDofs = k0Mesh->numGlobalDofs();
  int numFineGlobalDofs = solution->mesh()->numGlobalDofs();
  if (narrateSolution && (rank==0))
  {
    cout << "Solution (" << numFineGlobalDofs << " fine global dofs) and k0 mesh (";
    cout << numCoarseGlobalDofs << " coarse global dofs) initialized in " << initializationTime << " seconds.\n";
  }
  
  // to be fairer to preconditioners whose construction is cheap, we include construction time in the solve time (this is a change, made 7-7-15)
  Epetra_Time solveTimer(Comm);
  
  Teuchos::RCP<Solver> solver;
  if (!precondition)
  {
    solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition, schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio) );
    ((AztecSolver*) solver.get())->setAztecOutputLevel(AztecOutputLevel);
  }
  else if (schwarzOnly)
  {
    if (smootherType==GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ)
    {
      solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap, precondition, schwarzBlockFactorization,
                                             schwarzLevelOfFill, schwarzFillRatio, mesh, solution->getDofInterpreter(), hOnly) );
    }
    else
    {
      solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition, schwarzBlockFactorization,
                                             schwarzLevelOfFill, schwarzFillRatio) );
    }
    ((AztecSolver*) solver.get())->setAztecOutputLevel(AztecOutputLevel);
  }
  else
  {
    solver = initializeGMGSolver(solution, AztecOutputLevel, smootherType,
                                 schwarzOverlap, dimensionForSchwarzNeighbors, schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio,
                                 useStaticCondensation, hOnly, useHierarchicalNeighborsForSchwarz, k0Mesh, coarseSolverChoice,
                                 cgMaxIterations, cgTol, comboType,
                                 smootherWeight, useWeightMatrixForSchwarz);
  }

//  if (problemChoice==Stokes) {
//    if (rank==0) cout << "Writing fine Stokes matrix to /tmp/A_stokes.dat.\n";
//    solution->setWriteMatrixToFile(true, "/tmp/A_stokes.dat");
//  }

  int result;
  
  if (problemChoice != NavierStokes)
  {
    result = solution->solve(solver);
  }
  else
  {
//    result = 0;
//    if (rank==0) cout << "Navier-Stokes: doing two Newton steps and recording the results for the last\n";
//    
//    navierStokesFormFine->setSolver(solver);
//    result = navierStokesFormFine->solveAndAccumulate();
//    
//    solver = initializeGMGSolver(solution, AztecOutputLevel, smootherType,
//                                 schwarzOverlap, schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio,
//                                 useStaticCondensation, hOnly, useHierarchicalNeighborsForSchwarz, k0Mesh, coarseSolverChoice,
//                                 cgMaxIterations, cgTol, comboType,
//                                 smootherWeight, useWeightMatrixForSchwarz);
    navierStokesFormFine->setSolver(solver);
    result = navierStokesFormFine->solveAndAccumulate(); // second step has non-zero background flow (implying variable coefficients)
  }

  solveTime = solveTimer.ElapsedTime();

  if (result == 0)
  {
    if (!precondition)
    {
      iterationCount = ((AztecSolver *) solver.get())->iterationCount();
    }
    else if (schwarzOnly)
    {
      iterationCount = ((AztecSolver *) solver.get())->iterationCount();
    }
    else
    {
      iterationCount = ((GMGSolver *) solver.get())->iterationCount();
    }
  }
  else
  {
    iterationCount = -1;
  }

  if (reportTimings) solution->reportTimings();
  solution->clearComputedResiduals(); // force recomputation
  double energyErrorTotal = solution->energyErrorTotal(); // we now record energy error even if we don't print to console
//  double energyErrorTotal = reportEnergyError ? solution->energyErrorTotal() : -1;

  GMGSolver* fineSolver = dynamic_cast<GMGSolver*>(solver.get());
  if (fineSolver != NULL)
  {
    if (rank==0) cout << "************   Fine GMG Solver, timings   *************\n";
    fineSolver->gmgOperator()->reportTimingsSumOfOperators(StatisticChoice::MAX);

//    fineSolver->gmgOperator()->reportTimings();

    GMGSolver* coarseSolver = dynamic_cast<GMGSolver*>(fineSolver->gmgOperator()->getCoarseSolver().get());
    if (coarseSolver != NULL)
    {
      if (rank==0) cout << "\n************   Coarse GMG Solver, timings   *************\n";
      coarseSolver->gmgOperator()->reportTimingsSumOfOperators(StatisticChoice::MAX);
      vector<int> iterationCountLog = coarseSolver->getIterationCountLog();
      if (rank==0) Camellia::print("coarseSolver iteration counts:",iterationCountLog);
      double totalIterationCount = 0;
      for (int i=0; i<iterationCountLog.size(); i++)
      {
        totalIterationCount += iterationCountLog[i];
      }
      if (rank==0) cout << "Average coarse solver iteration count: " << totalIterationCount / iterationCountLog.size() << "\n\n";
    }
  }

  GlobalIndexType numFluxDofs = mesh->numFluxDofs();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  if ((rank==0) && reportEnergyError)
  {
    cout << "Mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
    cout << numGlobalDofs << " total dofs, including fields).\n";
    cout << "Energy error: " << energyErrorTotal << endl;
  }
  error = energyErrorTotal; // in future, might want to compute L^2 norm of error instead of energy error, in cases where we know exact solution

  if (writeAndExit)
  {
    Teuchos::RCP< Epetra_CrsMatrix > A = solution->getStiffnessMatrix();
    Teuchos::RCP< Epetra_CrsMatrix > M;
    if (schwarzOnly)
    {
      // if (rank==0) cout << "writeAndExit not yet supported for schwarzOnly.\n";
      
      if (rank==0) cout << "writing A to A_fine.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("A_fine.dat",*A, NULL, NULL, false);
      
      Teuchos::RCP< Epetra_CrsMatrix > S = ((AztecSolver *) solver.get())->getPreconditionerMatrix(solution->getPartitionMap());
      if (rank==0) cout << "writing Schwarz preconditioner to S.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("S.dat",*S, NULL, NULL, false);
    }
    else
    {
      Teuchos::RCP<GMGOperator> op = ((GMGSolver*)solver.get())->gmgOperator();
      if (rank==0) cout << "writing op to op.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("op.dat",*op->getMatrixRepresentation(), NULL, NULL, false);

      Teuchos::RCP< Epetra_CrsMatrix > A_coarse = op->getCoarseStiffnessMatrix();
      if (rank==0) cout << "writing A_coarse to A_coarse.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("A_coarse.dat",*A_coarse, NULL, NULL, false);

      Teuchos::RCP< Epetra_CrsMatrix > P = op->getProlongationOperator();
      if (rank==0) cout << "writing P to P.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("P.dat",*P, NULL, NULL, false);

      if (rank==0) cout << "writing A to A_fine.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("A_fine.dat",*A, NULL, NULL, false);

      if (rank==0) cout << "writing smoother to S.dat.\n";
      Teuchos::RCP< Epetra_CrsMatrix > S = op->getSmootherAsMatrix();
      EpetraExt::RowMatrixToMatrixMarketFile("S.dat",*S, NULL, NULL, false);

      if (op->getSmootherWeightVector() != Teuchos::null)
      {
        if (rank==0) cout << "writing smoother weight vector to w_vector.dat\n";
        Teuchos::RCP<Epetra_MultiVector> w = op->getSmootherWeightVector();
        EpetraExt::MultiVectorToMatrixMarketFile("w_vector.dat", *w, NULL, NULL, false);
      }
      
      // for now, we just do this on rank 0.  For a big mesh, we might want to distribute this
      set<GlobalIndexType> myCellIndices = mesh->globalDofAssignment()->cellsInPartition(-1);
      vector<GlobalIndexTypeToCast> myCellIndicesVector(myCellIndices.begin(),myCellIndices.end());
      Epetra_MpiComm Comm(MPI_COMM_WORLD);
      
      Epetra_Map elemMap(-1, myCellIndicesVector.size(), &myCellIndicesVector[0], 0, Comm);
      Epetra_CrsMatrix E(::Copy, elemMap, 0);
      
      for (GlobalIndexType cellIndex : myCellIndices)
      {
        CellPtr cell = mesh->getTopology()->getCell(cellIndex);
        vector<CellPtr> neighbors = cell->getNeighbors(mesh->getTopology());
        neighbors.push_back(cell); // for connectivity, cell counts as its own neighbor
        
        vector<double> values(neighbors.size(),1.0);
        vector<GlobalIndexTypeToCast> neighborIndices;
        for (CellPtr neighbor : neighbors)
        {
          neighborIndices.push_back(neighbor->cellIndex());
        }
        E.InsertGlobalValues(cellIndex, values.size(), &values[0], &neighborIndices[0]);
      }
      E.FillComplete();
      EpetraExt::RowMatrixToMatrixMarketFile("E.dat",E, NULL, NULL, false);
      if (rank==0) cout << "wrote fine mesh element connectivity matrix to E.dat.\n";
      
      {
      // some debugging stuff:
//        Epetra_CrsMatrix identity(::Copy, A->RowMap(), 0);
        Epetra_CrsMatrix identity(::Copy, A->RowMap(), A->ColMap(), 0);
        int myRows = identity.RowMap().NumMyElements();
        int myCols = identity.ColMap().NumMyElements();
        FieldContainer<GlobalIndexTypeToCast> colLIDs(myCols);
        for (int LID=0; LID<myCols; LID++)
        {
          colLIDs[LID] = LID;
        }
        for (int LID=0; LID<myRows; LID++)
        {
          FieldContainer<double> myData(myCols);
          for (int j=0; j<myCols; j++)
          {
            if (j==LID) myData(j) = 1.0;
            else myData(j) = 0.0;
          }
          identity.InsertMyValues(LID, myCols, &myData[0], &colLIDs[0]);
//          cout << "on rank " << rank << ", inserting 1.0 at (" << myGIDs[LID] << "," << myGIDs[LID] << ")\n";
        }
        identity.FillComplete();
        
        if (rank==0) cout << "writing identity to I.dat.\n";
        EpetraExt::RowMatrixToMatrixMarketFile("I.dat",identity, NULL, NULL, false);
        op->setFineStiffnessMatrix(&identity);

        if (rank==0) cout << "writing smoother for identity to W.dat.\n";
        Teuchos::RCP< Epetra_CrsMatrix > W = op->getSmootherAsMatrix();
        EpetraExt::RowMatrixToMatrixMarketFile("W.dat",*W, NULL, NULL, false);
      }
      
      return;
    }
  }
  
  if (saveVisualizationOutput)
  {
    static int exportOrdinal = 0;
    ostringstream exporterName;
    exporterName << "preconditionerSolution_" << exportOrdinal++;
    
    Teuchos::RCP<HDF5Exporter> exporter = Teuchos::rcp(new HDF5Exporter(solution->mesh(), exporterName.str(), "./") );
    exporter->exportSolution(solution);
    
  }
  
//  if (rank==0) cout << "NOTE: Exported solution for debugging.\n";
//  HDF5Exporter::exportSolution("/tmp/", "testSolution", solution);
//
//  solution->solve();
//  if (rank==0) cout << "NOTE: Exported direct solution for debugging.\n";
//  HDF5Exporter::exportSolution("/tmp/testSolution", "testSolution_direct", solution);
//  energyErrorTotal = solution->energyErrorTotal();
//  if (rank==0) cout << "Direct solution has energy error " << energyErrorTotal << endl;
}

void runMany(ProblemChoice problemChoice, int spaceDim, int delta_k, int minCells,
             bool conformingTraces, bool useStaticCondensation,
             GMGOperator::FactorType schwarzBlockFactorization, int schwarzLevelOfFill, double schwarzFillRatio,
             Solver::SolverChoice coarseSolverChoice,
             double cgTol, int cgMaxIterations, int aztecOutputLevel, RunManyPreconditionerChoices preconditionerChoices,
             int k, int k_coarse, int overlapLevel, int dimensionForSchwarzNeighbors, int numCellsRootMesh, bool reportTimings,
             bool hOnly, bool useHierarchicalNeighborsForSchwarz,
             int maxCells, bool useZeroMeanConstraints, GMGOperator::SmootherApplicationType comboType, double smootherWeight,
             bool useWeightMatrixForSchwarz, bool enhanceFieldsForH1TracesWhenConforming, double graphNormBeta)
{
  int rank = Teuchos::GlobalMPISession::getRank();

  string problemChoiceString;
  switch (problemChoice)
  {
  case Poisson:
    problemChoiceString = "Poisson";
      break;
    case ConvectionDiffusion:
      problemChoiceString = "ConvectionDiffusion";
      break;
      
    case ConvectionDiffusionExperimental:
      problemChoiceString = "ConvectionDiffusionExperimental";
      break;
    case LinearElasticity:
      problemChoiceString = "LinearElasticity";
      break;
  case Stokes:
    problemChoiceString = "Stokes";
    break;
  case NavierStokes:
    problemChoiceString = "Navier-Stokes";
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled problem choice");
    break;
  }

  string preconditionerChoiceString;
  switch (preconditionerChoices)
  {
  case All:
    preconditionerChoiceString = "All";
    break;
  case AllGMG:
    preconditionerChoiceString = "AllGMG";
    break;
  case AllSchwarz:
    preconditionerChoiceString = "AllSchwarz";
    break;
  case DontPrecondition:
    preconditionerChoiceString = "NoPrecondition";
    break;
  case GMGAlgebraicSchwarz:
    preconditionerChoiceString = "GMGAlgebraicSchwarz";
    break;
  case GMGGeometricSchwarz:
    preconditionerChoiceString = "GMGGeometricSchwarz";
    break;
  case GMGBlockJacobi:
    preconditionerChoiceString = "GMGBlockJacobi";
    break;
  case GMGPointJacobi:
    preconditionerChoiceString = "GMGPointJacobi";
    break;
  case GMGBlockSymmetricGaussSeidel:
    preconditionerChoiceString = "GMGBlockGaussSeidel";
    break;
  case GMGPointSymmetricGaussSeidel:
    preconditionerChoiceString = "GMGPointGaussSeidel";
    break;
  case AlgebraicSchwarz:
    preconditionerChoiceString = "AlgebraicSchwarz";
    break;
  case GeometricSchwarz:
    preconditionerChoiceString = "GeometricSchwarz";
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled preconditioner choice subset");
    break;
  }

  vector<bool> preconditionValues;

  vector<bool> schwarzOnly_maxChoices;
  vector<GMGOperator::SmootherChoice> smootherChoices;

  switch (preconditionerChoices)
  {
  case DontPrecondition:
    preconditionValues.push_back(false);
    schwarzOnly_maxChoices.push_back(false);
    smootherChoices.push_back(GMGOperator::NONE);
    break;
  case GMGAlgebraicSchwarz:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(false);
    smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    break;
  case GMGGeometricSchwarz:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(false);
    smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
    break;
  case GMGBlockJacobi:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(false);
    smootherChoices.push_back(GMGOperator::BLOCK_JACOBI);
    break;
  case GMGPointJacobi:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(false);
    smootherChoices.push_back(GMGOperator::POINT_JACOBI);
    break;
  case GMGBlockSymmetricGaussSeidel:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(false);
    smootherChoices.push_back(GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL);
    break;
  case GMGPointSymmetricGaussSeidel:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(false);
    smootherChoices.push_back(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
    break;
  case AlgebraicSchwarz:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(true);
    smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    break;
  case GeometricSchwarz:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(true);
    smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
    break;
  case AllGMG:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(false);
    smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
//      smootherChoices.push_back(GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL);
//    smootherChoices.push_back(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
//      smootherChoices.push_back(GMGOperator::BLOCK_JACOBI);
//    smootherChoices.push_back(GMGOperator::POINT_JACOBI);
    break;
  case AllSchwarz:
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(true);
    smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
    break;
  case All:
    preconditionValues.push_back(false);
    preconditionValues.push_back(true);
    schwarzOnly_maxChoices.push_back(false);
    schwarzOnly_maxChoices.push_back(true);
    smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
//      smootherChoices.push_back(GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL);
//    smootherChoices.push_back(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
//      smootherChoices.push_back(GMGOperator::BLOCK_JACOBI);
//    smootherChoices.push_back(GMGOperator::POINT_JACOBI);
    break;
  }

  vector<int> kValues;
  if (k == -1)
  {
    if (k_coarse < 1)
      kValues.push_back(1);
    if (k_coarse < 2)
      kValues.push_back(2);
    if (spaceDim < 3) kValues.push_back(4);
    if (spaceDim < 2) kValues.push_back(8);
    if (spaceDim < 2) kValues.push_back(16);
    if ((kValues.size() == 0) && !hOnly)
      kValues = {k_coarse + 1};
    else if ((kValues.size() == 0) && hOnly)
      kValues = {k_coarse};
  }
  else
  {
    kValues.push_back(k);
  }

  vector<int> numCellsValues;
  int numCells = minCells;
  while (numCells <= maxCells)
  {
    // want to do as many as we can with just one cell per processor
    numCellsValues.push_back(numCells);
    numCells *= 2;
  }

  ostringstream results;
  results << "Preconditioner\tSmoother\tOverlap\tnum_cells\tmesh_width\tk\tIterations\tSolve_time\tError\n";

  for (vector<bool>::iterator preconditionChoiceIt = preconditionValues.begin(); preconditionChoiceIt != preconditionValues.end(); preconditionChoiceIt++)
  {
    bool precondition = *preconditionChoiceIt;

    vector<bool> schwarzOnlyValues;
    vector<GMGOperator::SmootherChoice> smootherChoiceValues;
    if (precondition)
    {
      schwarzOnlyValues = schwarzOnly_maxChoices;
      smootherChoiceValues = smootherChoices;
    }
    else
    {
      // schwarzOnly and smootherChoice ignored; just use one of each
      schwarzOnlyValues.push_back(false);
      smootherChoiceValues.push_back(GMGOperator::NONE);
    }
    for (vector<bool>::iterator schwarzOnlyChoiceIt = schwarzOnlyValues.begin(); schwarzOnlyChoiceIt != schwarzOnlyValues.end(); schwarzOnlyChoiceIt++)
    {
      bool schwarzOnly = *schwarzOnlyChoiceIt;
      for (vector<GMGOperator::SmootherChoice>::iterator smootherIt = smootherChoiceValues.begin(); smootherIt != smootherChoiceValues.end(); smootherIt++)
      {
        GMGOperator::SmootherChoice smoother = *smootherIt;

        vector<int> overlapValues;
        if (precondition && ((smoother==GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ) || (smoother==GMGOperator::IFPACK_ADDITIVE_SCHWARZ)))
        {
          if (overlapLevel == -1)
          {
            overlapValues.push_back(0);
            overlapValues.push_back(1);
            if (spaceDim < 3) overlapValues.push_back(2);
          }
          else
          {
            overlapValues.push_back(overlapLevel);
          }
        }
        else
        {
          overlapValues.push_back(0);
        }

        // smoother choice description
        string S_str = smootherString(smoother);
        string M_str; // preconditioner descriptor for output
        if (!precondition)
        {
          M_str = "None";
        }
        else
        {
          if (schwarzOnly)
          {
            M_str = S_str;
            S_str = "-"; // no smoother
          }
          else
          {
            M_str = "GMG";
          }
        }

        for (vector<int>::iterator overlapValueIt = overlapValues.begin(); overlapValueIt != overlapValues.end(); overlapValueIt++)
        {
          int overlapValue = *overlapValueIt;
          for (vector<int>::iterator numCellsValueIt = numCellsValues.begin(); numCellsValueIt != numCellsValues.end(); numCellsValueIt++)
          {
            int numCells1D = *numCellsValueIt;
            for (vector<int>::iterator kValueIt = kValues.begin(); kValueIt != kValues.end(); kValueIt++)
            {
              int k = *kValueIt;

              int iterationCount;
              bool reportEnergyError = false;
              double solveTime, error;
              bool writeAndExit = false; // not supported for runMany (since it always writes to the same disk location)
              run(problemChoice, iterationCount, spaceDim, numCells1D, k, delta_k, k_coarse, conformingTraces,
                  useStaticCondensation, precondition, schwarzOnly, smoother, overlapValue, dimensionForSchwarzNeighbors,
                  schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio, coarseSolverChoice,
                  cgTol, cgMaxIterations, aztecOutputLevel, reportTimings, solveTime, error,
                  reportEnergyError, numCellsRootMesh, hOnly, useHierarchicalNeighborsForSchwarz, useZeroMeanConstraints, writeAndExit, comboType,
                  smootherWeight, useWeightMatrixForSchwarz, enhanceFieldsForH1TracesWhenConforming, graphNormBeta);

              int numCells = pow((double)numCells1D, spaceDim);

              ostringstream thisResult;

              thisResult << M_str << "\t" << S_str << "\t" << overlapValue << "\t" << numCells << "\t";
              thisResult << numCells1D << "\t" << k << "\t" << iterationCount << "\t" << solveTime;
              thisResult << "\t" << error << endl;

              if (rank==0) cout << thisResult.str();

              results << thisResult.str();
            }
          }
        }
      }
      if (rank==0) cout << results.str(); // output results so far
    }
  }

  if (rank == 0)
  {
    ostringstream filename;
    filename << problemChoiceString << "Driver" << spaceDim << "D_";
    if ((preconditionerChoiceString == "GMGGeometricSchwarz") && !useWeightMatrixForSchwarz)
    {
      filename << "GMGGeometricSchwarzUnweighted";
    }
    else if ((preconditionerChoiceString == "GMGGeometricSchwarz") && useWeightMatrixForSchwarz)
    {
      filename << "GMGGeometricSchwarzWEIGHTED";
    }
    else
    {
      filename << preconditionerChoiceString;
    }
    if (schwarzBlockFactorization != GMGOperator::Direct)
      filename << "_schwarzFactorization_" << getFactorizationTypeString(schwarzBlockFactorization);
    if (overlapLevel != -1)
    {
      filename << "_overlap" << overlapLevel;
    }
    if (dimensionForSchwarzNeighbors != -1)
    {
      filename << "_schwarzNeighborDim" << dimensionForSchwarzNeighbors;
    }
    if (k != -1)
    {
      filename << "_k" << k;
    }
    filename << "_deltak" << delta_k;
    if (k_coarse != 0)
      filename << "_kCoarse" << k_coarse;

    if (hOnly)
    {
      filename << "_hOnly";
      if (useHierarchicalNeighborsForSchwarz)
        filename << "TruncatedNeighbors";
      else
        filename << "StandardNeighbors";
    }
    
    // if coarse solver is not direct, then include in the file name:
    if ((coarseSolverChoice != Solver::KLU) && (coarseSolverChoice != Solver::MUMPS) && (coarseSolverChoice != Solver::SuperLUDist))
      filename << "_coarseSolver_" << Solver::solverChoiceString(coarseSolverChoice);
    if (useStaticCondensation)
      filename << "_withStaticCondensation";
    if (conformingTraces)
      filename << "_conformingTraces";
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
      filename << "_enhancedFields";
    if (comboType == GMGOperator::MULTIPLICATIVE)
      filename << "_multiplicative";
    if (graphNormBeta != 1.0)
      filename << "_graphNormBeta" << graphNormBeta;
    if (cgTol != 1e-10)
      filename << "_cgTol" << cgTol;
    filename << "_results.dat";
    ofstream fout(filename.str().c_str());
    fout << results.str();
    fout.close();
    cout << "Wrote results to " << filename.str() << ".\n";
  }
}

int main(int argc, char *argv[])
{
#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
  cout << "NOTE: enabling floating point exceptions for divide by zero.\n";
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, 0);
  int rank = Teuchos::GlobalMPISession::getRank();

  runGMGOperatorInDebugMode = false;

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  int k = -1; // poly order for field variables (-1 for a range of values)
  int k_coarse = -1; // poly order for field variables on the coarse mesh
  int delta_k = -1;   // test space enrichment; -1 for default detection (defaults to spaceDim)

  bool conformingTraces = false;
  bool precondition = true;

  int numCells = 2;
  int numCellsRootMesh = -1;

  maxDofsForKLU = 2000; // used when defining coarsest solve on 3-level solver -- will use SuperLUDist if not KLU
  coarseCGTol = 1e-10;
  coarseMaxIterations = 2000;

  int AztecOutputLevel = 1;
  int cgMaxIterations = 25000;
  int schwarzOverlap = -1;

  double smootherWeight = -1.0;
  
  int spaceDim = 1;

  bool useCondensedSolve = false;

  bool useWeightMatrixForSchwarz = false;
  
  string smootherChoiceStr = "Camellia-Schwarz";

  bool schwarzOnly = false;

  double cgTol = 1e-10;

  double fillRatio = 5;
  int levelOfFill = 2;

  bool runAutomatic = false;

  bool reportTimings = false;
  
  int dimensionForSchwarzNeighbors = -1;
  
  bool enhanceFieldsForH1TracesWhenConforming = false; // new 10-27-15

  bool hOnly = false;
  
  bool useHierarchicalNeighborsForSchwarz = false; // new 12-22-15
  
  narrateSolution = false;
  narrateCoarseSolution = false;

  bool useZeroMeanConstraints = false;

  bool writeAndExit = false;

  string schwarzFactorizationTypeString = "Direct";

  string problemChoiceString = "Poisson";

  string coarseSolverChoiceString = "GMG";

  string runManySubsetString = "All";
  
  double graphNormBeta = 1.0;

  int runManyMinCells = 2;
  int maxCells = -1;
  
  bool additiveComboType = false;

  cmdp.setOption("problem",&problemChoiceString,"problem choice: Poisson, ConvectionDiffusion, LinearElasticity, Stokes, Navier-Stokes");

  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");

  cmdp.setOption("coarseSolver", &coarseSolverChoiceString, "coarse solver choice: KLU, MUMPS, SuperLUDist, SimpleML");
  cmdp.setOption("coarsePolyOrder", &k_coarse, "polynomial order for field variables on coarse grid");

  cmdp.setOption("combineAdditive", "combineMultiplicative", &additiveComboType);
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("dimensionForSchwarzNeighbors", &dimensionForSchwarzNeighbors, "dimension to use for Schwarz neighbor relationship (only supported for geometric Schwarz; default of -1 means to use spaceDim-1 (face neighbors).");
  cmdp.setOption("graphNormBeta", &graphNormBeta);

  cmdp.setOption("useSchwarzPreconditioner", "useGMGPreconditioner", &schwarzOnly);
  cmdp.setOption("smoother", &smootherChoiceStr);

  cmdp.setOption("hOnly", "notHOnly", &hOnly);
  cmdp.setOption("useHierarchicalNeighborsForSchwarz", "useNaturalNeighborsForSchwarz", &useHierarchicalNeighborsForSchwarz);

  cmdp.setOption("enhanceFieldsForH1TracesWhenConforming", "equalOrderFieldsForH1TracesWhenConforming", &enhanceFieldsForH1TracesWhenConforming);
  
  cmdp.setOption("schwarzFactorization", &schwarzFactorizationTypeString, "Schwarz block factorization strategy: Direct, IC, ILU");
  cmdp.setOption("schwarzFillRatio", &fillRatio, "Schwarz block factorization: fill ratio for IC");
  cmdp.setOption("schwarzLevelOfFill", &levelOfFill, "Schwarz block factorization: level of fill for ILU");
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("maxDofsForKLU",&maxDofsForKLU, "for multi-level solves, maximum number of dofs to use with KLU solve at coarsest level");
  cmdp.setOption("narrateSolution", "dontNarrateSolution", &narrateSolution);
  cmdp.setOption("narrateCoarseSolution", "dontNarrateCoarseSolution", &narrateCoarseSolution);
  cmdp.setOption("precondition", "dontPrecondition", &precondition);

  cmdp.setOption("overlap", &schwarzOverlap, "Schwarz overlap level");

  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (1, 2, or 3)");
  
  cmdp.setOption("smootherWeight", &smootherWeight, "smoother weight for GMGOperator.");

  cmdp.setOption("azOutput", &AztecOutputLevel, "Aztec output level");
  cmdp.setOption("numCells", &numCells, "number of cells in the mesh");
  cmdp.setOption("numCellsRootMesh", &numCellsRootMesh, "number of cells in the root mesh");

  cmdp.setOption("maxIterations", &cgMaxIterations, "maximum number of CG iterations");
  cmdp.setOption("cgTol", &cgTol, "CG convergence tolerance");

  cmdp.setOption("coarseCGTol", &coarseCGTol, "coarse solve CG tolerance");
  cmdp.setOption("coarseMaxIterations", &coarseMaxIterations, "coarse solve max iterations");

  cmdp.setOption("reportTimings", "dontReportTimings", &reportTimings, "Report timings in Solution");

  cmdp.setOption("runMany", "runOne", &runAutomatic, "Run in automatic mode (ignores several input parameters)");
  cmdp.setOption("runManySubset", &runManySubsetString, "DontPrecondition, AllGMG, AllSchwarz, or All");
  cmdp.setOption("runManyMinCells", &runManyMinCells, "Minimum number of cells to use for mesh width");
  cmdp.setOption("runManyMaxCells", &maxCells, "Maximum number of cells to use for mesh width");

  cmdp.setOption("writeAndExit", "runNormally", &writeAndExit, "Write A, A_coarse, P, and S to disk, and exit without computing anything.");

  cmdp.setOption("useWeightedSchwarz","useUnweightedSchwarz",&useWeightMatrixForSchwarz, "Use weight matrix ('W' in Fischer and Lottes) to scale Schwarz smoother according to multiplicities.  Only applies to GMG geometric Schwarz.");

  cmdp.setOption("useZeroMeanConstraint", "usePointConstraint", &useZeroMeanConstraints, "Use a zero-mean constraint for the pressure (otherwise, use a vertex constraint at the origin)");

  cmdp.setOption("gmgOperatorDebug", "gmgOperatorNormal", &runGMGOperatorInDebugMode, "Run GMGOperator in a debug mode");
  
  cmdp.setOption("saveVisualizationOutput", "dontSaveVisualizationOutput", &saveVisualizationOutput);

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  ProblemChoice problemChoice;

  if (problemChoiceString == "Poisson")
  {
    problemChoice = Poisson;
  }
  else if (problemChoiceString == "ConvectionDiffusion")
  {
    problemChoice = ConvectionDiffusion;
  }
  else if (problemChoiceString == "ConvectionDiffusionExperimental")
  {
    problemChoice = ConvectionDiffusionExperimental;
  }
  else if (problemChoiceString == "LinearElasticity")
  {
    problemChoice = LinearElasticity;
  }
  else if (problemChoiceString == "Stokes")
  {
    problemChoice = Stokes;
  }
  else if (problemChoiceString == "Navier-Stokes")
  {
    problemChoice = NavierStokes;
  }
  else
  {
    if (rank==0) cout << "Problem choice not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  GMGOperator::SmootherChoice smootherChoice;

  if (smootherChoiceStr == "None")
  {
    smootherChoice = GMGOperator::NONE;
  }
  else if (smootherChoiceStr == "IfPack-Schwarz")
  {
    smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
  }
  else if (smootherChoiceStr == "Camellia-Schwarz")
  {
    smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;
  }
  else if (smootherChoiceStr == "Point-Jacobi")
  {
    smootherChoice = GMGOperator::POINT_JACOBI;
  }
  else if (smootherChoiceStr == "Block-Jacobi")
  {
    smootherChoice = GMGOperator::BLOCK_JACOBI;
  }
  else if (smootherChoiceStr == "Point-Gauss-Seidel")
  {
    smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
  }
  else if (smootherChoiceStr == "Block-Gauss-Seidel")
  {
    smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL;
  }
  else
  {
    if (rank==0) cout << "Smoother choice string not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  GMGOperator::SmootherApplicationType comboType = additiveComboType ? GMGOperator::ADDITIVE : GMGOperator::MULTIPLICATIVE;
//  if (comboType == GMGOperator::MULTIPLICATIVE) // default to a weight of 0.5 for smoother
//  {
//    if (smootherWeight == -1.0) smootherWeight = 0.5;
//  }
//  else if (comboType == GMGOperator::ADDITIVE)
//  {
//    if (smootherWeight == -1.0) smootherWeight = 1.0;
//  }

  Solver::SolverChoice coarseSolverChoice = Solver::solverChoiceFromString(coarseSolverChoiceString);

  RunManyPreconditionerChoices runManySubsetChoice;

  if (runManySubsetString == "All")
  {
    runManySubsetChoice = All;
  }
  else if (runManySubsetString == "DontPrecondition")
  {
    runManySubsetChoice = DontPrecondition;
  }
  else if (runManySubsetString == "AllGMG")
  {
    runManySubsetChoice = AllGMG;
  }
  else if (runManySubsetString == "AllSchwarz")
  {
    runManySubsetChoice = AllSchwarz;
  }
  else if (runManySubsetString == "AlgebraicSchwarz")
  {
    runManySubsetChoice = AlgebraicSchwarz;
  }
  else if (runManySubsetString == "GeometricSchwarz")
  {
    runManySubsetChoice = GeometricSchwarz;
  }
  else if (runManySubsetString == "GMGAlgebraicSchwarz")
  {
    runManySubsetChoice = GMGAlgebraicSchwarz;
  }
  else if (runManySubsetString == "GMGGeometricSchwarz")
  {
    runManySubsetChoice = GMGGeometricSchwarz;
  }
  else if (runManySubsetString == "GMGPointGaussSeidel")
  {
    runManySubsetChoice = GMGPointSymmetricGaussSeidel;
  }
  else if (runManySubsetString == "GMGBlockGaussSeidel")
  {
    runManySubsetChoice = GMGBlockSymmetricGaussSeidel;
  }
  else if (runManySubsetString == "GMGPointJacobi")
  {
    runManySubsetChoice = GMGPointJacobi;
  }
  else if (runManySubsetString == "GMGBlockJacobi")
  {
    runManySubsetChoice = GMGBlockJacobi;
  }
  else
  {
    if (rank==0) cout << "Run many subset string not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  GMGOperator::FactorType schwarzFactorType = getFactorizationType(schwarzFactorizationTypeString);

  if (delta_k==-1) delta_k = spaceDim;

  if (! runAutomatic)
  {
    int iterationCount;
    bool reportEnergyError = true;

    if (schwarzOverlap == -1) schwarzOverlap = 0;
    if (k == -1) k = 2;

    double solveTime, error;

    run(problemChoice, iterationCount, spaceDim, numCells, k, delta_k, k_coarse, conformingTraces,
        useCondensedSolve, precondition, schwarzOnly, smootherChoice, schwarzOverlap, dimensionForSchwarzNeighbors,
        schwarzFactorType, levelOfFill, fillRatio, coarseSolverChoice,
        cgTol, cgMaxIterations, AztecOutputLevel, reportTimings, solveTime, error,
        reportEnergyError, numCellsRootMesh, hOnly, useHierarchicalNeighborsForSchwarz,
        useZeroMeanConstraints, writeAndExit, comboType, smootherWeight,
        useWeightMatrixForSchwarz, enhanceFieldsForH1TracesWhenConforming, graphNormBeta);

    if (rank==0) cout << "Iteration count: " << iterationCount << "; solve time " << solveTime << " seconds." << endl;
  }
  else
  {
    if (maxCells == -1)
    {
      // by default, ensure max of 1 cell per MPI node
      int nProc = Teuchos::GlobalMPISession::getNProc();
      maxCells = 1;
      while (maxCells*2 <= nProc)
      {
        maxCells *= 2;
      }
    }
    if (problemChoice == Stokes)
    {
      maxCells = max(maxCells,2);
      runManyMinCells = max(runManyMinCells,2);
    }
    
    if (rank==0)
    {
      cout << "Running " << problemChoiceString << " solver in automatic mode (subset: ";
      cout << runManySubsetString << "), with spaceDim " << spaceDim;
      cout << ", delta_k = " << delta_k << ", ";
      if (conformingTraces)
        cout << "conforming traces, ";
      else
        cout << "non-conforming traces, ";
      cout << "CG tolerance = " << cgTol << ", max iterations = " << cgMaxIterations << endl;
    }

    runMany(problemChoice, spaceDim, delta_k, runManyMinCells,
            conformingTraces, useCondensedSolve,
            schwarzFactorType, levelOfFill, fillRatio,
            coarseSolverChoice,
            cgTol, cgMaxIterations, AztecOutputLevel,
            runManySubsetChoice, k, k_coarse, schwarzOverlap, dimensionForSchwarzNeighbors, numCellsRootMesh, reportTimings,
            hOnly, useHierarchicalNeighborsForSchwarz, maxCells,
            useZeroMeanConstraints, comboType, smootherWeight, useWeightMatrixForSchwarz, enhanceFieldsForH1TracesWhenConforming,
            graphNormBeta);
  }
  return 0;
}
