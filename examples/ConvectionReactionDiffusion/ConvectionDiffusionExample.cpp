#include "ConvectionDiffusionReactionFormulation.h"
#include "EnergyErrorFunction.h"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"

#include "EpetraExt_RowMatrixOut.h"

using namespace Camellia;
using namespace std;

/*
 Example 4.2 from Broersen and Stevenson:
 
 Manufactured solution:
 
 u(x,y) = g(b_1,x) * g(b_2,y)
 
 where
 
 g(a,z) = [z + (e^(a * z / epsilon) - 1) / (1 - e^(a / epsilon)]
 
 and convective direction beta = (b_1,b_2).
 
 We solve this on domain [0,1]^2.
 
 */

const static int INFLOW_TAG_ID  = 0; // arbitrary, just an identifier
const static int OUTFLOW_TAG_ID = 1;

class ExpTerm : public SimpleFunction<double>
{
  double _weight; // beta_1 / epsilon or beta_2 / epsilon
  bool _xTerm; // alternative is y-term; determines which of x or y enters the exponent
  double _denominator;
public:
  ExpTerm(double weight, bool xTerm)
  {
    _weight = weight;
    _xTerm = xTerm;
    _denominator = exp(-weight) - 1.0;
  }
  
  double value(double x, double y)
  {
    double arg = _xTerm ? x : y;
    return (exp(_weight * (arg - 1)) - exp(-_weight)) / _denominator;
  }
  
  FunctionPtr dx()
  {
    if (_xTerm)
    {
      // derivative will be this function times _weight without the constant
      //  - exp(-_weight)) / _denominator
      FunctionPtr thisFxn = Teuchos::rcp( new ExpTerm(_weight,_xTerm));
      double constPart = - exp(-_weight) / _denominator;
      return _weight * (thisFxn - constPart);
    }
    else
    {
      return Function::zero();
    }
  }
  
  FunctionPtr dy()
  {
    if (!_xTerm)
    {
      // derivative will be this function times _weight without the constant
      //  - exp(-_weight)) / _denominator
      FunctionPtr thisFxn = Teuchos::rcp( new ExpTerm(_weight,_xTerm));
      double constPart = - exp(-_weight) / _denominator;
      return _weight * (thisFxn - constPart);
    }
    else
    {
      return Function::zero();
    }
  }
};

FunctionPtr exactSolution(double epsilon, double beta_1, double beta_2)
{
  // example 4.2
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  
  double x_beta_epsilon = beta_1 / epsilon;
  double y_beta_epsilon = beta_2 / epsilon;

  FunctionPtr xTerm = x + (FunctionPtr) Teuchos::rcp( new ExpTerm(x_beta_epsilon, true));
  FunctionPtr yTerm = y + (FunctionPtr) Teuchos::rcp( new ExpTerm(y_beta_epsilon, false));
  
  return xTerm * yTerm;
}

class CellIndicatorFunction : public TFunction<double>
{
  std::set<GlobalIndexType> _cellIDs;
public:
  CellIndicatorFunction(GlobalIndexType cellID) : TFunction<double>(0)
  {
    _cellIDs.insert(cellID);
  }
  CellIndicatorFunction(std::vector<GlobalIndexType> &cellIDs) : TFunction<double>(0)
  {
    _cellIDs.insert(cellIDs.begin(),cellIDs.end());
  }
  
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    IndexType cellIndex = 0;
    int numPoints = values.dimension(1);
    for (GlobalIndexType cellID : cellIDs)
    {
      double value;
      if (_cellIDs.find(cellID) == _cellIDs.end())
        value = 0;
      else
        value = 1;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        values(cellIndex, ptIndex) = value;
      }
      cellIndex++;
    }
  }
};

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  MPIWrapper::CommWorld()->Barrier(); // can set a breakpoint here for debugger attachment
  
  int rank = mpiSession.getRank();
  
  // set defaults to match Broersen and Stevenson's experiment
  int meshWidth = 16;
  int polyOrder = 1, delta_k = 1;
  int spaceDim = 2;
  double epsilon = 1e-3;
  double beta_1 = 2.0, beta_2 = 1.0;
  bool useTriangles = true; // otherwise, quads
  int numRefinements = 0;
  double energyThreshold = 0.2; // for refinements
  string formulationChoice = "Ultraweak";
  bool conditionNumberEstimate = false;
  bool exportVisualization = false;
  bool reportRefinedCells = false;
  bool reportSolutionTimings = true;
  int quadratureEnrichment = 0;
  int quadratureEnrichmentL2 = 10;
  bool useDirectSolver = false;
  bool useCondensedSolve = false;
  bool exportMatrix = false;
  bool refineUniformly = false;
  int maxIterations = 2000;
  int iterativeOutputLevel = 100;
  double cgTol = 1e-6;
  double weightForL2TermsGraphNorm = -1;
  
  cmdp.setOption("meshWidth", &meshWidth );
  cmdp.setOption("polyOrder", &polyOrder );
  cmdp.setOption("delta_k", &delta_k );
  cmdp.setOption("spaceDim", &spaceDim);
  cmdp.setOption("epsilon", &epsilon);
  cmdp.setOption("exportMatrix", "dontExportMatrix", &exportMatrix);
  cmdp.setOption("beta_1", &beta_1);
  cmdp.setOption("beta_2", &beta_2);
  cmdp.setOption("useTriangles", "useQuads", &useTriangles);
  cmdp.setOption("useDirectSolver", "useIterativeSolver", &useDirectSolver);
  cmdp.setOption("numRefinements", &numRefinements);
  cmdp.setOption("quadratureEnrichment", &quadratureEnrichment, "quadrature enrichment for Solution");
  cmdp.setOption("quadratureEnrichmentL2", &quadratureEnrichmentL2);
  cmdp.setOption("energyThreshold", &energyThreshold);
  cmdp.setOption("formulationChoice", &formulationChoice);
  cmdp.setOption("refineUniformly", "refineUsingErrorIndicator", &refineUniformly);
  cmdp.setOption("reportRefinedCells", "dontReportRefinedCells", &reportRefinedCells);
  cmdp.setOption("reportConditionNumber", "dontReportConditionNumber", &conditionNumberEstimate);
  cmdp.setOption("reportSolutionTimings", "dontReportSolutionTimings", &reportSolutionTimings);
  cmdp.setOption("useDirect", "useIterative", &useDirectSolver);
  cmdp.setOption("useVisualization","noVisualization",&exportVisualization);
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("iterativeOutputLevel", &iterativeOutputLevel, "How many iterations to take before reporting iterative solver progress (0 to suppress output)");
  cmdp.setOption("iterativeTol", &cgTol);
  cmdp.setOption("maxIterations", &maxIterations);
  cmdp.setOption("weightForL2TermsGraphNorm", &weightForL2TermsGraphNorm, "weight to use for the L^2 terms in graph norm (ultraweak only); default value of -1 means use sqrt(epsilon)");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  if (weightForL2TermsGraphNorm == -1)
  {
    weightForL2TermsGraphNorm = sqrt(sqrt(epsilon));
  }
  
  ConvectionDiffusionReactionFormulation::FormulationChoice formulation;
  if (formulationChoice == "Ultraweak")
  {
    formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
  }
  else if (formulationChoice == "SUPG")
  {
    formulation = ConvectionDiffusionReactionFormulation::SUPG;
  }
  else if (formulationChoice == "Primal")
  {
    formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported formulation choice!");
  }
  
  bool formulationIsDPG = (formulation == ConvectionDiffusionReactionFormulation::ULTRAWEAK) || (formulation == ConvectionDiffusionReactionFormulation::PRIMAL);
  
  double alpha = 0; // no reaction term
  FunctionPtr u_exact = exactSolution(epsilon, beta_1, beta_2);

  ostringstream thisRunPrefix;
  thisRunPrefix << formulationChoice << "_eps" << epsilon << "_k" << polyOrder << "_";
  
  FunctionPtr beta = Function::constant({beta_1,beta_2});
  ConvectionDiffusionReactionFormulation form(formulation, spaceDim, beta, epsilon, alpha);
  
  // determine forcing function and RHS
  FunctionPtr f = form.forcingFunction(u_exact);
  RHSPtr rhs = form.rhs(f);
  
  // bilinear form
  BFPtr bf = form.bf();
  
  // inner product
  IPPtr ip, ipForResidual;
  int H1Order = -1;
  if (formulation == ConvectionDiffusionReactionFormulation::ULTRAWEAK)
  {
    ip = bf->graphNorm(weightForL2TermsGraphNorm);
    H1Order = polyOrder + 1;
  }
  if (formulation == ConvectionDiffusionReactionFormulation::PRIMAL)
  {
    H1Order = polyOrder;
    ip = bf->naiveNorm(spaceDim);
  }
  if (formulation == ConvectionDiffusionReactionFormulation::SUPG)
  {
    delta_k = 0;
    H1Order = polyOrder;
    ConvectionDiffusionReactionFormulation primalForm(ConvectionDiffusionReactionFormulation::PRIMAL, spaceDim, beta, epsilon, alpha);
    ipForResidual = primalForm.bf()->naiveNorm(spaceDim);
  }
  
  // set up timings printer
  double totalTimeG = 0, totalTimeB = 0, totalTimeT = 0, totalTimeK = 0;
  int totalElementCount = 0;
  std::function<void(int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType)> optimalTestTimingCallback;
  optimalTestTimingCallback = [&totalTimeG, &totalTimeB, &totalTimeT, &totalTimeK, &totalElementCount]
    (int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType) {
//      cout << "BF timings: on " << numElements << " elements, computed G in " << timeG;
//      cout << " seconds, B in " << timeB << " seconds; solve for T in " << timeT;
//      cout << " seconds; compute K=B^T T in " << timeK << " seconds." << endl;
      totalElementCount += numElements;
      totalTimeG += timeG;
      totalTimeB += timeB;
      totalTimeT += timeT;
      totalTimeK += timeK;
  };
  // same thing, now for RHS computation:
  std::function<void(int numElements, double timeRHS, ElementTypePtr elemType)> rhsTimingCallback;
  double totalTimeRHS = 0;
  int totalRHSElements = 0;
  rhsTimingCallback = [&totalTimeRHS, &totalRHSElements]
  (int numElements, double timeRHS, ElementTypePtr elemType) {
    totalTimeRHS += timeRHS;
    totalRHSElements += numElements;
  };
  // register the timing callbacks
  bf->setOptimalTestTimingCallback(optimalTestTimingCallback);
  bf->setRHSTimingCallback(rhsTimingCallback);
  
  // set up mesh
  vector<double> dimensions = {1.0,1.0};
  vector<int> elementCounts = {meshWidth,meshWidth};
  
  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                              elementCounts[0], elementCounts[1], useTriangles);

  // define inflow boundary
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
  SpatialFilterPtr inflowFilter = SpatialFilter::matchingX(0.0) | SpatialFilter::matchingY(0.0);
  vector<IndexType> inflowSides = meshTopo->getBoundarySidesThatMatch(inflowFilter);
  EntitySetPtr inflowEntitySet = meshTopo->createEntitySet();
  int sideDim = spaceDim - 1;
  for (IndexType sideEntityIndex : inflowSides)
  {
    inflowEntitySet->addEntity(sideDim, sideEntityIndex);
  }
  meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, INFLOW_TAG_ID, inflowEntitySet);
  
  SpatialFilterPtr outflowFilter = SpatialFilter::matchingX(1.0) | SpatialFilter::matchingY(1.0);
  vector<IndexType> outflowSides = meshTopo->getBoundarySidesThatMatch(outflowFilter);
  EntitySetPtr outflowEntitySet = meshTopo->createEntitySet();
  for (IndexType sideEntityIndex : outflowSides)
  {
    outflowEntitySet->addEntity(sideDim, sideEntityIndex);
  }
  meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, OUTFLOW_TAG_ID, outflowEntitySet);
  
  BCPtr bc = BC::bc();
  bc->addDirichlet(form.u_dirichlet(), INFLOW_TAG_ID, u_exact);
  bc->addDirichlet(form.u_dirichlet(), OUTFLOW_TAG_ID, u_exact); // u_exact should actually be zero on the whole boundary
//  bc->addDirichlet(form.u_dirichlet(), INFLOW_TAG_ID, Function::zero());
//  bc->addDirichlet(form.u_dirichlet(), OUTFLOW_TAG_ID, Function::zero()); // u_exact should actually be zero on the whole boundary
  
  SolutionPtr solution = Solution::solution(bf, mesh, bc, rhs, ip);
  solution->setUseCondensedSolve(useCondensedSolve);
  solution->setCubatureEnrichmentDegree(quadratureEnrichment);
  
  // Solution for L^2 projections:
  SolutionPtr bestSolution = Solution::solution(bf, mesh, bc, rhs, ip);
  bestSolution->setCubatureEnrichmentDegree(quadratureEnrichment);
  map<int,FunctionPtr> exactSolutionMap = {{form.u()->ID(), u_exact}};
  bestSolution->projectOntoMesh(exactSolutionMap);
  FunctionPtr u_best = Function::solution(form.u(), bestSolution);
  
  int refinementNumber = 0;
  ostringstream matrixFileName;
  matrixFileName << thisRunPrefix.str() << "ref" << refinementNumber << ".dat";
  
  if (exportMatrix)
  {
    solution->setWriteMatrixToFile(true, matrixFileName.str());
  }
  
  auto getGMGSolver = [&solution, &formulationIsDPG, &delta_k, &useCondensedSolve, &maxIterations, &cgTol, &iterativeOutputLevel] () -> Teuchos::RCP<GMGSolver>
  {
    vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(solution->mesh(), 1, delta_k);

    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(solution, meshesCoarseToFine, maxIterations, cgTol,
                                                                   GMGOperator::MultigridStrategy::V_CYCLE,
                                                                   Solver::getDirectSolver(),
                                                                   useCondensedSolve));

    if (!formulationIsDPG)
    {
      gmgSolver->setSmootherType(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
      gmgSolver->setUseConjugateGradient(false); // won't be SPD
    }
    else
    {
      gmgSolver->setSmootherType(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
    }
    
    gmgSolver->setAztecOutput(iterativeOutputLevel);

    return gmgSolver;
  };
  
  SolverPtr solver;
  if (useDirectSolver)
  {
    solver = Solver::getDirectSolver();
  }
  else
  {
    solver = getGMGSolver();
  }
  
  solution->solve(solver);
  if (conditionNumberEstimate)
  {
    int errCode;
    double conditionNumber = solution->conditionNumberEstimate(errCode);
    if (rank==0)
    {
      if (errCode != 0)
      {
        cout << "Condition Number estimation failed with error code " << errCode;
        cout << " (estimate, for what it's worth--very little, probably--was " << conditionNumber << ")\n";
      }
      else
      {
        cout << "Condition Number estimate initial mesh: " << conditionNumber << endl;
      }
    }
  }
  if (reportSolutionTimings)
  {
    solution->reportTimings();
  }
  LinearTermPtr residual = form.residual(solution);
  if (ipForResidual == Teuchos::null) ipForResidual = ip;
  RefinementStrategyPtr refStrategy;
  if (formulationIsDPG)
  {
    refStrategy = RefinementStrategy::energyErrorRefinementStrategy(mesh, residual, ipForResidual, energyThreshold);
  }
  else
  {
    VarPtr varForHessian = form.u();
    refStrategy = RefinementStrategy::hessianRefinementStrategy(solution, varForHessian, energyThreshold);
  }
  
  RieszRepPtr rieszRep = RieszRep::rieszRep(mesh, ipForResidual, residual);
  FunctionPtr energyErrorFunction = Teuchos::rcp( new EnergyErrorFunction(rieszRep) );
  rieszRep->computeRieszRep();
  
  vector<FunctionPtr> functionsToExport;
  vector<string> functionsToExportNames;
  
  if (formulation != ConvectionDiffusionReactionFormulation::SUPG)
  {
    functionsToExport = {u_exact, u_best, f, energyErrorFunction};
    functionsToExportNames = {"u_exact", "u_best", "f", "energy_error"};
  }
  else
  {
    // omit energy error for SUPG
    functionsToExport = {u_exact, u_best, f};
    functionsToExportNames = {"u_exact", "u_best", "f"};
  }

  if (formulation == ConvectionDiffusionReactionFormulation::SUPG)
  {
    functionsToExport.push_back(form.SUPGStabilizationWeight());
    functionsToExportNames.push_back("tau");
  }
  FunctionPtr meshIndicator = Function::meshSkeletonCharacteristic();
  vector<FunctionPtr> traceFunctionsToExport = {meshIndicator};
  vector<string> traceFunctionsToExportNames = {"mesh"};
  
  FunctionPtr u_soln = Function::solution(form.u(), solution);
  double errL2, bestErrL2;
  
  // for some reason, ParaView introduces visual artifacts when doing plot over line on traces when
  // the outputted resolution (HDF5Exporter's "num1DPoints" argument) is too low.  (10 is too low.)
  int numLinearPointsPlotting = max(polyOrder,15);

  Teuchos::RCP<HDF5Exporter> functionExporter;
  Teuchos::RCP<HDF5Exporter> solnExporter;
  if (exportVisualization)
  {
    string exporterName = "ConfusionFunctions";
    exporterName = thisRunPrefix.str() + exporterName;
    functionExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));
    functionExporter->exportFunction(functionsToExport, functionsToExportNames, refinementNumber, numLinearPointsPlotting);
    functionExporter->exportFunction(traceFunctionsToExport, traceFunctionsToExportNames, refinementNumber, numLinearPointsPlotting);
    
    exporterName = "ConfusionSolution";
    exporterName = thisRunPrefix.str() + exporterName;
    solnExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));
    solnExporter->exportSolution(solution, refinementNumber, numLinearPointsPlotting);
  }
  
  while (refinementNumber < numRefinements)
  {
    GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
    GlobalIndexType numActiveElements = mesh->numActiveElements();
    
    errL2 = (u_soln - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);
    
    bestSolution->projectOntoMesh(exactSolutionMap);
    bestErrL2 = (u_best - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);
    
    if (!refineUniformly)
    {
      refStrategy->refine(reportRefinedCells && (rank==0));
    }
    else
    {
      refStrategy->hRefineUniformly();
    }
   
    // recreate solver if neccessary, after refinement:
    SolverPtr solver;
    if (useDirectSolver)
    {
      solver = Solver::getDirectSolver();
    }
    else
    {
      solver = getGMGSolver();
    }

    if (formulationIsDPG)
    {
      double energyError = refStrategy->getEnergyError(refinementNumber);
      if (rank == 0)
      {
        cout << "Refinement " << refinementNumber << " has energy error " << energyError;
        cout << ", L^2 error " << errL2 << " (vs best L^2 error of " << bestErrL2 << ")";
        cout << " (" << numGlobalDofs << " dofs on " << numActiveElements << " elements)\n";
      }
    }
    else
    {
      double error = refStrategy->getEnergyError(refinementNumber);
      if (rank == 0)
      {
        cout << "Refinement " << refinementNumber << " has hessian error " << error;
        cout << ", L^2 error " << errL2 << " (vs best L^2 error of " << bestErrL2 << ")";
        cout << " (" << numGlobalDofs << " dofs on " << numActiveElements << " elements)\n";
      }
    }
    
    ostringstream matrixFileName;
    matrixFileName << thisRunPrefix.str() << "ref" << refinementNumber << ".dat";
    
    if (exportMatrix)
    {
      solution->setWriteMatrixToFile(true, matrixFileName.str());
    }
    
    solution->solve(solver);
    refinementNumber++;
    
    if (reportSolutionTimings)
    {
      solution->reportTimings();
    }
    
    if (exportVisualization)
    {
      solnExporter->exportSolution(solution, refinementNumber, numLinearPointsPlotting);
    }
    
    if (conditionNumberEstimate)
    {
      int errCode;
      double conditionNumber = solution->conditionNumberEstimate(errCode);
      if (rank==0)
      {
        if (errCode != 0)
        {
          cout << "Condition Number estimation failed with error code " << errCode;
          cout << " (estimate, for what it's worth--very little, probably--was " << conditionNumber << ")\n";
        }
        else
        {
          cout << "Condition Number estimate refinement " << refinementNumber << ": " << conditionNumber << endl;
        }
      }
    }
    
    if (formulationIsDPG)
    {
      Epetra_Time timer(*MPIWrapper::CommSerial());
      rieszRep->computeRieszRep();
      double rieszRepTime = timer.ElapsedTime();
      if (rank==0)
      {
        cout << "Computed Riesz rep in " << rieszRepTime << " seconds (timed on rank 0).\n";
      }
    }
    if (exportVisualization)
    {
      functionExporter->exportFunction(functionsToExport, functionsToExportNames, refinementNumber, numLinearPointsPlotting);
      functionExporter->exportFunction(traceFunctionsToExport, traceFunctionsToExportNames, refinementNumber, numLinearPointsPlotting);
    }
  }
  
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  GlobalIndexType numActiveElements = mesh->numActiveElements();

  bestSolution->projectOntoMesh(exactSolutionMap);
  bestErrL2 = (u_best - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);
  errL2 = (u_soln - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);
  
  double totalError = refStrategy->computeTotalEnergyError();
  if (rank == 0)
  {
    cout << "Refinement " << refinementNumber << " has total indicator error " << totalError;
    cout << ", L^2 error " << errL2 << " (vs best L^2 error of " << bestErrL2 << ")";
    cout << " (" << numGlobalDofs << " dofs on " << numActiveElements << " elements)\n";
  }

  // compute average timings across all MPI ranks.
  Epetra_CommPtr Comm = MPIWrapper::CommWorld();
  totalTimeG = MPIWrapper::sum(*Comm, totalTimeG);
  totalTimeB = MPIWrapper::sum(*Comm, totalTimeB);
  totalTimeT = MPIWrapper::sum(*Comm, totalTimeT);
  totalTimeK = MPIWrapper::sum(*Comm, totalTimeK);
  totalElementCount = MPIWrapper::sum(*Comm, totalElementCount);

  totalTimeRHS = MPIWrapper::sum(*Comm, totalTimeRHS);
  totalRHSElements = MPIWrapper::sum(*Comm, totalRHSElements);
  
  if ((rank == 0) && (formulation != ConvectionDiffusionReactionFormulation::SUPG))
  {
    cout << "Average time/element for optimal test computations:\n";
    cout << "G:   " <<  totalTimeG / totalElementCount << " sec.\n";
    cout << "B:   " <<  totalTimeB / totalElementCount << " sec.\n";
    cout << "T:   " <<  totalTimeT / totalElementCount << " sec.\n";
    cout << "K:   " <<  totalTimeK / totalElementCount << " sec.\n";
    cout << "RHS: " << totalTimeRHS / totalRHSElements << " sec.\n";
  }
  
  return 0;
}