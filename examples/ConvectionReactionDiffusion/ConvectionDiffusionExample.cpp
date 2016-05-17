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
  double epsilon = 1e-6;
  double beta_1 = 2.0, beta_2 = 1.0;
  bool useTriangles = true; // otherwise, quads
  int numRefinements = 0;
  double energyThreshold = 0.2; // for refinements
  string formulationChoice = "Ultraweak";
  bool conditionNumberEstimate = false;
  
  cmdp.setOption("meshWidth", &meshWidth );
  cmdp.setOption("polyOrder", &polyOrder );
  cmdp.setOption("delta_k", &delta_k );
  cmdp.setOption("spaceDim", &spaceDim);
  cmdp.setOption("epsilon", &epsilon);
  cmdp.setOption("beta_1", &beta_1);
  cmdp.setOption("beta_2", &beta_2);
  cmdp.setOption("useTriangles", "useQuads", &useTriangles);
  cmdp.setOption("numRefinements", &numRefinements);
  cmdp.setOption("energyThreshold", &energyThreshold);
  cmdp.setOption("formulationChoice", &formulationChoice);
  cmdp.setOption("reportConditionNumber", "dontReportConditionNumber", &conditionNumberEstimate);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  double alpha = 0; // no reaction term
  FunctionPtr u_exact = exactSolution(epsilon, beta_1, beta_2);
  
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
    ip = bf->graphNorm();
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
      cout << "BF timings: on " << numElements << " elements, computed G in " << timeG;
      cout << " seconds, B in " << timeB << " seconds; solve for T in " << timeT;
      cout << " seconds; compute K=B^T T in " << timeK << " seconds." << endl;
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
    cout << "RHS timings: on " << numElements << " elements, computed RHS in " << timeRHS << " seconds.\n";
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
  
  bool useCondensedSolve = false;
  solution->setUseCondensedSolve(useCondensedSolve);
  solution->solve();
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
  LinearTermPtr residual = form.residual(solution);
  if (ipForResidual == Teuchos::null) ipForResidual = ip;
  RefinementStrategy refStrategy(mesh, residual, ipForResidual, energyThreshold);
  FunctionPtr energyErrorFunction = Teuchos::rcp( new EnergyErrorFunction(refStrategy.getRieszRep()) );
  refStrategy.getRieszRep()->computeRieszRep();
  
  vector<FunctionPtr> functionsToExport = {u_exact, f, energyErrorFunction};
  vector<string> functionsToExportNames = {"u_exact", "f", "energy_error"};
  if (formulation == ConvectionDiffusionReactionFormulation::SUPG)
  {
    functionsToExport.push_back(form.SUPGStabilizationWeight());
    functionsToExportNames.push_back("tau");
  }
  FunctionPtr meshIndicator = Function::meshSkeletonCharacteristic();
  vector<FunctionPtr> traceFunctionsToExport = {meshIndicator};
  vector<string> traceFunctionsToExportNames = {"mesh"};
  
  int refinementNumber = 0;
  
  // for some reason, ParaView introduces visual artifacts when doing plot over line on traces when
  // the outputted resolution (HDF5Exporter's "num1DPoints" argument) is too low.  (10 is too low.)
  int numLinearPointsPlotting = max(polyOrder,15);

  HDF5Exporter functionExporter(mesh, "ConvectionDiffusionExampleFunctions");
  functionExporter.exportFunction(functionsToExport, functionsToExportNames, refinementNumber, numLinearPointsPlotting);
  functionExporter.exportFunction(traceFunctionsToExport, traceFunctionsToExportNames, refinementNumber, numLinearPointsPlotting);
  
  HDF5Exporter exporter(mesh, "ConvectionDiffusionExampleSolution");
  exporter.exportSolution(solution, refinementNumber, numLinearPointsPlotting);
  
  while (refinementNumber < numRefinements)
  {
    GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
    GlobalIndexType numActiveElements = mesh->numActiveElements();
    
    refStrategy.refine();
   
    vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, 1, delta_k);
    
//    int maxIters = 500;
//    double cgTol = 1e-6;
//    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(solution, meshesCoarseToFine, maxIters, cgTol,
//                                                                   GMGOperator::MultigridStrategy::V_CYCLE,
//                                                                   Solver::getDirectSolver(),
//                                                                   useCondensedSolve));
//    
//    if (formulation == ConvectionDiffusionReactionFormulation::SUPG)
//    {
//      gmgSolver->setSmootherType(GMGOperator::POINT_JACOBI);
//      gmgSolver->setUseConjugateGradient(false); // won't be SPD
//    }
//    
//    gmgSolver->setAztecOutput(25);
    
    double energyError = refStrategy.getEnergyError(refinementNumber);

    if (rank == 0)
    {
      cout << "Refinement " << refinementNumber << " has energy error " << energyError;
      cout << " (" << numGlobalDofs << " dofs on " << numActiveElements << " elements)\n";
    }
    
//    solution->solve(gmgSolver);
    solution->solve();
    refinementNumber++;
    
    exporter.exportSolution(solution, refinementNumber, numLinearPointsPlotting);
    
    
    
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
    
    refStrategy.getRieszRep()->computeRieszRep();
    functionExporter.exportFunction(functionsToExport, functionsToExportNames, refinementNumber, numLinearPointsPlotting);
    functionExporter.exportFunction(traceFunctionsToExport, traceFunctionsToExportNames, refinementNumber, numLinearPointsPlotting);
  }
  
  double energyError = refStrategy.computeTotalEnergyError();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  GlobalIndexType numActiveElements = mesh->numActiveElements();
  if (rank == 0)
  {
    cout << "Refinement " << refinementNumber << " has energy error " << energyError;
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
  
  if (rank == 0)
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