#include "ConvectionDiffusionReactionFormulation.h"
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
  IPPtr ip;
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
  }
  
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
  
  SolutionPtr solution = Solution::solution(bf, mesh, bc, rhs, ip);
  
  bool useCondensedSolve = false;
  solution->setUseCondensedSolve(useCondensedSolve);
  solution->solve();
  
  int refinementNumber = 0;
  FunctionPtr meshIndicator = Function::meshSkeletonCharacteristic();
  HDF5Exporter functionExporter(mesh, "ConvectionDiffusionExampleFunctions");
  functionExporter.exportFunction({u_exact, f}, {"u_exact", "f"}, refinementNumber);
  functionExporter.exportFunction({meshIndicator}, "mesh", refinementNumber);
  
  HDF5Exporter exporter(mesh, "ConvectionDiffusionExample", "/tmp");
  exporter.exportSolution(solution, refinementNumber);
  
  if (formulation == ConvectionDiffusionReactionFormulation::SUPG)
  {
    if (rank==0) cout << "Formulation = SUPG; exiting after initial solve (no refinement strategy defined for this case)\n";
    exit(0);
  }
  
  RefinementStrategy refStrategy(solution, energyThreshold);
 
  while (refinementNumber < numRefinements)
  {
    refStrategy.refine();
    functionExporter.exportFunction({u_exact, f}, {"u_exact", "f"}, refinementNumber);
    functionExporter.exportFunction({meshIndicator}, "mesh", refinementNumber);
   
    vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, 1, delta_k);
    
    int maxIters = 500;
    double cgTol = 1e-6;
    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(solution, meshesCoarseToFine, maxIters, cgTol,
                                                                   GMGOperator::MultigridStrategy::V_CYCLE,
                                                                   Solver::getDirectSolver(),
                                                                   useCondensedSolve));
    gmgSolver->setAztecOutput(25);
    
    double energyError = refStrategy.getEnergyError(refinementNumber);
    GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
    GlobalIndexType numActiveElements = mesh->numActiveElements();
    if (rank == 0)
    {
      cout << "Refinement " << refinementNumber << " has energy error " << energyError;
      cout << " (" << numGlobalDofs << " dofs on " << numActiveElements << " elements)\n";
    }
    
    solution->solve(gmgSolver);
    refinementNumber++;
    
    exporter.exportSolution(solution, refinementNumber);
  }
  
  double energyError = solution->energyErrorTotal();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  GlobalIndexType numActiveElements = mesh->numActiveElements();
  if (rank == 0)
  {
    cout << "Refinement " << refinementNumber << " has energy error " << energyError;
    cout << " (" << numGlobalDofs << " dofs on " << numActiveElements << " elements)\n";
  }

  return 0;
}