#include "BC.h"
#include "BF.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Function.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "Solution.h"
#include "Solver.h"
#include "SpatialFilter.h"
#include "SpatiallyFilteredFunction.h"
#include "VarFactory.h"

#include "Epetra_FECrsMatrix.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"

#include "Intrepid_FunctionSpaceTools.hpp"

/*
 
 Applying Camellia with *DPG* to the same DG example problem in DGAdvectionExample.
 
 See
 
 https://dealii.org/developer/doxygen/deal.II/step_12.html
 
 for a deal.II tutorial discussion of this example in the DG context.
 
 */

using namespace Camellia;
using namespace std;
using namespace Intrepid;

// beta: 1/|x| (-x2, x1)

class BetaX : public SimpleFunction<double>
{
public:
  double value(double x, double y)
  {
    double mag = sqrt(x*x + y*y);
    if (mag > 0)
      return (1.0 / mag) * -y;
    else
      return 0;
  }
};

class BetaY : public SimpleFunction<double>
{
public:
  double value(double x, double y)
  {
    double mag = sqrt(x*x + y*y);
    if (mag > 0)
      return (1.0 / mag) * x;
    else
      return 0;
  }
};

class U_Exact_CCW : public SimpleFunction<double>
{
public:
  double value(double x, double y)
  {
    // 1 inside the circle of radius 0.5 centered at the origin
    const static double r = 0.5;
    if (x*x + y*y <= r*r)
      return 1.0;
    else
      return 0;
  }
};

class ElementWiseIntegralFunction : public TFunction<double>
{
  FunctionPtr _fxn;
public:
  ElementWiseIntegralFunction(FunctionPtr f)
  {
    _fxn = f;
  }
  
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      GlobalIndexType cellID = basisCache->cellIDs()[cellOrdinal];
      BasisCachePtr basisCacheForIntegral = BasisCache::basisCacheForCell(basisCache->mesh(), cellID);
    
      FieldContainer<double> cellIntegrals(1);
      _fxn->integrate(cellIntegrals, basisCacheForIntegral);
      
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = cellIntegrals(0);
      }
    }
  }
};

// DPG has a built-in error measurement (the energy norm), but we use the below for consistency
// with the DGAdvectionExample:
void computeApproximateGradients(SolutionPtr soln, VarPtr u, const vector<GlobalIndexType> &cells,
                                 vector<double> &gradient_l2_norm, double weightWithPowerOfH);

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int polyOrder = 1;
  int horizontalElements = 2, verticalElements = 2;
  int pToAddTest = 2; // using at least spaceDim seems to be important for pure convection
  int numRefinements = 6;
  int spaceDim = 2;
  bool useEnergyNormForRefinements = false;
  bool useCondensedSolve = false;
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("delta_k", &pToAddTest);
  cmdp.setOption("horizontalElements", &horizontalElements);
  cmdp.setOption("verticalElements", &verticalElements);
  cmdp.setOption("numRefinements", &numRefinements);
  cmdp.setOption("useEnergyError", "useGradientIndicator", &useEnergyNormForRefinements);
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  
  string convectiveDirectionChoice = "CCW"; // counter-clockwise, the default.  Other options: left, right, up, down
  cmdp.setOption("convectiveDirection", &convectiveDirectionChoice);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  FunctionPtr beta_x, beta_y;
  
  SpatialFilterPtr unitInflow, zeroInflow;
  
  FunctionPtr u_exact;
  
  if (convectiveDirectionChoice == "CCW")
  {
    beta_x = Teuchos::rcp( new BetaX );
    beta_y = Teuchos::rcp( new BetaY );
    
    // set g = 1 on [0,0.5] x {0}
    unitInflow = SpatialFilter::matchingY(0) & (! SpatialFilter::greaterThanX(0.5));
    
    // set g = 0 on {1} x [0,1], (0.5,1.0] x {0}
    zeroInflow = SpatialFilter::matchingX(1) | (SpatialFilter::matchingY(0) & SpatialFilter::greaterThanX(0.5));
    
    u_exact = Teuchos::rcp(new U_Exact_CCW);
  }
  else if (convectiveDirectionChoice == "left")
  {
    beta_x = Function::constant(-1);
    beta_y = Function::zero();
    unitInflow = SpatialFilter::matchingX(1.0) & SpatialFilter::lessThanY(0.5);
    zeroInflow = SpatialFilter::matchingX(1.0) & !SpatialFilter::lessThanY(0.5);
    
    u_exact = Teuchos::rcp( new SpatiallyFilteredFunction<double>(Function::constant(1.0), SpatialFilter::lessThanY(0.5)) );
  }
  else if (convectiveDirectionChoice == "right")
  {
    beta_x = Function::constant(1.0);
    beta_y = Function::constant(0.0);
    
    unitInflow = SpatialFilter::matchingX(0.0) & SpatialFilter::lessThanY(0.5); // | SpatialFilter::matchingY(0.0);
    zeroInflow = SpatialFilter::matchingX(0.0) & !SpatialFilter::lessThanY(0.5);
    
    u_exact = Teuchos::rcp( new SpatiallyFilteredFunction<double>(Function::constant(1.0), SpatialFilter::lessThanY(0.5)) );
  }
  else if (convectiveDirectionChoice == "up")
  {
    beta_x = Function::zero();
    beta_y = Function::constant(1);
    
    unitInflow = SpatialFilter::matchingY(0.0);
    zeroInflow = !SpatialFilter::allSpace();
    
    u_exact = Teuchos::rcp( new SpatiallyFilteredFunction<double>(Function::constant(1.0), SpatialFilter::allSpace()) );
  }
  else if (convectiveDirectionChoice == "down")
  {
    beta_x = Function::zero();
    beta_y = Function::constant(-1);
    
    unitInflow = SpatialFilter::matchingY(1.0) & SpatialFilter::lessThanX(0.5);
    zeroInflow = SpatialFilter::matchingY(1.0) & !SpatialFilter::lessThanX(0.5);
    
    u_exact = Teuchos::rcp( new SpatiallyFilteredFunction<double>(Function::constant(1.0), SpatialFilter::lessThanX(0.5)) );
  }
  else
  {
    cout << "convective direction " << convectiveDirectionChoice << " is not a supported option.\n";
  }
  
  FunctionPtr beta = Function::vectorize(beta_x, beta_y);
    
  VarFactoryPtr vf = VarFactory::varFactory();
  
  VarPtr u = vf->fieldVar("u");
  FunctionPtr n = Function::normal();
  FunctionPtr parity = Function::sideParity();
  VarPtr u_n = vf->fluxVar("u_n", u * beta * n * parity, L2);
  VarPtr v = vf->testVar("v", HGRAD);
  
  BFPtr bf = BF::bf(vf);
  bf->addTerm(-u, beta * v->grad());
  bf->addTerm(u_n, v);
  
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(u_n, unitInflow, Function::constant(1.0) * beta * n * parity);
  bc->addDirichlet(u_n, zeroInflow, Function::zero() * beta * n * parity);
  
  /******* Define the mesh ********/
  // solve on [0,1]^2 with 8x8 initial elements
  double width = 1.0, height = 1.0;
  bool divideIntoTriangles = false;
  double x0 = 0.0, y0 = 0.0;
  int H1Order = polyOrder + 1; // polyOrder refers to the order of the fields, which here are L^2
  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, pToAddTest,
                                              width, height,
                                              horizontalElements, verticalElements,
                                              divideIntoTriangles, x0, y0);
  RHSPtr rhs = RHS::rhs(); // zero forcing
  
  // manually construct "graph" norm to minimize L^2 error of (beta u).  (bf->graphNorm() minimizes L^2 error of u.)
  IPPtr ip = IP::ip();
  ip->addTerm(v->grad());
  ip->addTerm(v);
  
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);
  
  SolverPtr solver = Solver::getDirectSolver();
  soln->setUseCondensedSolve(useCondensedSolve);
  
  int solveSuccess = soln->solve(solver);
  if (solveSuccess != 0)
    cout << "solve returned with error code " << solveSuccess << endl;
  
  ostringstream report;
  report << "elems\tdofs\ttrace_dofs\terr\n";
  
  ostringstream name;
  name << "DPGAdvection_" << convectiveDirectionChoice << "_k" << polyOrder << "_" << numRefinements << "refs";
  if (useEnergyNormForRefinements)
    name << "_energyErrorIndicator";
  else
    name << "_gradientErrorIndicator";
  HDF5Exporter exporter(mesh, name.str(), ".");
  
  exporter.exportSolution(soln,0); // 0 for the refinement number (or time step)
  
  FunctionPtr u_soln = Function::solution(u, soln);
  FunctionPtr u_err = u_soln - u_exact;
  
  int numElements = mesh->getActiveCellIDs().size();
  GlobalIndexType dofCount = mesh->numGlobalDofs();
  GlobalIndexType traceCount = mesh->numFluxDofs();

//  set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
//  int cubatureEnrichment = 5;
//  double l2_err_squared = 0;
//  for (GlobalIndexType cellID : cellIDs)
//  {
//    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, false, cubatureEnrichment);
//    l2_err_squared += (u_err * u_err)->integrate(basisCache);
//  }
//  double err = sqrt(l2_err_squared);
  
  double err = u_err->l2norm(mesh);
  
  cout << "Initial mesh has " << numElements << " active elements and " << dofCount << " degrees of freedom; ";
  cout << "L^2 error = " << err << ".\n";
  
  report << numElements << "\t" << dofCount << "\t" << traceCount << "\t" << err << "\n";
  
  for (int refNumber=1; refNumber<=numRefinements; refNumber++)
  {
    vector<double> errorIndicatorValues;
    double hPower = 1.0 + spaceDim / 2.0;
    
    set<GlobalIndexType> cellIDSet = mesh->getActiveCellIDs();
    vector<GlobalIndexType> cellIDs(cellIDSet.begin(),cellIDSet.end());
  
    if (useEnergyNormForRefinements)
    {
      const map<GlobalIndexType,double>* energyError = &soln->globalEnergyError();
      errorIndicatorValues.resize(cellIDs.size());
      for (int cellOrdinal=0; cellOrdinal<cellIDs.size(); cellOrdinal++)
      {
        GlobalIndexType cellID = cellIDs[cellOrdinal];
        errorIndicatorValues[cellOrdinal] = energyError->find(cellID)->second;
//        cout << "energy error for cell " << cellID << ": " << errorIndicatorValues[cellOrdinal] << endl;
      }
    }
    else
    {
      computeApproximateGradients(soln, u, cellIDs, errorIndicatorValues, hPower);
    }
    
    // refine the top 30% of cells
    int numCellsToRefine = 0.3 * cellIDs.size();
    int numCellsNotToRefine = (cellIDs.size()-numCellsToRefine);
    vector<double> errorIndicatorValuesCopy = errorIndicatorValues;
    std::nth_element(errorIndicatorValues.begin(),errorIndicatorValues.begin() + numCellsNotToRefine,
                     errorIndicatorValues.end());
    
    double threshold = errorIndicatorValues[numCellsNotToRefine-1];
    vector<GlobalIndexType> cellsToRefine;
    for (int i=0; i<cellIDs.size(); i++)
    {
      if (errorIndicatorValuesCopy[i] > threshold)
      {
        cellsToRefine.push_back(cellIDs[i]);
      }
    }
    mesh->hRefine(cellsToRefine);
//    mesh->enforceOneIrregularity();
    
    int numElements = mesh->getActiveCellIDs().size();
    GlobalIndexType dofCount = mesh->numGlobalDofs();
    GlobalIndexType traceCount = mesh->numFluxDofs();
    
    int solveSuccess = soln->solve(solver);
    if (solveSuccess != 0)
      cout << "solve returned with error code " << solveSuccess << endl;

    double err = u_err->l2norm(mesh);
    
    cout << "Ref. " << refNumber << " mesh has " << numElements << " active elements and " << dofCount << " degrees of freedom; ";
    cout << "L^2 error = " << err << ".\n";
    
    report << numElements << "\t" << dofCount << "\t" << traceCount << "\t" << err << "\n";
    
    exporter.exportSolution(soln,refNumber);
  }
  
  ostringstream reportTitle;
  reportTitle << "DPGAdvection_" << convectiveDirectionChoice << "_k" << polyOrder << "_" << numRefinements << "refs";
  if (useEnergyNormForRefinements)
    reportTitle << "_energyErrorIndicator";
  else
    reportTitle << "_gradientErrorIndicator";
  reportTitle << ".dat";
  
  ofstream fout(reportTitle.str().c_str());
  fout << report.str();
  fout.close();
  cout << "Wrote results to " << reportTitle.str() << ".\n";
  
  return 0;
}

// DPG has a built-in error measurement (the energy norm), but we use the below for consistency
// with the DGAdvectionExample:
void computeApproximateGradients(SolutionPtr soln, VarPtr u, const vector<GlobalIndexType> &cells,
                                 vector<double> &gradient_l2_norm, double weightWithPowerOfH)
{
  // imitates https://dealii.org/developer/doxygen/deal.II/namespaceDerivativeApproximation.html
  
  // we require that u be a scalar field variable
  TEUCHOS_TEST_FOR_EXCEPTION(u->rank() != 0, std::invalid_argument, "u must be a scalar variable");
  TEUCHOS_TEST_FOR_EXCEPTION(u->varType() != FIELD, std::invalid_argument, "u must be a field variable");
  
  int cellCount = cells.size();
  gradient_l2_norm.resize(cells.size());
  
  int onePoint = 1;
  MeshTopologyViewPtr meshTopo = soln->mesh()->getTopology();
  int spaceDim = meshTopo->getDimension();
  
  set<GlobalIndexType> cellsAndNeighborsSet;
  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    cellsAndNeighborsSet.insert(cells[cellOrdinal]);
    CellPtr cell = meshTopo->getCell(cells[cellOrdinal]);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    cellsAndNeighborsSet.insert(neighborIDs.begin(),neighborIDs.end());
  }
  vector<GlobalIndexType> cellsAndNeighbors(cellsAndNeighborsSet.begin(),cellsAndNeighborsSet.end());
  
  int cellsAndNeighborsCount = cellsAndNeighbors.size();
  
  FieldContainer<double> cellValues(cellsAndNeighborsCount,onePoint); // values at cell centers
  FieldContainer<double> cellCenters(cellsAndNeighborsCount,spaceDim);
  FieldContainer<double> cellDiameter(cellsAndNeighborsCount,onePoint); // h-values
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForTopology;
  
  FunctionPtr hFunction = Function::h();
  FunctionPtr solnFunction = Function::solution(u, soln);
  Teuchos::Array<int> cellValueDim;
  cellValueDim.push_back(1);
  cellValueDim.push_back(1);
  
  map<GlobalIndexType,int> cellIDToOrdinal; // lookup table for value access
  
  // setup: compute cell centers, and solution values at those points
  for (int cellOrdinal=0; cellOrdinal<cellsAndNeighborsCount; cellOrdinal++)
  {
    GlobalIndexType cellID = cellsAndNeighbors[cellOrdinal];
    cellIDToOrdinal[cellID] = cellOrdinal;
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    if (basisCacheForTopology.find(cellTopo->getKey()) == basisCacheForTopology.end())
    {
      FieldContainer<double> centroid(onePoint,spaceDim);
      int nodeCount = cellTopo->getNodeCount();
      FieldContainer<double> cellNodes(nodeCount,spaceDim);
      CamelliaCellTools::refCellNodesForTopology(cellNodes, cellTopo);
      for (int node=0; node<nodeCount; node++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          centroid(0,d) += cellNodes(node,d);
        }
      }
      for (int d=0; d<spaceDim; d++)
      {
        centroid(0,d) /= nodeCount;
      }
      basisCacheForTopology[cellTopo->getKey()] = BasisCache::basisCacheForReferenceCell(cellTopo, 0); // 0 cubature degree
      basisCacheForTopology[cellTopo->getKey()]->setRefCellPoints(centroid);
    }
    BasisCachePtr basisCache = basisCacheForTopology[cellTopo->getKey()];
    basisCache->setPhysicalCellNodes(soln->mesh()->physicalCellNodesForCell(cellID), {cellID}, false);
    
    FieldContainer<double> cellValue(cellValueDim,&cellValues(cellOrdinal,0));
    solnFunction->values(cellValue, basisCache);
    for (int d=0; d<spaceDim; d++)
    {
      cellCenters(cellOrdinal,d) = basisCache->getPhysicalCubaturePoints()(0,0,d);
    }
    if (weightWithPowerOfH != 0)
    {
      cellDiameter(cellOrdinal,0) = soln->mesh()->getCellMeasure(cellID);
    }
  }
  
  // now compute the gradients requested
  FieldContainer<double> Y(spaceDim,spaceDim); // the matrix we'll invert to compute the gradient
  FieldContainer<double> b(spaceDim); // RHS for matrix problem
  FieldContainer<double> grad(spaceDim); // LHS for matrix problem
  vector<double> distanceVector(spaceDim);
  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    Y.initialize(0.0);
    b.initialize(0.0);
    GlobalIndexType cellID = cells[cellOrdinal];
    CellPtr cell = meshTopo->getCell(cellID);
    int myOrdinalInCellAndNeighbors = cellIDToOrdinal[cellID];
    double myValue = cellValues(myOrdinalInCellAndNeighbors,0);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    for (GlobalIndexType neighborID : neighborIDs)
    {
      int neighborOrdinalInCellAndNeighbors = cellIDToOrdinal[neighborID];
      double neighborValue = cellValues(neighborOrdinalInCellAndNeighbors,0);
      
      double dist_squared = 0;
      for (int d=0; d<spaceDim; d++)
      {
        distanceVector[d] = cellCenters(neighborOrdinalInCellAndNeighbors,d) - cellCenters(myOrdinalInCellAndNeighbors,d);
        dist_squared += distanceVector[d] * distanceVector[d];
      }
      
      for (int d1=0; d1<spaceDim; d1++)
      {
        b(d1) += distanceVector[d1] * (neighborValue - myValue) / dist_squared;
        for (int d2=0; d2<spaceDim; d2++)
        {
          Y(d1,d2) += distanceVector[d1] * distanceVector[d2] / dist_squared;
        }
      }
    }
    SerialDenseWrapper::solveSystem(grad, Y, b);
    double l2_value_squared = 0;
    for (int d=0; d<spaceDim; d++)
    {
      l2_value_squared += grad(d) * grad(d);
    }
    if (weightWithPowerOfH == 0)
    {
      gradient_l2_norm[cellOrdinal] = sqrt(l2_value_squared);
    }
    else
    {
      gradient_l2_norm[cellOrdinal] = sqrt(l2_value_squared) * pow(cellDiameter(myOrdinalInCellAndNeighbors,0), weightWithPowerOfH);
    }
  }
}