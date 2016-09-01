//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ConvectionDiffusionReactionFormulation.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  MPIWrapper::CommWorld()->Barrier(); // can set a breakpoint here for debugger attachment
  
  int rank = mpiSession.getRank();
  
  int meshWidth = 16;
  int polyOrder = 1, delta_k = 1;
  int spaceDim = 2;
  double epsilon = 1e-3;
  double beta_1 = 2.0, beta_2 = 1.0;
  bool useTriangles = true; // otherwise, quads
  int numRefinements = 3;
  double energyThreshold = 0.2; // for refinements
  string formulationChoice = "SUPG";
  
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
  double H1Order;
  
  ConvectionDiffusionReactionFormulation::FormulationChoice formulation;
  if (formulationChoice == "Ultraweak")
  {
    formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    H1Order = polyOrder + 1;
  }
  else if (formulationChoice == "SUPG")
  {
    formulation = ConvectionDiffusionReactionFormulation::SUPG;
    H1Order = polyOrder;
  }
  else if (formulationChoice == "Primal")
  {
    formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
    H1Order = polyOrder;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported formulation choice!");
  }
  
  FunctionPtr beta = Function::constant({beta_1,beta_2});
  ConvectionDiffusionReactionFormulation form(formulation, spaceDim, beta, epsilon, alpha);
  
  // bilinear form
  BFPtr bf = form.bf();
  
  // set up mesh
  vector<double> dimensions = {1.0,1.0};
  vector<int> elementCounts = {meshWidth,meshWidth};
  
  Epetra_Time timer(*MPIWrapper::CommWorld());
  

  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                              elementCounts[0], elementCounts[1], useTriangles);
  
  double initialConstructionTime = timer.ElapsedTime();
  if (rank==0)
  {
    cout << "initial mesh constructed (including lookup tables) in " << initialConstructionTime << " seconds.\n";
  }

  int refinementNumber = 0;

  while (refinementNumber < numRefinements)
  {
    set<GlobalIndexType> allActiveCells = mesh->getActiveCellIDsGlobal();
    timer.ResetStartTime();
    mesh->hRefine(allActiveCells);
    refinementNumber++;
    double refinementTime = timer.ElapsedTime();
    GlobalIndexType numActiveElements = mesh->numActiveElements();
    if (rank==0)
    {
      cout << "Refinement " << refinementNumber << " completed in " << refinementTime << " seconds (";
      cout << numActiveElements << " active elements).\n";
    }
  }
  
  return 0;
}