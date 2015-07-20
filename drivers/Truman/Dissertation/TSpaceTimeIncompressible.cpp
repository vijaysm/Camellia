#include "Solution.h"
#include "RHS.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Amesos_config.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "BF.h"
#include "RefinementStrategy.h"
#include "GMGSolver.h"
#include "SpaceTimeIncompressibleFormulation.h"
#include "IncompressibleProblems.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace Camellia;

int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  int commRank = Teuchos::GlobalMPISession::getRank();

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  // problem parameters:
  int spaceDim = 2;
  double Re = 40;
  bool steady = false;
  string problemChoice = "TaylorGreen";
  int numRefs = 1;
  int p = 2, delta_p = 2;
  int numXElems = 1;
  int numTElems = 1;
  int numSlabs = 1;
  bool useConformingTraces = false;
  string solverChoice = "KLU";
  string multigridStrategyString = "W-cycle";
  bool useCondensedSolve = false;
  bool useConjugateGradient = true;
  bool logFineOperator = false;
  double solverTolerance = 1e-8;
  double nonlinearTolerance = 1e-5;
  int maxLinearIterations = 10000;
  int maxNonlinearIterations = 20;
  int cgMaxIterations = 2000;
  double cgTol = 1e-10;
  bool computeL2Error = false;
  bool exportSolution = false;
  bool saveSolution = false;
  bool loadSolution = false;
  int loadRef = 0;
  int loadDirRef = 0;
  string norm = "Graph";
  string rootDir = ".";
  string tag="";
  cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("Re", &Re, "Re");
  cmdp.setOption("steady", "transient", &steady, "use steady incompressible Navier-Stokes");
  cmdp.setOption("problem", &problemChoice, "Kovasznay, TaylorGreen");
  cmdp.setOption("polyOrder",&p,"polynomial order for field variable u");
  cmdp.setOption("delta_p", &delta_p, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("numXElems",&numXElems,"number of elements in x direction");
  cmdp.setOption("numTElems",&numTElems,"number of elements in t direction");
  cmdp.setOption("numSlabs",&numSlabs,"number of time slabs to use");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLU, MUMPS, GMG-Direct, GMG-ILU, GMG-IC");
  cmdp.setOption("multigridStrategy", &multigridStrategyString, "Multigrid strategy: V-cycle, W-cycle, Full, or Two-level");
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("logFineOperator", "dontLogFineOperator", &logFineOperator);
  cmdp.setOption("CG", "GMRES", &useConjugateGradient);
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");
  cmdp.setOption("nonlinearTolerance", &nonlinearTolerance, "nonlinear solver tolerance");
  cmdp.setOption("maxLinearIterations", &maxLinearIterations, "maximum number of iterations for linear solver");
  cmdp.setOption("maxNonlinearIterations", &maxNonlinearIterations, "maximum number of iterations for Newton solver");
  cmdp.setOption("outputDir", &rootDir, "output directory");
  cmdp.setOption("computeL2Error", "skipL2Error", &computeL2Error, "compute L2 error");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("saveSolution", "skipSave", &saveSolution, "save mesh and solution to HDF5");
  cmdp.setOption("loadSolution", "skipLoad", &loadSolution, "load mesh and solution from HDF5");
  cmdp.setOption("loadRef", &loadRef, "load refinement number");
  cmdp.setOption("loadDirRef", &loadDirRef, "which refinement directory to load from");
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  map<string, Teuchos::RCP<IncompressibleProblem>> problems;
  problems["Kovasznay"] = Teuchos::rcp(new KovasznayProblem(steady, Re));
  problems["TaylorGreen"] = Teuchos::rcp(new TaylorGreenProblem(steady, Re, numXElems, numSlabs));
  problems["Cylinder"] = Teuchos::rcp(new CylinderProblem(steady, Re, numSlabs));
  problems["SquareCylinder"] = Teuchos::rcp(new SquareCylinderProblem(steady, Re, numSlabs));
  Teuchos::RCP<IncompressibleProblem> problem = problems.at(problemChoice);

  // if (commRank == 0)
  // {
  //   Solver::printAvailableSolversReport();
  //   cout << endl;
  // }
  Teuchos::RCP<Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  totalTimer->start(true);

  for (; problem->currentStep() < problem->numSlabs(); problem->advanceStep())
  {
    if (problem->numSlabs() > 1 && commRank == 0 && !steady)
      cout << "Solving time slab [" << problem->currentT0() << ", " << problem->currentT1() << "]" << endl;

    ostringstream problemName;
    string isSteady = "Steady";
    if (!steady)
      isSteady = "Transient";
    problemName << isSteady << problemChoice << spaceDim << "D_slab" << problem->currentStep() << "_" << norm << "_" << Re << "_p" << p << "_" << solverChoice;
    if (tag != "")
      problemName << "_" << tag;
    ostringstream saveDir;
    saveDir << problemName.str() << "_ref" << loadRef;

    int success = mkdir((rootDir+"/"+saveDir.str()).c_str(), S_IRWXU | S_IRWXG);

    string dataFileLocation = rootDir + "/" + saveDir.str() + "/" + saveDir.str() + ".data";
    string exportName = saveDir.str();

    ostringstream loadDir;
    loadDir << problemName.str() << "_ref" << loadDirRef;
    string loadFilePrefix = "";
    if (loadSolution)
    {
      loadFilePrefix = rootDir + "/" + loadDir.str() + "/" + saveDir.str();
      if (commRank == 0) cout << "Loading previous solution " << loadFilePrefix << endl;
    }
    // ostringstream saveDir;
    // saveDir << problemName.str() << "_ref" << loadRef;
    string saveFilePrefix = rootDir + "/" + saveDir.str() + "/" + problemName.str();
    if (saveSolution && commRank == 0) cout << "Saving to " << saveFilePrefix << endl;

    Teuchos::ParameterList parameters;
    parameters.set("spaceDim", spaceDim);
    parameters.set("steady", steady);
    parameters.set("mu", 1./Re);
    parameters.set("useConformingTraces", useConformingTraces);
    parameters.set("fieldPolyOrder", p);
    parameters.set("delta_p", delta_p);
    parameters.set("numTElems", numTElems);
    parameters.set("norm", norm);
    parameters.set("savedSolutionAndMeshPrefix", loadFilePrefix);
    SpaceTimeIncompressibleFormulationPtr form = Teuchos::rcp(new SpaceTimeIncompressibleFormulation(problem, parameters));

    MeshPtr mesh = form->solutionUpdate()->mesh();
    vector<MeshPtr> meshesCoarseToFine;
    MeshPtr k0Mesh = Teuchos::rcp( new Mesh (mesh->getTopology()->deepCopy(), form->bf(), 1, delta_p) );
    meshesCoarseToFine.push_back(k0Mesh);
    meshesCoarseToFine.push_back(mesh);
    // mesh->registerObserver(k0Mesh);

    // Set up boundary conditions
    problem->setBCs(form);

    // Set up solution
    SolutionPtr solutionUpdate = form->solutionUpdate();
    SolutionPtr solutionBackground = form->solutionBackground();
    // dynamic_cast<AnalyticalIncompressibleProblem*>(problem.get())->projectExactSolution(solutionBackground);

    RefinementStrategyPtr refStrategy = form->getRefinementStrategy();
    Teuchos::RCP<HDF5Exporter> exporter;
    if (exportSolution)
      exporter = Teuchos::rcp(new HDF5Exporter(mesh,exportName, rootDir));

    Teuchos::RCP<Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");
    map<string, SolverPtr> solvers;
    solvers["KLU"] = Solver::getSolver(Solver::KLU, true);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
    solvers["SuperLUDist"] = Solver::getSolver(Solver::SuperLUDist, true);
#endif
#ifdef HAVE_AMESOS_MUMPS
    solvers["MUMPS"] = Solver::getSolver(Solver::MUMPS, true);
#endif
    bool useStaticCondensation = false;

    GMGOperator::MultigridStrategy multigridStrategy;
    if (multigridStrategyString == "Two-level")
    {
      multigridStrategy = GMGOperator::TWO_LEVEL;
    }
    else if (multigridStrategyString == "W-cycle")
    {
      multigridStrategy = GMGOperator::W_CYCLE;
    }
    else if (multigridStrategyString == "V-cycle")
    {
      multigridStrategy = GMGOperator::V_CYCLE;
    }
    else if (multigridStrategyString == "Full")
    {
      multigridStrategy = GMGOperator::FULL_MULTIGRID;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized multigrid strategy");
    }

    ofstream dataFile(dataFileLocation);
    dataFile << "ref\t " << "elements\t " << "dofs\t " << "energy\t " << "l2\t " << "solvetime\t" << "iterations\t " << endl;

    // {
    //   // ostringstream saveFile;
    //   // saveFile << saveFilePrefix << "_ref" << -1;
    //   // form->save(saveFile.str());
    //   exporter->exportSolution(solutionBackground, -1);
    //   if (commRank == 0)
    //     cout << "Done exporting" << endl;
    // }

    for (int refIndex=loadRef; refIndex <= numRefs; refIndex++)
    {
      double l2Update = 1e10;
      int iterCount = 0;
      solverTime->start(true);
      Teuchos::RCP<GMGSolver> gmgSolver;
      if (solverChoice[0] == 'G')
      {
        // gmgSolver = Teuchos::rcp( new GMGSolver(solutionUpdate, k0Mesh, maxLinearIterations, solverTolerance, Solver::getDirectSolver(true), useStaticCondensation));
        bool reuseFactorization = true;
        SolverPtr coarseSolver = Solver::getDirectSolver(reuseFactorization);
        gmgSolver = Teuchos::rcp(new GMGSolver(solutionUpdate, meshesCoarseToFine, cgMaxIterations, cgTol, multigridStrategy, coarseSolver, useCondensedSolve));
        gmgSolver->setUseConjugateGradient(useConjugateGradient);
        int azOutput = 20; // print residual every 20 CG iterations
        gmgSolver->setAztecOutput(azOutput);
        gmgSolver->gmgOperator()->setNarrateOnRankZero(logFineOperator,"finest GMGOperator");

        // gmgSolver->setAztecOutput(azOutput);
        // if (solverChoice == "GMG-Direct")
        //   gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::Direct);
        // if (solverChoice == "GMG-ILU")
        //   gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::ILU);
        // if (solverChoice == "GMG-IC")
        //   gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::IC);
      }
      while (l2Update > nonlinearTolerance && iterCount < maxNonlinearIterations)
      {
        if (solverChoice[0] == 'G')
          solutionUpdate->solve(gmgSolver);
        else
          solutionUpdate->condensedSolve(solvers[solverChoice]);

        // Compute L2 norm of update
        double u1L2Update = solutionUpdate->L2NormOfSolutionGlobal(form->u(1)->ID());
        double u2L2Update = solutionUpdate->L2NormOfSolutionGlobal(form->u(2)->ID());
        l2Update = sqrt(u1L2Update*u1L2Update + u2L2Update*u2L2Update);
        if (commRank == 0)
          cout << "Nonlinear Update:\t " << l2Update << endl;

        form->updateSolution();
        iterCount++;
      }
      double solveTime = solverTime->stop();

      double energyError = solutionUpdate->energyErrorTotal();
      double l2Error = 0;
      if (computeL2Error)
      {
        l2Error = problem->computeL2Error(form, solutionBackground);
      }
      if (commRank == 0)
      {
        cout << "Refinement: " << refIndex
          << " \tElements: " << mesh->numActiveElements()
          << " \tDOFs: " << mesh->numGlobalDofs()
          << " \tEnergy Error: " << energyError
          << " \tL2 Error: " << l2Error
          << " \tSolve Time: " << solveTime
          << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
          // << " \tIteration Count: " << iterationCount
          << endl;
        dataFile << refIndex
          << " " << mesh->numActiveElements()
          << " " << mesh->numGlobalDofs()
          << " " << energyError
          << " " << l2Error
          << " " << solveTime
          << " " << totalTimer->totalElapsedTime(true)
          // << " " << iterationCount
          << endl;
      }

      if (exportSolution)
        exporter->exportSolution(solutionBackground, refIndex);

      if (saveSolution)
      {
        ostringstream saveFile;
        saveFile << saveFilePrefix << "_ref" << refIndex;
        form->save(saveFile.str());
      }

      if (refIndex != numRefs)
      {
        // k0Mesh = Teuchos::rcp( new Mesh (mesh->getTopology()->deepCopy(), form->bf(), 1, delta_p) );
        // meshesCoarseToFine.push_back(k0Mesh);
        refStrategy->refine();
        meshesCoarseToFine.push_back(mesh);
      }
    }
    dataFile.close();
  }
  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "Total time = " << totalTime << endl;

  return 0;
}
