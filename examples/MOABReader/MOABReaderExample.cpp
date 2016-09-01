//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "MeshFactory.h"

#ifdef HAVE_MOAB

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  string meshFileName;
  bool readInParallel = true;
  
  cmdp.setOption("meshFile", &meshFileName );
  cmdp.setOption("readDistributed", "readReplicated", &readInParallel );
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  MeshTopologyPtr meshTopo = MeshFactory::importMOABMesh(meshFileName, readInParallel);
  
  int spaceDim = meshTopo->getDimension();
  int cellCount = meshTopo->activeCellCount();
  if (rank==0) cout << spaceDim << "D mesh topology has " << cellCount << " cells.\n";
  
  return 0;
}
#else

int main(int argc, char *argv[])
{
  cout << "Error - HAVE_MOAB preprocessor macro not defined.\n";
  
  return 0;
}

#endif
