//
//  MPIWrapper.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#include "MPIWrapper.h"

#include "Epetra_SerialDistributor.h"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Array.hpp"

using namespace Intrepid;
using namespace Camellia;
using namespace std;

void MPIWrapper::allGather(const Epetra_Comm &Comm, FieldContainer<int> &allValues, int myValue)
{
  FieldContainer<int> myValueFC(1);
  myValueFC[0] = myValue;
  MPIWrapper::allGather(Comm, allValues, myValueFC);
}

void MPIWrapper::allGather(const Epetra_Comm &Comm, FieldContainer<int> &allValues, FieldContainer<int> &myValues)
{
  int numProcs = Comm.NumProc();
  if (numProcs != allValues.dimension(0))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "allValues first dimension must be #procs");
  }
  if (allValues.size() / numProcs != myValues.size())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "myValues size invalid");
  }
#ifdef HAVE_MPI
  Comm.GatherAll(&myValues[0], &allValues[0], allValues.size()/numProcs);
#else
#endif
}

Epetra_CommPtr& MPIWrapper::CommSerial()
{
  static Epetra_CommPtr Comm = Teuchos::rcp( new Epetra_SerialComm() );
  return Comm;
}

Epetra_CommPtr& MPIWrapper::CommWorld()
{
#ifdef HAVE_MPI
  static Epetra_CommPtr Comm = Teuchos::rcp( new Epetra_MpiComm(MPI_COMM_WORLD) );
#else
  static Epetra_CommPtr Comm = Teuchos::rcp( new Epetra_SerialComm() );
#endif
  return Comm;
}

int MPIWrapper::rank()
{
  return Teuchos::GlobalMPISession::getRank();
}

template<typename ScalarType>
void MPIWrapper::entryWiseSum(FieldContainer<ScalarType> &values)
{
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  entryWiseSum<ScalarType>(Comm,values);
#else
#endif
}

void MPIWrapper::entryWiseSum(FieldContainer<double> &values)   // sums values entry-wise across all processors
{
  entryWiseSum<double>(values);
}

void MPIWrapper::entryWiseSum(const Epetra_Comm &Comm, std::vector<int> &values)
{
  entryWiseSum<int>(Comm, values);
}

void MPIWrapper::entryWiseSum(const Epetra_Comm &Comm, std::vector<long> &values)
{
  entryWiseSum<long>(Comm, values);
}

void MPIWrapper::entryWiseSum(const Epetra_Comm &Comm, std::vector<long long> &values)
{
  entryWiseSum<long long>(Comm, values);
}

void MPIWrapper::entryWiseSum(const Epetra_Comm &Comm, std::vector<double> &values)
{
  entryWiseSum<double>(Comm, values);
}

bool MPIWrapper::globalAnd(const Epetra_Comm &Comm, bool value)
{
  int myInt = value ? 0 : 1;
  return 0 == MPIWrapper::sum(Comm, myInt);
}

bool MPIWrapper::globalOr(const Epetra_Comm &Comm, bool value)
{
  return !globalAnd(Comm, !value);
}

Teuchos::RCP<Epetra_Distributor> MPIWrapper::getDistributor(const Epetra_Comm &Comm)
{
  Teuchos::RCP<Epetra_Distributor> distributor;
#ifdef HAVE_MPI
  const Epetra_MpiComm* mpiComm = dynamic_cast<const Epetra_MpiComm*>(&Comm);
  if (mpiComm != NULL)
  {
    distributor = Teuchos::rcp(new Epetra_MpiDistributor(*mpiComm));
  }
  else
  {
    const Epetra_SerialComm* serialComm = dynamic_cast<const Epetra_SerialComm*>(&Comm);
    distributor = Teuchos::rcp(new Epetra_SerialDistributor(*serialComm));
  }
#else
  const Epetra_SerialComm* serialComm = dynamic_cast<const Epetra_SerialComm*>(&Comm);
  if (serialComm == NULL)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_mesh->Comm() is not an instance of Epetra_SerialComm*, even though HAVE_MPI evaluated to false.")
  }
  distributor = Teuchos::rcp(new Epetra_SerialDistributor(*serialComm));
#endif
  return distributor;
}

// sum the contents of valuesToSum across all processors, and returns the result:
// (valuesToSum may vary in length across processors)
double MPIWrapper::sum(const FieldContainer<double> &valuesToSum)
{
  // this is fairly inefficient in the sense that the MPI overhead will dominate the cost here.
  // insofar as it's possible to group such calls into entryWiseSum() calls, this is preferred.
  double mySum = 0;
  for (int i=0; i<valuesToSum.size(); i++)
  {
    mySum += valuesToSum[i];
  }

  return sum(mySum);
}

double MPIWrapper::sum(double mySum)
{
#ifdef HAVE_MPI
  double mySumCopy = mySum;
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.SumAll(&mySumCopy, &mySum, 1);
#else
#endif
  return mySum;
}

void MPIWrapper::entryWiseSum(FieldContainer<int> &values)
{
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  FieldContainer<int> valuesCopy = values; // it appears this copy is necessary
  Comm.SumAll(&valuesCopy[0], &values[0], values.size());
#else
#endif
}

// sum the contents of valuesToSum across all processors, and returns the result:
// (valuesToSum may vary in length across processors)
int MPIWrapper::sum(const FieldContainer<int> &valuesToSum)
{
  // this is fairly inefficient in the sense that the MPI overhead will dominate the cost here.
  // insofar as it's possible to group such calls into entryWiseSum() calls, this is preferred.
  int mySum = 0;
  for (int i=0; i<valuesToSum.size(); i++)
  {
    mySum += valuesToSum[i];
  }

  return sum(mySum);
}

int MPIWrapper::sum(int mySum)
{
#ifdef HAVE_MPI
  int mySumCopy = mySum;
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.SumAll(&mySumCopy, &mySum, 1);

#else
#endif
  return mySum;
}

void MPIWrapper::entryWiseSum(FieldContainer<GlobalIndexType> &values)
{
#ifdef HAVE_MPI
  // cast to long long:
  Teuchos::Array<int> dim;
  values.dimensions(dim);
  FieldContainer<long long> valuesLongLong(dim);
  for (int i=0; i<values.size(); i++)
  {
    valuesLongLong[i] = (long long) values[i];
  }

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  FieldContainer<long long> valuesLongLongCopy = valuesLongLong; // it appears this copy is necessary
  Comm.SumAll(&valuesLongLongCopy[0], &valuesLongLong[0], valuesLongLong.size());

  // copy back to original container:
  for (int i=0; i<values.size(); i++)
  {
    values[i] = (GlobalIndexType) valuesLongLong[i];
  }
#else
#endif
}
// sum the contents of valuesToSum across all processors, and returns the result:
// (valuesToSum may vary in length across processors)
GlobalIndexType MPIWrapper::sum(const FieldContainer<GlobalIndexType> &valuesToSum)
{
  // this is fairly inefficient in the sense that the MPI overhead will dominate the cost here.
  // insofar as it's possible to group such calls into entryWiseSum() calls, this is preferred.
  GlobalIndexType mySum = 0;
  for (int i=0; i<valuesToSum.size(); i++)
  {
    mySum += valuesToSum[i];
  }

  return sum(mySum);
}

GlobalIndexType MPIWrapper::sum(GlobalIndexType mySum)
{
  long long mySumLongLong = mySum;
#ifdef HAVE_MPI
  long long mySumCopy = mySum;
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.SumAll(&mySumCopy, &mySumLongLong, 1);
#else
#endif
  return mySumLongLong;
}

Teuchos_CommPtr& MPIWrapper::TeuchosCommSerial()
{
  static Teuchos_CommPtr Comm = Teuchos::rcp( new Teuchos::SerialComm<int>() );
  return Comm;
}

Teuchos_CommPtr& MPIWrapper::TeuchosCommWorld()
{
#ifdef HAVE_MPI
  static Teuchos_CommPtr Comm = Teuchos::rcp( new Teuchos::MpiComm<int> (MPI_COMM_WORLD) );
#else
  static Teuchos_CommPtr Comm = Teuchos::rcp( new Teuchos::SerialComm<int>() );
#endif
  return Comm;
}
