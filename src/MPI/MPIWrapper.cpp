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
  MPIWrapper::allGatherHomogeneous(Comm, allValues, myValueFC);
}

void MPIWrapper::allGatherHomogeneous(const Epetra_Comm &Comm, FieldContainer<int> &allValues, FieldContainer<int> &myValues)
{
  int numProcs = Teuchos::GlobalMPISession::getNProc();
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

// \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
//        May be inefficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
template<typename Scalar>
void MPIWrapper::allGatherCompact(const Epetra_Comm &Comm, FieldContainer<Scalar> &gatheredValues,
                                  FieldContainer<Scalar> &myValues, FieldContainer<int> &offsets)
{
  int mySize = myValues.size();
  int totalSize;
  Comm.SumAll(&mySize, &totalSize, 1);

  int myOffset = 0;
  Comm.ScanSum(&mySize,&myOffset,1);
  myOffset -= mySize;

  gatheredValues.resize(totalSize);
  for (int i=0; i<mySize; i++)
  {
    gatheredValues[myOffset+i] = myValues[i];
  }
  MPIWrapper::entryWiseSum(Comm, gatheredValues);

  offsets.resize(Comm.NumProc());
  offsets[Comm.MyPID()] = myOffset;
  MPIWrapper::entryWiseSum(Comm, offsets);
}

void MPIWrapper::allGatherCompact(const Epetra_Comm &Comm,
                                  vector<pair<pair<unsigned,unsigned>,int>> &gatheredValues,
                                  vector<pair<pair<unsigned,unsigned>,int>> &myValues,
                                  vector<int> &offsets)
{
  // rewrite myValues as triples of ints
  vector<int> myValuesExpanded(myValues.size() * 3);
  vector<int> gatheredValuesExpanded;
  for (int i=0; i<myValues.size(); i++)
  {
    myValuesExpanded[i*3+0] = (int) myValues[i].first.first;
    myValuesExpanded[i*3+1] = (int) myValues[i].first.second;
    myValuesExpanded[i*3+2] =       myValues[i].second;
  }
  allGatherCompact(Comm, gatheredValuesExpanded, myValuesExpanded, offsets);
  int expandedTotalSize = gatheredValuesExpanded.size();
  int totalSize = expandedTotalSize / 3;
  gatheredValues.resize(totalSize);
  
  for (int i=0; i<totalSize; i++)
  {
    gatheredValues[i].first.first  = (unsigned) gatheredValuesExpanded[i*3+0];
    gatheredValues[i].first.second = (unsigned) gatheredValuesExpanded[i*3+1];
    gatheredValues[i].second       =            gatheredValuesExpanded[i*3+2];
  }
  int numOffsets = offsets.size();
  for (int i=0; i<numOffsets; i++)
  {
    offsets[i] /= 3;
  }
}

void MPIWrapper::allGatherCompact(const Epetra_Comm &Comm,
                                  std::vector<int> &gatheredValues,
                                  std::vector<int> &myValues,
                                  std::vector<int> &offsets)
{
  MPIWrapper::allGatherCompact<int>(Comm,gatheredValues,myValues,offsets);
}

void MPIWrapper::allGatherCompact(const Epetra_Comm &Comm,
                                  vector<unsigned> &gatheredValues,
                                  vector<unsigned> &myValues,
                                  vector<int> &offsets)
{
  vector<int> myValuesInt(myValues.begin(),myValues.end());
  vector<int> gatheredValuesInt;
  MPIWrapper::allGatherCompact(Comm, gatheredValuesInt, myValuesInt, offsets);
  int gatheredSize = gatheredValuesInt.size();
  gatheredValues.resize(gatheredSize);
  for (int i=0; i<gatheredSize; i++)
  {
    gatheredValues[i] = (unsigned) gatheredValuesInt[i];
  }
}

void MPIWrapper::allGatherCompact(const Epetra_Comm &Comm,
                                  std::vector<double> &gatheredValues,
                                  std::vector<double> &myValues,
                                  std::vector<int> &offsets)
{
  MPIWrapper::allGatherCompact<double>(Comm,gatheredValues,myValues,offsets);
}

void MPIWrapper::allGatherCompact(const Epetra_Comm &Comm,
                                  FieldContainer<int> &gatheredValues,
                                  FieldContainer<int> &myValues,
                                  FieldContainer<int> &offsets)
{
  MPIWrapper::allGatherCompact<int>(Comm,gatheredValues,myValues,offsets);
}

void MPIWrapper::allGatherCompact(const Epetra_Comm &Comm,
                                  FieldContainer<double> &gatheredValues,
                                  FieldContainer<double> &myValues,
                                  FieldContainer<int> &offsets)
{
  MPIWrapper::allGatherCompact<double>(Comm,gatheredValues,myValues,offsets);
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

// sum the contents of inValues across all processors, and stores the result in outValues
// the rank of outValues determines the nature of the sum:
// if outValues has dimensions (D1,D2,D3), say, then inValues must agree in the first three dimensions,
// but may be of arbitrary shape beyond that.  All values on all processors with matching address
// (d1,d2,d3) will be summed and stored in outValues(d1,d2,d3).
//void MPIWrapper::entryWiseSum(FieldContainer<double> &outValues, const FieldContainer<double> &inValues) {
//  outValues.initialize();
//  int outRank = outValues.rank();
//  for (int i=0; i<outRank; i++) {
//    TEUCHOS_TEST_FOR_EXCEPTION(outValues.dimension(i) != inValues.dimension(i), std::invalid_argument, "inValues must match outValues in all outValues's dimensions");
//  }
//  double inEntriesPerOutEntry = 1;
//  for (int i=outRank; i<inValues.rank(); i++) {
//    inEntriesPerOutEntry *= inValues.dimension(i);
//  }
//
//}

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
