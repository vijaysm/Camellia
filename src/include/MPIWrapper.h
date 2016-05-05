//
//  MPIWrapper.h
//  Camellia
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#ifndef __Camellia__MPIWrapper__
#define __Camellia__MPIWrapper__

#include "TypeDefs.h"

#include <iostream>

// MPI includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#endif

#include "Epetra_SerialComm.h"
#include "Teuchos_Comm.hpp"

#include "Intrepid_FieldContainer.hpp"

namespace Camellia
{
// static class to provide a FieldContainer-based interface to some common MPI tasks
// (Can be used even with MPI disabled)
class MPIWrapper
{
private:
  template<typename Scalar>
  static void allGatherCompact(const Epetra_Comm &Comm,
                               Intrepid::FieldContainer<Scalar> &gatheredValues,
                               Intrepid::FieldContainer<Scalar> &myValues,
                               Intrepid::FieldContainer<int> &offsets);
public:
  // sum the contents of inValues across all processors, and stores the result in outValues
  // the rank of outValues determines the nature of the sum:
  // if outValues has dimensions (D1,D2,D3), say, then inValues must agree in the first three dimensions,
  // but may be of arbitrary shape beyond that.  All values on all processors with matching address
  // (d1,d2,d3) will be summed and stored in outValues(d1,d2,d3).
  //  static void entryWiseSum(FieldContainer<double> &outValues, const FieldContainer<double> &inValues);

  static void allGather(const Epetra_Comm &Comm, Intrepid::FieldContainer<int> &allValues, int myValue);
  static void allGatherHomogeneous(const Epetra_Comm &Comm,
                                   Intrepid::FieldContainer<int> &values,
                                   Intrepid::FieldContainer<int> &myValues); // assumes myValues is same size on every proc.

  template<typename Scalar>
  static void allGatherCompact(const Epetra_Comm &Comm,
                               std::vector<Scalar> &gatheredValues,
                               std::vector<Scalar> &myValues,
                               std::vector<int> &offsets);
  
  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        May be inefficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  static void allGatherCompact(const Epetra_Comm &Comm,
                               Intrepid::FieldContainer<int> &gatheredValues,
                               Intrepid::FieldContainer<int> &myValues,
                               Intrepid::FieldContainer<int> &offsets);
  
  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        May be inefficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  static void allGatherCompact(const Epetra_Comm &Comm,
                               std::vector<int> &gatheredValues,
                               std::vector<int> &myValues,
                               std::vector<int> &offsets);
  
  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        May be inefficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  static void allGatherCompact(const Epetra_Comm &Comm,
                               std::vector<unsigned> &gatheredValues,
                               std::vector<unsigned> &myValues,
                               std::vector<int> &offsets);
  
  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        May be inefficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  static void allGatherCompact(const Epetra_Comm &Comm,
                               std::vector<double> &gatheredValues,
                               std::vector<double> &myValues,
                               std::vector<int> &offsets);

  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        May be inefficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  static void allGatherCompact(const Epetra_Comm &Comm,
                               std::vector<std::pair<std::pair<unsigned,unsigned>,int>> &gatheredValues,
                               std::vector<std::pair<std::pair<unsigned,unsigned>,int>> &myValues,
                               std::vector<int> &offsets);
  
  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        May be inefficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  static void allGatherCompact(const Epetra_Comm &Comm,
                               Intrepid::FieldContainer<double> &gatheredValues,
                               Intrepid::FieldContainer<double> &myValues,
                               Intrepid::FieldContainer<int> &offsets);
  
  static Teuchos::RCP<Epetra_Distributor> getDistributor(const Epetra_Comm &Comm);
  
  static Epetra_CommPtr& CommSerial();
  static Epetra_CommPtr& CommWorld();
  
  static Teuchos_CommPtr& TeuchosCommSerial();
  static Teuchos_CommPtr& TeuchosCommWorld();
  
  static int rank();

  //! sum values entry-wise across all processors in Comm
  template<typename ScalarType>
  static void entryWiseSum(const Epetra_Comm &Comm, Intrepid::FieldContainer<ScalarType> &values);
  
  //! sum values entry-wise across all processors in Comm, after casting
  template<typename ScalarType, typename ScalarTypeToCast>
  static void entryWiseSumAfterCasting(const Epetra_Comm &Comm, Intrepid::FieldContainer<ScalarType> &values);
  
  //! sum values entry-wise across all processors in Comm
  template<typename ScalarType>
  static void entryWiseSum(const Epetra_Comm &Comm, std::vector<ScalarType> &values);
  
  //! sum values entry-wise across all processors
  template<typename ScalarType>
  static void entryWiseSum(Intrepid::FieldContainer<ScalarType> &values);
  
  //! sum values entry-wise across all processors
  template<typename ScalarType>
  static void entryWiseSum(std::vector<ScalarType> &values);
  
  //! sums values entry-wise across all processors
  static void entryWiseSum(Intrepid::FieldContainer<double> &values);
  
  //! sum values entry-wise across all processors in Comm: char specialization because Epetra_Comm does not provide
  static void entryWiseSum(const Epetra_Comm &Comm, std::vector<char> &values)
  {
#ifdef HAVE_MPI
    // check whether Comm is MpiComm
    const Epetra_MpiComm* mpiComm = dynamic_cast<const Epetra_MpiComm*>(&Comm);
    if (mpiComm != NULL)
    {
      std::vector<char> valuesCopy = values; // it appears this copy is necessary
      
      MPI_Comm raw_MpiComm = mpiComm->Comm();
      MPI_Allreduce(&valuesCopy[0], &values[0], values.size(), MPI_CHAR, MPI_SUM, raw_MpiComm);
    }
#else
#endif
  }
  
  static void entryWiseSum(const Epetra_Comm &Comm, std::vector<int> &values);
  static void entryWiseSum(const Epetra_Comm &Comm, std::vector<long> &values);
  static void entryWiseSum(const Epetra_Comm &Comm, std::vector<long long> &values);
  static void entryWiseSum(const Epetra_Comm &Comm, std::vector<double> &values);
  
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static double sum(const Intrepid::FieldContainer<double> &valuesToSum);
  static double sum(double myValue);

  static void entryWiseSum(Intrepid::FieldContainer<int> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static int sum(const Intrepid::FieldContainer<int> &valuesToSum);
  static int sum(int myValue);
  
  //! sum value across communicator
  template<typename ScalarType>
  static ScalarType sum(const Epetra_Comm &Comm, ScalarType value);

  static void entryWiseSum(Intrepid::FieldContainer<GlobalIndexType> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static GlobalIndexType sum(const Intrepid::FieldContainer<GlobalIndexType> &valuesToSum);
  static GlobalIndexType sum(GlobalIndexType myValue);
};
  
  //! sum values entry-wise across all processors
  template<typename ScalarType>
  void MPIWrapper::entryWiseSum(std::vector<ScalarType> &values)
  {
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    entryWiseSum<ScalarType>(Comm,values);
#else
#endif
  }
  
  template<typename ScalarType>
  void MPIWrapper::entryWiseSum(const Epetra_Comm &Comm, std::vector<ScalarType> &values)
  {
    std::vector<ScalarType> valuesCopy = values; // it appears this copy is necessary
    Comm.SumAll(&valuesCopy[0], &values[0], values.size());
  }
  
  //! sum values entry-wise across all processors in Comm
  template<typename ScalarType>
  void MPIWrapper::entryWiseSum(const Epetra_Comm &Comm, Intrepid::FieldContainer<ScalarType> &values)
  {
    Intrepid::FieldContainer<ScalarType> valuesCopy = values; // it appears this copy is necessary
    Comm.SumAll(&valuesCopy[0], &values[0], values.size());
  }
  
  template<typename ScalarType, typename ScalarTypeToCast>
  void MPIWrapper::entryWiseSumAfterCasting(const Epetra_Comm &Comm, Intrepid::FieldContainer<ScalarType> &values)
  {
#ifdef HAVE_MPI
    // cast to ScalarTypeToCast:
    Teuchos::Array<int> dim;
    values.dimensions(dim);
    Intrepid::FieldContainer<ScalarTypeToCast> valuesCast(dim);
    for (int i=0; i<values.size(); i++)
    {
      valuesCast[i] = (ScalarTypeToCast) values[i];
    }
    
    Intrepid::FieldContainer<ScalarTypeToCast> valuesCastCopy = valuesCast; // it appears this copy is necessary
    Comm.SumAll(&valuesCastCopy[0], &valuesCast[0], valuesCast.size());
    
    // copy back to original container:
    for (int i=0; i<values.size(); i++)
    {
      values[i] = (ScalarType) valuesCast[i];
    }
#else
#endif
  }
  
  template<typename ScalarType>
  ScalarType MPIWrapper::sum(const Epetra_Comm &Comm, ScalarType value)
  {
    ScalarType valueCopy = value;
    Comm.SumAll(&valueCopy, &value, 1);
    return value;
  }

  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        May be inefficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  template<typename Scalar>
  void MPIWrapper::allGatherCompact(const Epetra_Comm &Comm, std::vector<Scalar> &gatheredValues,
                                    std::vector<Scalar> &myValues, std::vector<int> &offsets)
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
//    MPIWrapper::entryWiseSum<Scalar>(Comm, gatheredValues);
    MPIWrapper::entryWiseSum(Comm, gatheredValues);
    
    offsets.resize(Comm.NumProc());
    offsets[Comm.MyPID()] = myOffset;
    MPIWrapper::entryWiseSum(Comm, offsets);
  }
}

#endif /* defined(__Camellia_debug__MPIWrapper__) */
