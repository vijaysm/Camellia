// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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
public:
  template<typename Scalar>
  static void allGather(const Epetra_Comm &Comm, std::vector<Scalar> &allValues, Scalar myValue);
  
  template<typename Scalar>
  static void allGather(const Epetra_Comm &Comm, std::vector<Scalar> &allValues, std::vector<Scalar> myValues);
  
  static void allGather(const Epetra_Comm &Comm, Intrepid::FieldContainer<int> &allValues, int myValue);
  static void allGather(const Epetra_Comm &Comm,
                        Intrepid::FieldContainer<int> &values,
                        Intrepid::FieldContainer<int> &myValues); // assumes myValues is same size on every proc.

  template<typename DataType>
  static void allGatherVariable(const Epetra_Comm &Comm,
                               std::vector<DataType> &gatheredValues,
                               std::vector<DataType> &myValues,
                               std::vector<int> &offsets);
  
  template<typename Scalar>
  static void allGatherVariable(const Epetra_Comm &Comm,
                                Intrepid::FieldContainer<Scalar> &gatheredValues,
                                Intrepid::FieldContainer<Scalar> &myValues,
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
  
  static bool globalAnd(const Epetra_Comm &Comm, bool value);
  static bool globalOr(const Epetra_Comm &Comm, bool value);
  
  template<typename DataType>
  static void sendDataVectors(Epetra_CommPtr Comm, const std::vector<int> &recipients,
                              const std::vector<std::vector<DataType>> &dataVectors,
                              std::vector<DataType> &receivedVector);
  template<typename DataType>
  static void sendDataVectors(Epetra_CommPtr Comm,
                              const std::map<int,std::vector<DataType>> &recipientDataVectors,
                              std::vector<DataType> &receivedVector);
  
  template<typename KeyType, typename ValueType>
  static void sendDataMaps(Epetra_CommPtr Comm,
                           const std::map<int,std::map<KeyType,ValueType>> &recipientDataMaps,
                           std::map<KeyType,ValueType> &receivedMap);
  
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
};
  
  template<typename Scalar>
  void MPIWrapper::allGather(const Epetra_Comm &Comm, std::vector<Scalar> &allValues, Scalar myValue)
  {
    std::vector<Scalar> myValueVector(1);
    myValueVector[0] = myValue;
    MPIWrapper::allGather(Comm, allValues, myValueVector);
  }
  
  template<typename Scalar>
  void MPIWrapper::allGather(const Epetra_Comm &Comm, std::vector<Scalar> &allValues, std::vector<Scalar> myValues)
  {
    int numProcs = Comm.NumProc();
    if (allValues.size() / numProcs != myValues.size())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "myValues size invalid");
    }
#ifdef HAVE_MPI
    if (numProcs > 1)
    {
      int byteCount = myValues.size() * sizeof(Scalar);
      const Epetra_MpiComm* mpiComm = dynamic_cast<const Epetra_MpiComm*>(&Comm);
      TEUCHOS_TEST_FOR_EXCEPTION(mpiComm == NULL, std::invalid_argument, "numProcs > 1, but Comm is not of type Epetra_MpiComm");
      MPI_Allgather(&myValues[0], byteCount, MPI_BYTE, &allValues[0], byteCount, MPI_BYTE, mpiComm->Comm());
    }
    else
    {
      allValues = myValues;
    }
//    Comm.GatherAll(&myValues[0], &allValues[0], allValues.size()/numProcs);
#else
#endif
  }
  
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
  
  template<typename KeyType, typename ValueType>
  void MPIWrapper::sendDataMaps(Epetra_CommPtr Comm,
                                const std::map<int,std::map<KeyType,ValueType>> &recipientDataMaps,
                                std::map<KeyType,ValueType> &receivedMap)
  {
    // convert to a vector, and send that.
    typedef std::vector<std::pair<KeyType,ValueType>> MapVector;
    std::vector<MapVector> dataVectors;
    std::vector<int> recipients;
    for (auto recipientDataMapEntry : recipientDataMaps)
    {
      int recipient = recipientDataMapEntry.first;
      recipients.push_back(recipient);
      auto dataMap = &recipientDataMapEntry.second;
      dataVectors.push_back(MapVector(dataMap->begin(),dataMap->end()));
    }
    MapVector receivedVector;
    sendDataVectors(Comm,recipients,dataVectors,receivedVector);
    receivedMap.insert(receivedVector.begin(),receivedVector.end());
  }

  template<typename DataType>
  void MPIWrapper::sendDataVectors(Epetra_CommPtr Comm,
                                   const std::map<int,std::vector<DataType>> &recipientDataVectors,
                                   std::vector<DataType> &receivedVector)
  {
    int numRecipients = recipientDataVectors.size();
    std::vector<std::vector<DataType>> dataVectors(numRecipients);
    std::vector<int> recipients(numRecipients);
    int ordinal = 0;
    for (auto recipientVectorEntry : recipientDataVectors)
    {
      recipients[ordinal] = recipientVectorEntry.first;
      dataVectors[ordinal] = recipientVectorEntry.second;
      ordinal++;
    }
    sendDataVectors(Comm,recipients,dataVectors,receivedVector);
  }
  
  template<typename DataType>
  void MPIWrapper::sendDataVectors(Epetra_CommPtr Comm, const std::vector<int> &recipients,
                                   const std::vector<std::vector<DataType>> &dataVectors,
                                   std::vector<DataType> &receivedVector)
  {
    // check that recipients are in numerical order:
    int numRecipients = recipients.size();
    for (int i=1; i<numRecipients; i++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(recipients[i-1] >= recipients[i], std::invalid_argument, "recipients list must be in numerical order, and populated with distinct recipients");
    }
    
    Teuchos::RCP<Epetra_Distributor> distributor = MPIWrapper::getDistributor(*Comm);
    
    TEUCHOS_TEST_FOR_EXCEPTION(dataVectors.size() != numRecipients, std::invalid_argument, "dataVectors and recipients list must be the same length");
    
    const int *recipientPtr = (numRecipients > 0) ? &recipients[0] : NULL;
    
    bool deterministic = true; // doesn't do anything at present, according to docs.
    int numProcsThatWillSendToMe;
    distributor->CreateFromSends(numRecipients, recipientPtr, deterministic, numProcsThatWillSendToMe);
    
    std::vector<int> lengthForEachSendProc(dataVectors.size()); // number of entries in each vector
    int vectorOrdinal = 0;
    std::vector<DataType> packedData;
    for (const std::vector<DataType> dataVector : dataVectors)
    {
      lengthForEachSendProc[vectorOrdinal] = dataVector.size();
      packedData.insert(packedData.end(),dataVector.begin(),dataVector.end());
      vectorOrdinal++;
    }
    
    char* export_chars = (packedData.size() > 0) ? reinterpret_cast<char*>(&packedData[0]) : NULL;
    int* lengthForEachSendProcPtr = (lengthForEachSendProc.size() > 0) ? &lengthForEachSendProc[0] : NULL;
    char* import_chars = NULL;

    int len_import_chars = 0;
    int err = distributor->Do(export_chars, (int)sizeof(DataType), lengthForEachSendProcPtr, len_import_chars, import_chars);
    
    DataType* recvd_rows = reinterpret_cast<DataType*>(import_chars);
    int numEntriesReceived = len_import_chars/sizeof(DataType);
    
    receivedVector.resize(numEntriesReceived);
    for (int i=0; i<numEntriesReceived; i++)
    {
      receivedVector[i] = recvd_rows[i];
    }
    
    if (import_chars != 0) delete [] import_chars;
  }
  
  template<typename ScalarType>
  ScalarType MPIWrapper::sum(const Epetra_Comm &Comm, ScalarType value)
  {
    ScalarType valueCopy = value;
    Comm.SumAll(&valueCopy, &value, 1);
    return value;
  }

  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  template<typename DataType>
  void MPIWrapper::allGatherVariable(const Epetra_Comm &Comm, std::vector<DataType> &gatheredValues,
                                     std::vector<DataType> &myValues, std::vector<int> &offsets)
  {
    std::vector<int> valuesSizes(Comm.NumProc());
    
    // sizes will be in bytes
    int mySize = myValues.size() * sizeof(DataType);
    MPIWrapper::allGather(Comm, valuesSizes, mySize);
    
    offsets.resize(Comm.NumProc());
    int offset = 0;
    for (int i=0; i<Comm.NumProc(); i++)
    {
      offsets[i] = offset;
      offset += valuesSizes[i];
    }
    int totalSize = offset;
    gatheredValues.resize(totalSize / sizeof(DataType));
    
#ifdef HAVE_MPI
    if (Comm.NumProc() > 1)
    {
      const Epetra_MpiComm* mpiComm = dynamic_cast<const Epetra_MpiComm*>(&Comm);
      TEUCHOS_TEST_FOR_EXCEPTION(mpiComm == NULL, std::invalid_argument, "numProcs > 1, but Comm is not of type Epetra_MpiComm");
      MPI_Allgatherv(reinterpret_cast<char*>(&myValues[0]), mySize, MPI_CHAR, reinterpret_cast<char*>(&gatheredValues[0]), &valuesSizes[0], &offsets[0], MPI_CHAR, mpiComm->Comm());
    }
    else
    {
      gatheredValues = myValues;
    }
#else
#endif
    // now, go through offsets to make the sizes in terms of DataType rather than bytes
    for (int i=0; i<offsets.size(); i++)
    {
      offsets[i] /= sizeof(DataType);
    }
  }
  
  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  template<typename Scalar>
  void MPIWrapper::allGatherVariable(const Epetra_Comm &Comm, Intrepid::FieldContainer<Scalar> &gatheredValues,
                                     Intrepid::FieldContainer<Scalar> &myValues, Intrepid::FieldContainer<int> &offsets)
  {
    std::vector<int> valuesSizes(Comm.NumProc());
    
    // sizes will be in bytes
    int mySize = myValues.size() * sizeof(Scalar);
    MPIWrapper::allGather(Comm, valuesSizes, mySize);
    
    offsets.resize(Comm.NumProc());
    int offset = 0;
    for (int i=0; i<Comm.NumProc(); i++)
    {
      offsets[i] = offset;
      offset += valuesSizes[i];
    }
    int totalSize = offset;
    gatheredValues.resize(totalSize / sizeof(Scalar));

#ifdef HAVE_MPI
    if (Comm.NumProc() > 1)
    {
      const Epetra_MpiComm* mpiComm = dynamic_cast<const Epetra_MpiComm*>(&Comm);
      TEUCHOS_TEST_FOR_EXCEPTION(mpiComm == NULL, std::invalid_argument, "numProcs > 1, but Comm is not of type Epetra_MpiComm");
      char* myValuesCharCast = reinterpret_cast<char*>(&myValues[0]);
      char* gatheredValuesCharCast = reinterpret_cast<char*>(&gatheredValues[0]);
      MPI_Allgatherv(myValuesCharCast, mySize, MPI_CHAR, gatheredValuesCharCast, &valuesSizes[0], &offsets[0], MPI_CHAR, mpiComm->Comm());
    }
    else
    {
      gatheredValues = myValues;
    }
#else
#endif
    // now, go through offsets to make the sizes in Scalar rather than bytes
    for (int i=0; i<offsets.size(); i++)
    {
      offsets[i] /= sizeof(Scalar);
    }
  }
}

#endif /* defined(__Camellia_debug__MPIWrapper__) */
