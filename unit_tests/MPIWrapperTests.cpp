//
//  MPIWrapperTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/25/15.
//
//

#include "MPIWrapper.h"

#include "Intrepid_FieldContainer.hpp"
#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( MPIWrapper, AllGatherCompact, Scalar )
{
  int myRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  FieldContainer<int> expectedOffsets(numProcs);
  int numGlobalEntries = 0; // counts the total number of entries
  for (int i=0; i < numProcs; i++)
  {
    expectedOffsets[i] = numGlobalEntries;
    for (int j=0; j <= i; j++)
    {
      numGlobalEntries++;
    }
  }

  int myOffset = expectedOffsets[myRank];

  // put a variable number of values on each processor
  FieldContainer<Scalar> myValues(myRank+1);
  for (int i=0; i<=myRank; i++)
  {
    myValues[i] = myOffset + i;
  }

//    std::cout << "rank " << myRank << " values:";
//    for (int i=0; i<=myRank; i++) {
//      std::cout << " " << myValues[i];
//    }
//    std::cout << std::endl;

  FieldContainer<Scalar> allValuesExpected(numGlobalEntries);
  for (int i=0; i<numGlobalEntries; i++)
  {
    allValuesExpected[i] = i;
  }

  FieldContainer<Scalar> allValues;
  FieldContainer<int> offsets;
  MPIWrapper::allGatherCompact(*MPIWrapper::CommWorld(),allValues,myValues,offsets);

  TEST_COMPARE_ARRAYS(expectedOffsets, offsets);

  TEST_COMPARE_ARRAYS(allValuesExpected, allValues);
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( MPIWrapper, AllGatherCompact, int );
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( MPIWrapper, AllGatherCompact, double );

  TEUCHOS_UNIT_TEST(MPIWrapper, SendDataVector)
  {
    Epetra_CommPtr Comm = MPIWrapper::CommWorld();
    
    // send the message ((char)myPID * 3) to processor (myPID + 1) % numProcs
    int myPID = Comm->MyPID();
    int numProcs = Comm->NumProc();
    vector<int> recipients = {(myPID + 1) % numProcs};
    char messageToSend = (char)myPID * 3;
    vector<vector<char>> messagesToSend = {{messageToSend}};
    vector<char> messagesReceived;
    MPIWrapper::sendDataVectors(Comm,recipients,messagesToSend,messagesReceived);
    
    TEST_EQUALITY(messagesReceived.size(), 1);
    int expectedSendingPID = (myPID + numProcs - 1) % numProcs;
    char expectedMessage = (char) expectedSendingPID * 3;
    TEST_EQUALITY(messagesReceived[0], expectedMessage);
    
    // now, let's try something similar, but with pairs and multiple recipients
    if (1 % numProcs != 7 % numProcs)
    {
      recipients = {(myPID + 1) % numProcs, (myPID + 7) % numProcs};
    }
    else
    {
      recipients = {(myPID + 1) % numProcs};
    }
    // recipients are required by MPIWrapper to be in numerical order
    std::sort(recipients.begin(), recipients.end());
    typedef pair<int,unsigned> PairDataType;
    vector<vector<PairDataType>> pairMessagesToSend;
    for (int recipient : recipients)
    {
      unsigned value = recipient * 2 + myPID;
      pairMessagesToSend.push_back({{myPID,value}});
    }
    vector<PairDataType> pairMessagesReceived;
    MPIWrapper::sendDataVectors(Comm,recipients,pairMessagesToSend,pairMessagesReceived);
    
    // We receive one message from each sender, and we should receive as many messages as we sent:
    TEST_EQUALITY(pairMessagesReceived.size(), recipients.size());
    // senders are (myPID - 1) % numProcs, (myPID - 7) % numProcs
    set<int> expectedSenders = {(myPID + numProcs - 1) % numProcs};
    if (1 % numProcs != 7 % numProcs)
    {
      // since numProcs >= 1, 7 * numProcs - 7 >= 0, so we keep thing positive post-modulus.
      int expectedSender = (myPID + 7 * numProcs - 7) % numProcs;
      expectedSenders.insert(expectedSender);
    }
    TEST_EQUALITY(expectedSenders.size(), pairMessagesReceived.size());
    for (PairDataType message : pairMessagesReceived)
    {
      int sender = message.first;
      TEST_ASSERT(expectedSenders.find(sender) != expectedSenders.end());
      unsigned expectedValue = myPID * 2 + sender;
      unsigned actualValue = message.second;
      TEST_EQUALITY(actualValue, expectedValue);
    }
  }
} // namespace
