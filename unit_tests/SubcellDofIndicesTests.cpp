//
//  SubcellDofIndicesTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/19/16.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BasisFactory.h"
#include "CellTopology.h"
#include "SubcellDofIndices.h"
#include "TypeDefs.h"

using namespace Camellia;
using namespace std;

namespace
{
  void testPackAndUnpack(SubcellDofIndices &scDofIndices, Teuchos::FancyOStream &out, bool &success)
  {
    int dataSize = scDofIndices.dataSize();
    // allocate a buffer 3x that, initialized with -3
    vector<char> dataBuffer(3*dataSize,-3);
    char* dataLocation = &dataBuffer[0];
    scDofIndices.write(dataLocation, dataSize);
    int bytesWritten = dataLocation - &dataBuffer[0];
    TEST_EQUALITY(dataSize, bytesWritten);
    SubcellDofIndices readScDofIndices;
    const char* constDataLocation = &dataBuffer[0];
    readScDofIndices.read(constDataLocation, dataSize);
    int bytesRead = constDataLocation - &dataBuffer[0];
    TEST_EQUALITY(dataSize, bytesRead);
    TEST_ASSERT(scDofIndices.subcellDofIndices == readScDofIndices.subcellDofIndices);
  }
  
  TEUCHOS_UNIT_TEST( SubcellDofIndices, PackAndUnpack )
  {
    SubcellDofIndices scDofIndices;
    int spaceDim = 3;
    scDofIndices.subcellDofIndices.resize(spaceDim+1);
    int varID = 3;
    CellTopoPtr hex = CellTopology::hexahedron();
    int H1Order = 3;
    BasisPtr basis = BasisFactory::basisFactory()->getBasis(H1Order, hex, FUNCTION_SPACE_HGRAD);
    for (int d=0; d<=spaceDim; d++)
    {
      int numSubcells = hex->getSubcellCount(d);
      for (int scord=0; scord<numSubcells; scord++)
      {
        vector<int> localDofOrdinals = basis->dofOrdinalsForSubcell(d, scord);
        vector<GlobalIndexType> globalDofIndices(localDofOrdinals.begin(), localDofOrdinals.end());
        scDofIndices.subcellDofIndices[d][scord][varID] = globalDofIndices;
      }
    }
    testPackAndUnpack(scDofIndices, out, success);
  }
} // namespace
