//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

/*
 *  ElementTypeFactory.cpp
 *
 */

#include "ElementTypeFactory.h"
#include "CellTopology.h"

using namespace Camellia;

ElementTypePtr ElementTypeFactory::getElementType( DofOrderingPtr trialOrderPtr, DofOrderingPtr testOrderPtr, CellTopoPtr cellTopoPtr)
{
  pair< CellTopologyKey, pair<  DofOrdering*, DofOrdering* > > key; // int is cellTopo key
  key = make_pair( cellTopoPtr->getKey(), make_pair( trialOrderPtr.get(), testOrderPtr.get() ) );
  if ( _elementTypes.find(key) == _elementTypes.end() )
  {
    Teuchos::RCP< ElementType > typePtr = Teuchos::rcp( new ElementType( trialOrderPtr, testOrderPtr, cellTopoPtr ) );
    _elementTypes[key] = typePtr;
  }
  return _elementTypes[key];
}

ElementTypePtr ElementTypeFactory::getElementType(DofOrderingPtr trialOrderPtr, DofOrderingPtr testOrderPtr, CellTopoPtrLegacy shardsTopo)
{
  CellTopoPtr cellTopo = CellTopology::cellTopology(*shardsTopo);
  return getElementType(trialOrderPtr, testOrderPtr, cellTopo);
}