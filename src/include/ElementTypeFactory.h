#ifndef DPG_ELEMENT_TYPE_FACTORY
#define DPG_ELEMENT_TYPE_FACTORY

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER


/*
 *  ElementTypeFactory.h
 *
 */

#include "TypeDefs.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "ElementType.h"

#include "CellTopology.h"

namespace Camellia
{
class ElementTypeFactory
{
  std::map< std::pair< Camellia::CellTopologyKey, std::pair<  DofOrdering*, DofOrdering* > >,
      Teuchos::RCP< ElementType > > _elementTypes;

public:
  Teuchos::RCP< ElementType > getElementType(DofOrderingPtr trialOrderPtr, DofOrderingPtr testOrderPtr, CellTopoPtr cellTopoPtr);
  Teuchos::RCP< ElementType > getElementType(DofOrderingPtr trialOrderPtr, DofOrderingPtr testOrderPtr, CellTopoPtrLegacy cellTopoPtr);
};
}

#endif