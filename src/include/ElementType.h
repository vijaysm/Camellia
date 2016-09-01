// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#ifndef DPG_ELEMENT_TYPE
#define DPG_ELEMENT_TYPE

/*
 *  ElementType.h
 *
 */

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DofOrdering.h"

#include "CellTopology.h"

namespace Camellia
{
class ElementType
{
public: // TODO: create accessors for these Ptrs, and make them private...
  Teuchos::RCP< DofOrdering > trialOrderPtr;
  Teuchos::RCP< DofOrdering > testOrderPtr;
  CellTopoPtr cellTopoPtr;

  ElementType(Teuchos::RCP< DofOrdering > trialOrderPtr,
              Teuchos::RCP< DofOrdering > testOrderPtr,
              CellTopoPtr cellTopoPtr)
  {
    this->trialOrderPtr = trialOrderPtr;
    this->testOrderPtr = testOrderPtr;
    this->cellTopoPtr = cellTopoPtr;
  }
  
  bool equals(const ElementType &rhs) const
  {
    if (*this->trialOrderPtr != *rhs.trialOrderPtr) return false;
    if (*this->testOrderPtr != *rhs.testOrderPtr) return false;
    if (this->cellTopoPtr->getKey() != rhs.cellTopoPtr->getKey()) return false;
    return true;
  }
};
}

inline bool operator==(const Camellia::ElementType& lhs, const Camellia::ElementType& rhs)
{
  return lhs.equals(rhs);
}

inline bool operator!=(const Camellia::ElementType& lhs, const Camellia::ElementType& rhs){ return !(lhs == rhs); }

#endif
