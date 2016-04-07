//
//  RangeList.h
//  Camellia
//
//  Created by Nate Roberts on 4/7/16.
//
//

#ifndef Camellia_RangeList_h
#define Camellia_RangeList_h

#include <vector>

namespace Camellia
{
  template<class IntegerType> class RangeList
  {
    // _rangeBegins and _rangeEnds have the same length
    // _rangeBegins stores the beginning of each range
    // _rangeEnds stores the final value in the range
    // _rangeEnds[i] - _rangeBegins[i] + 1 gives the # of elements in the ith range
    std::vector<IntegerType> _rangeBegins; // in order container
    std::vector<IntegerType> _rangeEnds;
    int _size;
    
    // ! returns the ordinal of the last range whose first element is less than or equal to value
    int rangeOrdinalLowerBound(IntegerType value) const;
    void mergeIfPossible(int entryOrdinal);
  public:
    // ! Constructor
    RangeList();
    
    // ! Inserts the element
    void insert(IntegerType value);
    
    // ! Inserts count contiguous elements starting with firstValue
    void insertRange(IntegerType firstValue, int count);
    
    // ! Removes the element, if it is present in some range in the list
    void remove(IntegerType value);
    
    // ! Returns true if the value is contained in the range
    bool contains(IntegerType value) const;
    
    // ! number of ranges in the range list
    int length() const;
    
    // ! number of elements in the range list
    int size() const;
  };
}

#include "RangeListDef.h"

#endif
