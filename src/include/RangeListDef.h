//
//  RangeListDef.h
//  Camellia
//
//  Created by Nate Roberts on 4/7/16.
//
//

#ifndef Camellia_RangeListDef_h
#define Camellia_RangeListDef_h

namespace Camellia {
  template<class IntegerType>
  RangeList<IntegerType>::RangeList()
  {
    _size = 0;
  }
  
  template<class IntegerType>
  void RangeList<IntegerType>::insert(IntegerType value)
  {
    int rangeOrdinal = this->rangeOrdinalLowerBound(value); // first range that ends past value
    int lastEntryOrdinal = this->length() - 1;
    if (rangeOrdinal > lastEntryOrdinal)
    {
      // first, check whether we can add to the last element, if there is one
      if ((lastEntryOrdinal >= 0) && (value == 1 + _rangeEnds[lastEntryOrdinal]))
      {
        _rangeEnds[lastEntryOrdinal]++;
      }
      else
      {
        _rangeBegins.push_back(value);
        _rangeEnds.push_back(value);
      }
      _size++;
      // in this case, no chance of two ranges wanting to be merged
    }
    else
    {
      // we should be able to add value to the rangeOrdinal range, or the previous one
      if (_rangeBegins[rangeOrdinal] <= value) // we know that (_rangeEnds[rangeOrdinal] >= value)
      {
        // value already present; can return
        return;
      }
      else
      {
        // check whether we can add to previous one entries
        if ((rangeOrdinal == 0) && (value == _rangeBegins[0]-1))
        {
          _rangeBegins[0] -= 1;
          _size++;
        }
        else if ((rangeOrdinal > 0) && (value == 1 + _rangeEnds[rangeOrdinal-1]))
        {
          _rangeEnds[rangeOrdinal-1]++;
          _size++;
          mergeIfPossible(rangeOrdinal - 1);
        }
        else // need a new singleton entry prior to rangeOrdinal entry
        {
          _rangeBegins.insert(_rangeBegins.begin()+rangeOrdinal,value);
          _rangeEnds.insert(_rangeEnds.begin()+rangeOrdinal,value);
          _size++;
        }
      }
    }
  }
  
  template<class IntegerType>
  void RangeList<IntegerType>::insertRange(IntegerType firstValue, int count)
  {
    // more efficient implementations are possible; for now we just repeatedly call insert()
    for (int i=0; i<count; i++)
    {
      this->insert(firstValue+i);
    }
  }
  
  template<class IntegerType>
  void RangeList<IntegerType>::mergeIfPossible(int entryOrdinal)
  {
    // check if we can merge with the next entry
    if (entryOrdinal+1 < _rangeBegins.size())
    {
      if ((_rangeEnds[entryOrdinal]==_rangeBegins[entryOrdinal+1]) || (_rangeEnds[entryOrdinal]+1 ==_rangeBegins[entryOrdinal+1]))
      {
        // can merge, then
        _rangeEnds[entryOrdinal] = _rangeEnds[entryOrdinal+1];
        _rangeBegins.erase(_rangeBegins.begin() + entryOrdinal + 1);
        _rangeEnds.erase(_rangeEnds.begin() + entryOrdinal + 1);
      }
    }
    // check if we can merge with the previous entry
    if (entryOrdinal - 1 >= 0)
    {
      if ((_rangeEnds[entryOrdinal-1]==_rangeBegins[entryOrdinal]) || (_rangeEnds[entryOrdinal-1]+1==_rangeBegins[entryOrdinal]))
      {
        _rangeEnds[entryOrdinal] = _rangeEnds[entryOrdinal+1];
        _rangeBegins.erase(_rangeBegins.begin() + entryOrdinal);
        _rangeEnds.erase(_rangeEnds.begin() + entryOrdinal);
      }
    }
  }

  template<class IntegerType>
  void RangeList<IntegerType>::remove(IntegerType value)
  {
    if (!contains(value)) return;
    int rangeOrdinal = this->rangeOrdinalLowerBound(value); // first range whose end is greater than value
    // since contains(value) returned true, the range at rangeOrdinal must contain the value
    // is this a singleton range?
    if (_rangeBegins[rangeOrdinal] == _rangeEnds[rangeOrdinal])
    {
      _rangeBegins.erase(_rangeBegins.begin() + rangeOrdinal);
      _rangeEnds.erase(_rangeEnds.begin() + rangeOrdinal);
    }
    else if (_rangeBegins[rangeOrdinal] == value)
    {
      _rangeBegins[rangeOrdinal] += 1;
    }
    else if (_rangeEnds[rangeOrdinal] == value)
    {
      _rangeEnds[rangeOrdinal] -= 1;
    }
    else
    {
      // then we need to split the range
      _rangeBegins.insert(_rangeBegins.begin() + rangeOrdinal + 1, value + 1);
      _rangeEnds.insert(_rangeEnds.begin() + rangeOrdinal + 1, _rangeEnds[rangeOrdinal]);
      _rangeEnds[rangeOrdinal] = value - 1;
    }
    _size--;
  }
  
  template<class IntegerType>
  bool RangeList<IntegerType>::contains(IntegerType value) const
  {
    int rangeOrdinalLowerBound = this->rangeOrdinalLowerBound(value); // first range whose end is greater than value
    if (rangeOrdinalLowerBound >= this->length())
    {
      // not found
      return false;
    }
    else
    {
      return (_rangeBegins[rangeOrdinalLowerBound] <= value);
    }
  }
  
  template<class IntegerType>
  int RangeList<IntegerType>::length() const
  {
    return _rangeBegins.size();
  }
  
  template<class IntegerType>
  int RangeList<IntegerType>::rangeOrdinalLowerBound(IntegerType value) const
  {
    // find the first range that ends beyond the specified value, then take the difference between its location and the start
    auto lowerBoundIt = std::lower_bound(_rangeEnds.begin(),_rangeEnds.end(),value);
    return lowerBoundIt - _rangeEnds.begin();
  }
  
  template<class IntegerType>
  int RangeList<IntegerType>::size() const
  {
    return _size;
  }
  
}

#endif