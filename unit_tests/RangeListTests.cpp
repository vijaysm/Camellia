//
//  RangeListTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/7/16.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "RangeList.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( RangeList, Insert )
  {
    RangeList<int> mylist;
    TEST_EQUALITY(mylist.length(), 0);
    TEST_EQUALITY(mylist.size(), 0);
    TEST_ASSERT(!mylist.contains(0));
    
    mylist.insert(0);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 1);
    TEST_ASSERT(mylist.contains(0));
    
    mylist.insert(0);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 1);
    TEST_ASSERT(!mylist.contains(1));
    
    mylist.insert(1);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 2);
    TEST_ASSERT(mylist.contains(1));
    TEST_ASSERT(!mylist.contains(3));
    
    mylist.insert(3);
    TEST_EQUALITY(mylist.length(), 2);
    TEST_EQUALITY(mylist.size(), 3);
    TEST_ASSERT(mylist.contains(3));
    TEST_ASSERT(!mylist.contains(2));
    
    mylist.insert(2);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 4);
    
    TEST_ASSERT(mylist.contains(0));
    TEST_ASSERT(mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
  }
  
  TEUCHOS_UNIT_TEST( RangeList, Remove )
  {
    RangeList<int> mylist;
    mylist.insertRange(0, 4);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 4);
    TEST_ASSERT(mylist.contains(0));
    TEST_ASSERT(mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
    
    mylist.remove(1);
    TEST_EQUALITY(mylist.length(), 2);
    TEST_EQUALITY(mylist.size(), 3);
    TEST_ASSERT(mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
    
    mylist.remove(1);
    TEST_EQUALITY(mylist.length(), 2);
    TEST_EQUALITY(mylist.size(), 3);
    TEST_ASSERT(mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
    
    mylist.remove(0);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 2);
    TEST_ASSERT(!mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
    
    mylist.remove(3);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 1);
    TEST_ASSERT(!mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(!mylist.contains(3));
    
    mylist.remove(2);
    TEST_EQUALITY(mylist.length(), 0);
    TEST_EQUALITY(mylist.size(), 0);
    TEST_ASSERT(!mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(!mylist.contains(2));
    TEST_ASSERT(!mylist.contains(3));
  }
  //  TEUCHOS_UNIT_TEST( Int, Assignment )
  //  {
  //    int i1 = 4;
  //    int i2 = i1;
  //    TEST_EQUALITY( i2, i1 );
  //  }
} // namespace
