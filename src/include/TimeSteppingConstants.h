// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  TimeSteppingConstants.h
//  Camellia
//
//  Created by Nate Roberts on 8/20/15.
//
//

#ifndef Camellia_TimeSteppingConstants_h
#define Camellia_TimeSteppingConstants_h

namespace Camellia
{
  enum TimeStepType
  {
    FORWARD_EULER,
    CRANK_NICOLSON,
    BACKWARD_EULER
  };
}


#endif
