//
//  EnergyErrorFunction.h
//  Camellia
//
//  Created by Nate Roberts on 11/12/15.
//
//

#ifndef __Camellia__EnergyErrorFunction__
#define __Camellia__EnergyErrorFunction__

#include <iostream>

#include "Function.h"
#include "TypeDefs.h"

namespace Camellia
{
  class EnergyErrorFunction : public TFunction<double>
  {
    SolutionPtr _soln;
    RieszRepPtr _rieszRep;
  public:
    EnergyErrorFunction(SolutionPtr soln);
    EnergyErrorFunction(RieszRepPtr rieszRep);
    
    virtual void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    
    static FunctionPtr energyErrorFunction(SolutionPtr soln);
  };
}
#endif /* defined(__Camellia__EnergyErrorFunction__) */
