//
//  ConvectionDiffusionReactionFormulation.h
//  Camellia
//
//  Created by Nate Roberts on Mar 3, 2016.
//
//

#ifndef Camellia_ConvectionDiffusionReactionFormulation_h
#define Camellia_ConvectionDiffusionReactionFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"

namespace Camellia
{
  class ConvectionDiffusionReactionFormulation
  {
  public:
    enum FormulationChoice
    {
      PRIMAL,
      SUPG,
      ULTRAWEAK
    };
  private:
    FormulationChoice _formulationChoice;
    
    int _spaceDim;
    double _epsilon, _alpha; // _alpha: weight on the reaction term
    FunctionPtr _beta;
    
    VarFactoryPtr _vf;
    BFPtr _bf;
    
    static const string S_U;
    static const string S_SIGMA;
    
    static const string S_UHAT;
    static const string S_SIGMA_N;
    
    static const string S_V;
    static const string S_TAU;
  public:
    ConvectionDiffusionReactionFormulation(FormulationChoice formulation, int spaceDim, FunctionPtr beta, double epsilon, double alpha);
    
    VarFactoryPtr vf();
    BFPtr bf();
    
    // field variables:
    VarPtr u();
    VarPtr sigma();
    
    // traces:
    VarPtr sigma_n();
    VarPtr uhat();
    
    // test variables:
    VarPtr v();
    VarPtr tau();
    
    // ! Returns the forcing function corresponding to the specified exact solution u
    FunctionPtr forcingFunction(FunctionPtr u_exact);
  };
}

#endif