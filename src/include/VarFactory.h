//
//  VarFactory.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_VarFactory_h
#define Camellia_VarFactory_h

#include "Var.h"
#include "LinearTerm.h"

namespace Camellia
{
// The basic function of VarFactory is to assign unique test/trial IDs and keep track of the variables in play.
// Usually, you have exactly one VarFactory for each problem, and you pass this into the bilinear form (BF) object
// on construction.

class VarFactory
{
  std::map< std::string, VarPtr > _testVars;
  std::map< std::string, VarPtr > _trialVars;
  std::map< int, VarPtr > _testVarsByID;
  std::map< int, VarPtr > _trialVarsByID;
  int _nextTrialID;
  int _nextTestID;

  int getTestID(int IDarg);
  int getTrialID(int IDarg);
protected:
  VarFactory(const std::map< std::string, VarPtr > &trialVars, const std::map< std::string, VarPtr > &testVars,
             const std::map< int, VarPtr > &trialVarsByID, const std::map< int, VarPtr > &testVarsByID,
             int nextTrialID, int nextTestID);
  void addTestVar(VarPtr var);
  void addTrialVar(VarPtr var);

public:
  enum BubnovChoice { BUBNOV_TRIAL, BUBNOV_TEST };

  VarFactory();
  VarFactoryPtr getBubnovFactory(BubnovChoice choice);
  
  // ! returns true if some trace or flux variable returns false when isDefinedOnTemporalInterface() is called.
  bool hasSpaceOnlyTrialVariable() const;
  
  // accessors:
  VarPtr test(int testID);
  VarPtr trial(int trialID);

  std::vector<int> testIDs();

  std::vector<int> trialIDs();

  VarPtr testVar(std::string name, Space fs, int ID = -1);
  VarPtr fieldVar(std::string name, Space fs = L2, int ID = -1);

  // fluxes are trace variables which trace a term involving a normal (and which therefore
  // need to be multiplied by a sgn(n) term; i.e. the two cells which see a side will see opposite
  // values of the flux variable).
  VarPtr fluxVar(std::string name, LinearTermPtr termTraced, Space fs = L2, int ID = -1);
  VarPtr fluxVar(std::string name, VarPtr termTraced, Space fs = L2, int ID = -1);
  VarPtr fluxVar(std::string name, Space fs = L2, int ID = -1);

  // Methods for creating space-time fluxes that are not defined on purely temporal interfaces:
  VarPtr fluxVarSpaceOnly(std::string name, LinearTermPtr termTraced, Space fs = L2, int ID = -1);
  VarPtr fluxVarSpaceOnly(std::string name, VarPtr termTraced, Space fs = L2, int ID = -1);
  VarPtr fluxVarSpaceOnly(std::string name, Space fs = L2, int ID = -1);

  VarPtr traceVar(std::string name, LinearTermPtr termTraced, Space fs = HGRAD, int ID = -1);
  VarPtr traceVar(std::string name, VarPtr termTraced, Space fs = HGRAD, int ID = -1);
  VarPtr traceVar(std::string name, Space fs = HGRAD, int ID = -1);

  // Methods for creating space-time traces that are not defined on purely temporal interfaces:
  VarPtr traceVarSpaceOnly(std::string name, LinearTermPtr termTraced, Space fs = HGRAD_SPACE_L2_TIME, int ID = -1);
  VarPtr traceVarSpaceOnly(std::string name, VarPtr termTraced, Space fs = HGRAD_SPACE_L2_TIME, int ID = -1);
  VarPtr traceVarSpaceOnly(std::string name, Space fs = HGRAD_SPACE_L2_TIME, int ID = -1);

  const std::map< int, VarPtr > & testVars() const;
  const std::map< int, VarPtr > & trialVars() const;

  std::vector< VarPtr > fieldVars() const;

  std::vector< VarPtr > fluxVars() const;

  std::vector< VarPtr > traceVars() const;

  // returns a new VarFactory with the same test space, and a subspace of the trial space
  VarFactoryPtr trialSubFactory(std::vector< VarPtr > &trialVars) const;

  static VarFactoryPtr varFactory()
  {
    return Teuchos::rcp(new VarFactory);
  }
};
}

#endif
