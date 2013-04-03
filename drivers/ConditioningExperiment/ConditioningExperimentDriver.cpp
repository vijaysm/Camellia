//
//  ConditioningExperimentDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/12/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "VarFactory.h"
#include "IP.h"
#include "Function.h"
#include "BF.h"
#include "MeshFactory.h"
#include "MeshUtilities.h"

#include "Legendre.hpp"
#include "Lobatto.hpp"

enum TestType {
  L2Part,
  FullNorm
};

void setupHCurlTest(TestType testType, VarFactory &varFactory, VarPtr &var, IPPtr &ip) {
  var = varFactory.testVar("\\omega", HCURL);
  ip = Teuchos::rcp( new IP );
  if (testType==L2Part) {
    ip->addTerm(var);
  } else {
    FunctionPtr h = Teuchos::rcp( new hFunction );
    ip->addTerm(var);
    ip->addTerm(h * var->curl());
  }
}

void setupHDivTest(TestType testType, VarFactory &varFactory, VarPtr &var, IPPtr &ip) {
  var = varFactory.testVar("\\tau", HDIV);
  ip = Teuchos::rcp( new IP );
  if (testType==L2Part) {
    ip->addTerm(var);
  } else {
    FunctionPtr h = Teuchos::rcp( new hFunction );
    ip->addTerm(var);
    ip->addTerm(h * var->div());
  }
}

void setupHGradTest(TestType testType, VarFactory &varFactory, VarPtr &var, IPPtr &ip) {
  var = varFactory.testVar("q", HGRAD);
  ip = Teuchos::rcp( new IP );
  if (testType==L2Part) {
    ip->addTerm(var);
  } else {
    FunctionPtr h = Teuchos::rcp( new hFunction );
    ip->addTerm(var);
    ip->addTerm(h * var->grad());
  }
}

int main(int argc, char *argv[]) {
  vector< Space > spaces;
  spaces.push_back(HGRAD);
  spaces.push_back(HDIV);
  spaces.push_back(HCURL);
  vector< TestType > testTypes;
  testTypes.push_back(L2Part);
  testTypes.push_back(FullNorm);
  for (vector<TestType>::iterator typeIt = testTypes.begin(); typeIt != testTypes.end(); typeIt++) {
    TestType testType = *typeIt;
    string typeName = (testType==L2Part) ? "L2" : "fullNorm";
    cout << "*************** " << typeName << " tests ***************\n";
    for (vector< Space >::iterator spaceIt = spaces.begin(); spaceIt != spaces.end(); spaceIt++) {
      Space space = *spaceIt;
      VarFactory varFactory;
      VarPtr var;
      IPPtr ip;
      string spaceName;
      if (space==HGRAD) {
        spaceName = "grad";
        setupHGradTest(testType, varFactory, var, ip);
      } else if (space==HDIV) {
        spaceName = "div";
        setupHDivTest(testType, varFactory, var, ip);
      } else if (space==HCURL) {
        spaceName = "curl";
        setupHCurlTest(testType, varFactory, var, ip);
      }
      cout << spaceName << ":\n";
      VarPtr u = varFactory.fieldVar("u"); // we don't really care about the trial space
      BFPtr bf = Teuchos::rcp( new BF(varFactory) );
      int pToAdd = 0;
      for (int testOrder=1; testOrder<=10; testOrder++) {
        MeshPtr mesh = MeshFactory::quadMesh(bf, testOrder, pToAdd);
        ostringstream fileNameStream;
        fileNameStream << spaceName << "_" << typeName << "_p" << testOrder << ".dat";
        string fileName = fileNameStream.str();
        double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, mesh, fileName);
        cout << maxConditionNumber << endl;
      }
    }
  }
  
//  int polyOrder = 20;
//  FieldContainer<double> values(polyOrder+1), dvalues(polyOrder+1);
//  double x = 0.5;
//  Legendre<>::values(values,dvalues,x,polyOrder);
//  cout << "Legendre values at x=0.5:\n";
//  for (int i=0; i<values.size(); i++) {
//    cout << i << ": " << values[i] << endl;
//  }
//  cout << "Legendre derivatives at x=0.5:\n";
//  for (int i=0; i<dvalues.size(); i++) {
//    cout << i << ": " << dvalues[i] << endl;
//  }
//  
//  Lobatto<>::values(values,dvalues,x,polyOrder);
//  cout << "Lobatto values at x=0.5:\n";
//  for (int i=0; i<values.size(); i++) {
//    cout << i << ": " << values[i] << endl;
//  }
//  cout << "Lobatto derivatives at x=0.5:\n";
//  for (int i=0; i<dvalues.size(); i++) {
//    cout << i << ": " << dvalues[i] << endl;
//  }
  return 0;
}