// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

//  SchwarzSolver.h
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//

#ifndef Camellia_SchwarzSolver_h
#define Camellia_SchwarzSolver_h

#include "Solver.h"

namespace Camellia
{
class SchwarzSolver : public Solver
{
  int _overlapLevel;
  int _maxIters;
  bool _printToConsole;
  double _tol;
public:
  SchwarzSolver(int overlapLevel, int maxIters, double tol);
  void setPrintToConsole(bool printToConsole);
  int solve();
  void setTolerance(double tol);
};
}

#endif
