//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef TEST_RHS_ONE
#define TEST_RHS_ONE

/*
 *  TestRHSOne.h
 *
 */

#include "RHS.h"

#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;

class TestRHSOne : public RHS
{
public:
  TestRHSOne() : RHS(true) {} // true: legacy subclass of RHS
  bool nonZeroRHS(int testVarID)
  {
    return true;
  }

  void rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values)
  {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    values.resize(numCells,numPoints);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        values(cellIndex,ptIndex) = 1.0;
      }
    }
  }

  static void expectedRHSForLinearOnUnitTri(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests
    rhsVector.resize(1,3);
    rhsVector(0,0) = 0.16666666666666666;
    rhsVector(0,1) = 0.16666666666666666;
    rhsVector(0,2) = 0.16666666666666666;
  }

  static void expectedRHSForCubicOnTri(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests
    rhsVector.resize(1,10);
    rhsVector(0,0) = -0.68125;
    rhsVector(0,1) = 0.9904379096353516;
    rhsVector(0,2) = -0.4071045763020193;
    rhsVector(0,3) = -0.0770833333333327;
    rhsVector(0,4) = 0.9904379096353504;
    rhsVector(0,5) = -2.08125;
    rhsVector(0,6) = 1.1249999999999993;
    rhsVector(0,7) = -0.40710457630201635;
    rhsVector(0,8) = 1.1249999999999996;
    rhsVector(0,9) = -0.07708333333333385;
  }

  static void expectedRHSForCubicOnQuad(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests
    rhsVector.resize(1,16);
    rhsVector(0,0) = 0.027777777777777714;
    rhsVector(0,1) = 0.1388888888888888;
    rhsVector(0,2) = 0.1388888888888888;
    rhsVector(0,3) = 0.0277777777777777;
    rhsVector(0,4) = 0.13888888888888878;
    rhsVector(0,5) = 0.6944444444444449;
    rhsVector(0,6) = 0.6944444444444449;
    rhsVector(0,7) = 0.13888888888888867;
    rhsVector(0,8) = 0.13888888888888873;
    rhsVector(0,9) = 0.6944444444444449;
    rhsVector(0,10) = 0.6944444444444446;
    rhsVector(0,11) = 0.13888888888888865;
    rhsVector(0,12) = 0.027777777777777693;
    rhsVector(0,13) = 0.13888888888888865;
    rhsVector(0,14) = 0.13888888888888865;
    rhsVector(0,15) = 0.02777777777777768;
  }

  static void expectedRHSForCubicOnUnitTri(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests (on lower half of (0,1)^2, the ref tri)
    rhsVector.resize(1,10);
    rhsVector(0,0) = 0.008333333333334025;
    rhsVector(0,1) = 0.04166666666666485;
    rhsVector(0,2) = 0.04166666666666741;
    rhsVector(0,3) = 0.00833333333333286;
    rhsVector(0,4) = 0.04166666666666671;
    rhsVector(0,5) = 0.22500000000000042;
    rhsVector(0,6) = 0.041666666666667185;
    rhsVector(0,7) = 0.041666666666586194;
    rhsVector(0,8) = 0.04166666666666674;
    rhsVector(0,9) = 0.008333333333333165;
  }

  static void expectedRHSForCubicOnUnitQuad(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests (on (0,1)^2, NOT ref quad)
    rhsVector.resize(1,16);
    rhsVector(0,0) = 0.006944444444444406;
    rhsVector(0,1) = 0.03472222222222243;
    rhsVector(0,2) = 0.03472222222222221;
    rhsVector(0,3) = 0.00694444444444442;
    rhsVector(0,4) = 0.03472222222222207;
    rhsVector(0,5) = 0.17361111111111094;
    rhsVector(0,6) = 0.17361111111111116;
    rhsVector(0,7) = 0.03472222222222213;
    rhsVector(0,8) = 0.03472222222222207;
    rhsVector(0,9) = 0.17361111111111072;
    rhsVector(0,10) = 0.17361111111111116;
    rhsVector(0,11) = 0.03472222222222221;
    rhsVector(0,12) = 0.00694444444444442;
    rhsVector(0,13) = 0.03472222222222232;
    rhsVector(0,14) = 0.03472222222222232;
    rhsVector(0,15) = 0.006944444444444434;
  }
};
#endif