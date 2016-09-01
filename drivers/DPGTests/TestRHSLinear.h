//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef TEST_RHS_LINEAR
#define TEST_RHS_LINEAR

/*
 *  TestRHSLinear.h
 *
 */

#include "RHS.h"

#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;

class TestRHSLinear : public RHS
{
public:
  TestRHSLinear() : RHS(true) {} // true: legacy subclass of RHS
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
        double x = physicalPoints(cellIndex,ptIndex,0);
        double y = physicalPoints(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex) = 6.0*x+12.0*y;
      }
    }
  }

  static void expectedRHSForCubicOnQuad(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests (on reference quad)
    rhsVector.resize(1,16);
    rhsVector(0,0) = -0.49999999999999967;
    rhsVector(0,1) = -2.0393446629166325;
    rhsVector(0,2) = -1.2939886704167036;
    rhsVector(0,3) = -0.16666666666666619;
    rhsVector(0,4) = -1.5786893258332635;
    rhsVector(0,5) = -5.590169943749476;
    rhsVector(0,6) = -1.8633899812498225;
    rhsVector(0,7) = 0.08797734083340591;
    rhsVector(0,8) = -0.0879773408334048;
    rhsVector(0,9) = 1.863389981249827;
    rhsVector(0,10) = 5.590169943749478;
    rhsVector(0,11) = 1.5786893258332626;
    rhsVector(0,12) = 0.16666666666666657;
    rhsVector(0,13) = 1.2939886704167032;
    rhsVector(0,14) = 2.0393446629166325;
    rhsVector(0,15) = 0.49999999999999917;
  }

  static void expectedRHSForCubicOnTri(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests (on lower half of ref quad)
    rhsVector.resize(1,10);
    rhsVector(0,0) = -5.704166666666664;
    rhsVector(0,1) = 7.702737698905779;
    rhsVector(0,2) = -2.452737698905783;
    rhsVector(0,3) = -0.8916666666666593;
    rhsVector(0,4) = 8.865034433436877;
    rhsVector(0,5) = -17.32499999999999;
    rhsVector(0,6) = 8.27172025984384;
    rhsVector(0,7) = -4.365034433436867;
    rhsVector(0,8) = 10.228279740156152;
    rhsVector(0,9) = -0.32916666666667055;
  }

  static void expectedRHSForCubicOnUnitTri(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests (on lower half of (0,1)^2, the ref tri reflected)
    rhsVector.resize(1,10);
    rhsVector(0,0) = 0.06666666666666643;
    rhsVector(0,1) = 0.2181694990624914;
    rhsVector(0,2) = 0.03183050093749884;
    rhsVector(0,3) = 0.0666666666666682;
    rhsVector(0,4) = 0.3136610018750199;
    rhsVector(0,5) = 1.8000000000000114;
    rhsVector(0,6) = 0.09549150281252139;
    rhsVector(0,7) = 0.6863389981249797;
    rhsVector(0,8) = 0.6545084971874751;
    rhsVector(0,9) = 0.0666666666666651;
  }

  static void expectedRHSForCubicOnUnitQuad(FieldContainer<double> &rhsVector)
  {
    // values from Mathematica notebook RHSIntegrationTests (on (0,1)^2, NOT ref quad)
    rhsVector.resize(1,16);
    rhsVector(0,0) = 0.;
    rhsVector(0,1) = 0.05758191713541816;
    rhsVector(0,2) = 0.15075141619790955;
    rhsVector(0,3) = 0.04166666666666663;
    rhsVector(0,4) = 0.1151638342708412;
    rhsVector(0,5) = 0.8637287570313141;
    rhsVector(0,6) = 1.3295762523437755;
    rhsVector(0,7) = 0.32349716760417513;
    rhsVector(0,8) = 0.30150283239582354;
    rhsVector(0,9) = 1.7954237476562191;
    rhsVector(0,10) = 2.261271242968684;
    rhsVector(0,11) = 0.5098361657291579;
    rhsVector(0,12) = 0.08333333333333304;
    rhsVector(0,13) = 0.4742485838020878;
    rhsVector(0,14) = 0.5674180828645792;
    rhsVector(0,15) = 0.125;
  }
};

#endif