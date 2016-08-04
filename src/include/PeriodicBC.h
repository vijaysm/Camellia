// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PeriodicBC.h
//  Camellia-debug
//
//  Created by Nate Roberts on 6/12/14.
//
//

#ifndef __Camellia_debug__PeriodicBC__
#define __Camellia_debug__PeriodicBC__

#include "TypeDefs.h"

#include "Teuchos_RCP.hpp"

#include <vector>

namespace Camellia
{
class PeriodicBC
{
  SpatialFilterPtr _pointFilter0, _pointFilter1;
  TFunctionPtr<double> _transform0to1, _transform1to0;
public:
  PeriodicBC(SpatialFilterPtr pointFilter0, SpatialFilterPtr pointFilter1, TFunctionPtr<double> transform0to1, TFunctionPtr<double> transform1to0);

  std::vector<double> getMatchingPoint(const std::vector<double> &point, int whichSide);
  std::vector<int> getMatchingSides(const std::vector<double> &point); // includes 0 if the point matches pointFilter0, 1 if it matches pointFilter1, and empty if it matches neither.

  static Teuchos::RCP<PeriodicBC> periodicBC(SpatialFilterPtr pointFilter1, SpatialFilterPtr pointFilter2, TFunctionPtr<double> transform1to2, TFunctionPtr<double> transform2to1);
  static Teuchos::RCP<PeriodicBC> xIdentification(double x1, double x2);
  static Teuchos::RCP<PeriodicBC> yIdentification(double y1, double y2);
  static Teuchos::RCP<PeriodicBC> zIdentification(double z1, double z2);
};

typedef Teuchos::RCP<PeriodicBC> PeriodicBCPtr;
}

#endif /* defined(__Camellia_debug__PeriodicBC__) */
