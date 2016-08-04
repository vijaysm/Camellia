// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

//
//  LegendreHVOL_LineBasis.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_LegendreHVOL_LineBasis_h
#define Camellia_debug_LegendreHVOL_LineBasis_h

#include "Basis.h"
#include "Intrepid_FieldContainer.hpp"

namespace Camellia
{
template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class LegendreHVOL_LineBasis;
template<class Scalar, class ArrayScalar> class LegendreHVOL_LineBasis : public Basis<Scalar,ArrayScalar>
{
protected:
  void initializeTags() const;
  int _degree;

  Intrepid::FieldContainer<double> _legendreL2norms;
  void initializeL2normValues();
public:
  LegendreHVOL_LineBasis(int degree); // conforming means not strictly hierarchical, but has e.g. vertex dofs defined...

  void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const;
};
}

#include "LegendreHVOL_LineBasisDef.h"

#endif
