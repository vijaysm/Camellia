// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER
//
//  IntrepidBasisConstruction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/10/13.
//
//

#ifndef Camellia_debug_doubleBasisConstruction_h
#define Camellia_debug_doubleBasisConstruction_h

#include "Basis.h"

namespace Camellia
{
BasisPtr lobattoQuadHGRAD(int polyOrder, bool conforming=false);

BasisPtr intrepidLineHGRAD(int polyOrder);

BasisPtr intrepidQuadHGRAD(int polyOrder);
BasisPtr intrepidQuadHDIV(int polyOrder);

BasisPtr intrepidHexHGRAD(int polyOrder);
BasisPtr intrepidHexHDIV(int polyOrder);

}

#endif
