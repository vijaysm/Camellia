// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  IP.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//

#ifndef Camellia_IP_SWITCH
#define Camellia_IP_SWITCH

#include "TypeDefs.h"

#include "IP.h"

using namespace std;

namespace Camellia
{
class IPSwitcher : public IP
{
  IPPtr _ip1;
  IPPtr _ip2;
  double _minH; // min element  size for when to
public:
  IPSwitcher(IPPtr ip1, IPPtr ip2, double minH);

  void computeInnerProductMatrix(Intrepid::FieldContainer<double> &innerProduct,
                                 Teuchos::RCP<DofOrdering> dofOrdering,
                                 Teuchos::RCP<BasisCache> basisCache);

  void computeInnerProductVector(Intrepid::FieldContainer<double> &ipVector,
                                 VarPtr var, TFunctionPtr<double> fxn,
                                 Teuchos::RCP<DofOrdering> dofOrdering,
                                 Teuchos::RCP<BasisCache> basisCache);

  double computeMaxConditionNumber(DofOrderingPtr testSpace, BasisCachePtr basisCache);

  // added by Jesse
  LinearTermPtr evaluate(map< int, TFunctionPtr<double>> &varFunctions, bool boundaryPart);

  bool hasBoundaryTerms();

  void printInteractions();
};
}

#endif
