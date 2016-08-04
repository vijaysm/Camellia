// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  MeshTransformationFunction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include "TypeDefs.h"

#include "Function.h"
#include "Mesh.h"

#ifndef Camellia_debug_MeshTransformationFunction_h
#define Camellia_debug_MeshTransformationFunction_h

#include "TypeDefs.h"

namespace Camellia
{
class MeshTransformationFunction : public TFunction<double>
{
  map< GlobalIndexType, TFunctionPtr<double> > _cellTransforms; // cellID --> cell transformation function
  Camellia::EOperator _op;
  MeshPtr _mesh;
  int _maxPolynomialDegree;
protected:
  MeshTransformationFunction(MeshPtr mesh, map< GlobalIndexType, TFunctionPtr<double> > cellTransforms, Camellia::EOperator op);
public:
  MeshTransformationFunction(MeshPtr mesh, set<GlobalIndexType> cellIDsToTransform); // might be responsible for only a subset of the curved cells.

  int maxDegree();

  void updateCells(const set<GlobalIndexType> &cellIDs);

  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);

  bool mapRefCellPointsUsingExactGeometry(Intrepid::FieldContainer<double> &cellPoints, const Intrepid::FieldContainer<double> &refCellPoints, GlobalIndexType cellID);

  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();

  void didHRefine(const set<GlobalIndexType> &parentCellIDs);
  void didPRefine(const set<GlobalIndexType> &cellIDs);

  ~MeshTransformationFunction();
};
}


#endif
