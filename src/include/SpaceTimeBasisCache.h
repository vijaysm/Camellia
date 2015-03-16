//
//  SpaceTimeBasisCache.h
//  Camellia
//
//  Created by Nate Roberts on 3/11/15.
//
//

#ifndef __Camellia__SpaceTimeBasisCache__
#define __Camellia__SpaceTimeBasisCache__

#include "BasisCache.h"
#include "TensorBasis.h"

class SpaceTimeBasisCache : public BasisCache {
  typedef Teuchos::RCP< Intrepid::FieldContainer<double> > FCPtr;
  typedef Teuchos::RCP< const Intrepid::FieldContainer<double> > constFCPtr;
  
  BasisCachePtr _spatialCache, _temporalCache;

  // side constructor:
  SpaceTimeBasisCache(int sideIndex, Teuchos::RCP<SpaceTimeBasisCache> volumeCache, int trialDegree, int testDegree);

  Camellia::EOperator spaceOp(Camellia::EOperator op);
  Camellia::EOperator timeOp(Camellia::EOperator op);
  
  Intrepid::EOperator spaceOpForSizing(Camellia::EOperator op);
  Intrepid::EOperator timeOpForSizing(Camellia::EOperator op);
  
  constFCPtr getTensorBasisValues(TensorBasis<double>* tensorBasis,
                                  int fieldIndex, int pointIndex,
                                  constFCPtr spatialValues,
                                  constFCPtr temporalValues,
                                  Intrepid::EOperator spaceOp,
                                  Intrepid::EOperator timeOp) const;
protected:
  virtual void createSideCaches();
public:
  // volume constructors:
  SpaceTimeBasisCache(MeshPtr spaceTimeMesh, ElementTypePtr spaceTimeElementType,
                      const FieldContainer<double> &physicalNodesSpatial,
                      const FieldContainer<double> &physicalNodesTemporal,
                      const FieldContainer<double> &physicalNodesSpaceTime,
                      const std::vector<GlobalIndexType> &cellIDs,
                      bool testVsTest, int cubatureDegreeEnrichment);
  SpaceTimeBasisCache(const FieldContainer<double> &physicalNodesSpatial,
                      const FieldContainer<double> &physicalNodesTemporal,
                      const FieldContainer<double> &physicalCellNodes,
                      CellTopoPtr cellTopo, int cubDegree);
  
  BasisCachePtr getSpatialBasisCache();
  BasisCachePtr getTemporalBasisCache();
  
  virtual constFCPtr getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual constFCPtr getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual constFCPtr getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
};


#endif /* defined(__Camellia__SpaceTimeBasisCache__) */
