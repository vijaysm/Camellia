//
//  SpatialFilter.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/28/13.
//
//

#include "SpatialFilter.h"

bool SpatialFilter::matchesPoint(double x) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x) unimplemented.");
  return false;
}

bool SpatialFilter::matchesPoint(double x, double y) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y) unimplemented.");
  return false;
}

bool SpatialFilter::matchesPoint(double x, double y, double z) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y,z) unimplemented.");
}

bool SpatialFilter::matchesPoint(vector<double>&point) {
  if (point.size() == 3) {
    return matchesPoint(point[0],point[1],point[2]);
  } else if (point.size() == 2) {
    return matchesPoint(point[0],point[1]);
  } else if (point.size() == 1) {
    return matchesPoint(point[0]);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point is of unsupported dimension.");
    return false;
  }
}

bool SpatialFilter::matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) {
  const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
  //    cout << "points:\n" << *points;
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  int spaceDim = points->dimension(2);
  TEUCHOS_TEST_FOR_EXCEPTION(numCells != pointsMatch.dimension(0), std::invalid_argument, "numCells do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION(numPoints != pointsMatch.dimension(1), std::invalid_argument, "numPoints do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim > 3, std::invalid_argument, "matchesPoints supports 1D, 2D, and 3D only.");
  pointsMatch.initialize(false);
  bool somePointMatches = false;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      vector<double> point;
      for (int d=0; d<spaceDim; d++) {
        point.push_back((*points)(cellIndex,ptIndex,d));
      }
      if (matchesPoint(point)) {
        somePointMatches = true;
        pointsMatch(cellIndex,ptIndex) = true;
      }
    }
  }
  return somePointMatches;
}

SpatialFilterPtr SpatialFilter::allSpace() {
  return Teuchos::rcp( new SpatialFilterUnfiltered );
}

SpatialFilterPtr SpatialFilter::unionFilter(SpatialFilterPtr a, SpatialFilterPtr b) {
  return Teuchos::rcp( new SpatialFilterLogicalOr(a,b) );
}

SpatialFilterPtr SpatialFilter::negatedFilter(SpatialFilterPtr filterToNegate) {
  return Teuchos::rcp( new NegatedSpatialFilter(filterToNegate) );
}



bool SpatialFilterUnfiltered::matchesPoint(vector<double> &point) {
  return true;
}


SpatialFilterLogicalOr::SpatialFilterLogicalOr(SpatialFilterPtr sf1, SpatialFilterPtr sf2) {
  _sf1 = sf1;
  _sf2 = sf2;
}
bool SpatialFilterLogicalOr::matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) {
  const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  FieldContainer<bool> pointsMatch2(pointsMatch);
  bool somePointMatches1 = _sf1->matchesPoints(pointsMatch,basisCache);
  bool somePointMatches2 = _sf2->matchesPoints(pointsMatch2,basisCache);
  if ( !somePointMatches2 ) {
    // then what's in pointsMatch is exactly right
    return somePointMatches1;
  } else if ( !somePointMatches1 ) {
    // then what's in pointsMatch2 is exactly right
    pointsMatch = pointsMatch2;
    return somePointMatches2;
  } else {
    // need to combine them:
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        pointsMatch(cellIndex,ptIndex) |= pointsMatch2(cellIndex,ptIndex);
      }
    }
    // if we're here, then some point matched: return true:
    return true;
  }
}

NegatedSpatialFilter::NegatedSpatialFilter(SpatialFilterPtr filterToNegate) {
  _filterToNegate = filterToNegate;
}
bool NegatedSpatialFilter::matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) {
  const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  _filterToNegate->matchesPoints(pointsMatch,basisCache);
  bool somePointMatches = false;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      pointsMatch(cellIndex,ptIndex) = ! pointsMatch(cellIndex,ptIndex);
      somePointMatches |= pointsMatch(cellIndex,ptIndex);
    }
  }
  return somePointMatches;
}
