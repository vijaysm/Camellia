// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  AnisotropicRefinementStrategy.h
//  Camellia
//
// ***********************************************************************
//
//                  Camellia AnisotropicRefinementStrategy:
//
// This code was originally part of the RefinementStrategy class, and
// is not really supported at the moment.  The hope is to modernize it
// (with distributed computation support, e.g.) at some point.
//
// ***********************************************************************
//
//  Created by Nate Roberts on 8/9/16.
//
//

#ifndef __Camellia__AnisotropicRefinementStrategy__
#define __Camellia__AnisotropicRefinementStrategy__

#include "RefinementStrategy.h"

namespace Camellia {
  template <typename Scalar>
  class AnisotropicRefinementStrategy : TRefinementStrategy<Scalar>
  {
    double _anisotropicThreshhold;
    double _maxAspectRatio;
    SolutionPtr _solution;
    RieszRepPtr _rieszRep;
  public:
    AnisotropicRefinementStrategy( TSolutionPtr<Scalar> solution, double relativeEnergyThreshold, double min_h = 0, int max_p = 10, bool preferPRefinements = false);
    AnisotropicRefinementStrategy( MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip, double relativeEnergyThreshold, double min_h = 0, int max_p = 10, bool preferPRefinements = false);
    
    virtual void refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr);
    void refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, map<GlobalIndexType,double> &threshMap);
    void refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, map<GlobalIndexType,double> &threshMap, map<GlobalIndexType, bool> useHRefMap);
    
    void getAnisotropicCellsToRefine(map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells, vector<GlobalIndexType> &regCells);
    void getAnisotropicCellsToRefine(map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells, vector<GlobalIndexType> &regCells,
                                     map<GlobalIndexType,double> &threshMap);
    bool enforceAnisotropicOneIrregularity(vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells);
    
    void setAnisotropicThreshhold(double value);
    void setMaxAspectRatio(double value);
    
  };
}

#endif /* defined(__Camellia__AnisotropicRefinementStrategy__) */
