//
//  AdaptiveSolveStrategy.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_RefinementStrategy_h
#define Camellia_RefinementStrategy_h



#include "Teuchos_RCP.hpp"

#include "IP.h"
#include "LinearTerm.h"
#include "RieszRep.h"
#include "TypeDefs.h"

using namespace std;

namespace Camellia
{
struct RefinementResults
{
  int numElements;
  int numDofs;
  double totalEnergyError;
};

template <typename Scalar>
class TRefinementStrategy
{
protected:

  static RefinementResults setResults(GlobalIndexType numElements, GlobalIndexType numDofs, double totalEnergyError);
  TSolutionPtr<Scalar> _solution;

  TRieszRepPtr<Scalar> _rieszRep;

  double _relativeEnergyThreshold;
  bool _enforceOneIrregularity;
  bool _reportPerCellErrors;
  double _anisotropicThreshhold;
  double _maxAspectRatio;
  vector< RefinementResults > _results;
  double _min_h;
  int _max_p;
  bool _preferPRefinements;

  MeshPtr mesh();
public:
  TRefinementStrategy( TSolutionPtr<Scalar> solution, double relativeEnergyThreshold, double min_h = 0, int max_p = 10, bool preferPRefinements = false);
  TRefinementStrategy( MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip, double relativeEnergyThreshold, double min_h = 0, int max_p = 10, bool preferPRefinements = false);
  
  double computeTotalEnergyError();
  
  void setEnforceOneIrregularity(bool value);
  void setAnisotropicThreshhold(double value);
  void setMaxAspectRatio(double value);

  virtual void refine(bool printToConsole=false);
  virtual void refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr);
  void refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, map<GlobalIndexType,double> &threshMap);
  void refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, map<GlobalIndexType,double> &threshMap, map<GlobalIndexType, bool> useHRefMap);

  void getAnisotropicCellsToRefine(map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells, vector<GlobalIndexType> &regCells);
  void getAnisotropicCellsToRefine(map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells, vector<GlobalIndexType> &regCells,
                                   map<GlobalIndexType,double> &threshMap);
  bool enforceAnisotropicOneIrregularity(vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells);

  virtual void refineCells(vector<GlobalIndexType> &cellIDs);
  static void pRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs);
  static void hRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs);
  static void hRefineUniformly(MeshPtr mesh);
  void getCellsAboveErrorThreshhold(vector<GlobalIndexType> &cellsToRefine);
  void setMinH(double value);
  void setReportPerCellErrors(bool value);

  double getEnergyError(int refinementNumber);
  GlobalIndexType getNumElements(int refinementNumber);
  GlobalIndexType getNumDofs(int refinementNumber);
  
  TRieszRepPtr<Scalar> getRieszRep();
  
  // ! computes approximate gradients in the scalar field variable indicated.  It is not clear that this is the
  // ! best place for this, but it's placed here for now because this can be useful in context of refinements.
  static void computeApproximateGradients(SolutionPtr soln, VarPtr u, const std::vector<GlobalIndexType> &cells,
                                          std::vector<double> &gradient_l2_norm, double weightWithPowerOfH);
  
  virtual ~TRefinementStrategy() {}
};

extern template class TRefinementStrategy<double>;
}


#endif
