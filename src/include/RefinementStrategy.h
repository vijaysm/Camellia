// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  RefinementStrategy.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//

#ifndef Camellia_RefinementStrategy_h
#define Camellia_RefinementStrategy_h

#include "Teuchos_RCP.hpp"

#include "ErrorIndicator.h"
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
public:
  enum RefinementType
  {
    H_REFINEMENT, // distinction among h-refinement types comes in the RefinementPattern
    P_REFINEMENT
  };
protected:

  static RefinementResults setResults(GlobalIndexType numElements, GlobalIndexType numDofs, double totalEnergyError);
  ErrorIndicatorPtr _errorIndicator;

  double _relativeErrorThreshold;
  bool _enforceOneIrregularity;
  bool _reportPerCellErrors;
  vector< RefinementResults > _results;
  double _min_h;
  int _max_p;
  bool _preferPRefinements;
  
  MeshPtr mesh();
public:
  TRefinementStrategy( ErrorIndicatorPtr errorIndicator, double relativeErrorThreshold, double min_h = 0, int max_p = 10, bool preferPRefinements = false);
  TRefinementStrategy( TSolutionPtr<Scalar> solution, double relativeEnergyThreshold, double min_h = 0, int max_p = 10,
                      bool preferPRefinements = false);
  TRefinementStrategy( MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip, double relativeEnergyThreshold,
                      double min_h = 0, int max_p = 10, bool preferPRefinements = false, int cubatureEnrichment = 0);
  
  double computeTotalEnergyError();
  
  // ! Set the energy threshold (relative to maximum element error) to use for refinements
  void setRelativeErrorThreshold(double value);
  void setEnforceOneIrregularity(bool value);

  virtual void refine(bool printToConsole=false);
  virtual void hRefineUniformly();

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
  
  static RefinementStrategyPtr energyErrorRefinementStrategy(SolutionPtr soln, double relativeEnergyThreshold);
  static RefinementStrategyPtr energyErrorRefinementStrategy(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
                                                             double relativeEnergyThreshold, int cubatureEnrichmentDegree = 0);
  static RefinementStrategyPtr gradientRefinementStrategy(SolutionPtr soln, VarPtr scalarVar, double relativeEnergyThreshold);
  static RefinementStrategyPtr hessianRefinementStrategy(SolutionPtr soln, VarPtr scalarVar, double relativeEnergyThreshold);
  
  virtual ~TRefinementStrategy() {}
};

extern template class TRefinementStrategy<double>;
}


#endif
