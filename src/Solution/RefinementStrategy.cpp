//
//  RefinementStrategy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RefinementStrategy.h"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;
using namespace Intrepid;

template <typename Scalar>
TRefinementStrategy<Scalar>::TRefinementStrategy( TSolutionPtr<Scalar> solution, double relativeEnergyThreshold, double min_h,
    int max_p, bool preferPRefinements)
{
  _solution = solution;
  _relativeEnergyThreshold = relativeEnergyThreshold;
  _enforceOneIrregularity = true;
  _reportPerCellErrors = false;
  _anisotropicThreshhold = 10.0;
  _maxAspectRatio = 2^5; // five anisotropic refinements of an element
  _min_h = min_h;
  _preferPRefinements = preferPRefinements;
  _max_p = max_p;
}

template <typename Scalar>
TRefinementStrategy<Scalar>::TRefinementStrategy( MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
    double relativeEnergyThreshold, double min_h,
    int max_p, bool preferPRefinements)
{
  _rieszRep = Teuchos::rcp( new TRieszRep<Scalar>(mesh, ip, residual) );
  _relativeEnergyThreshold = relativeEnergyThreshold;
  _enforceOneIrregularity = true;
  _reportPerCellErrors = false;
  _anisotropicThreshhold = 10.0;
  _maxAspectRatio = 2^5; // five anisotropic refinements of an element
  _min_h = min_h;
  _preferPRefinements = preferPRefinements;
  _max_p = max_p;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::computeApproximateGradients(SolutionPtr soln, VarPtr u, const vector<GlobalIndexType> &cells,
                                                              vector<double> &gradient_l2_norm, double weightWithPowerOfH)
{
  
  // imitates https://dealii.org/developer/doxygen/deal.II/namespaceDerivativeApproximation.html
  
  // we require that u be a scalar field variable
  TEUCHOS_TEST_FOR_EXCEPTION(u->rank() != 0, std::invalid_argument, "u must be a scalar variable");
  TEUCHOS_TEST_FOR_EXCEPTION(u->varType() != FIELD, std::invalid_argument, "u must be a field variable");
  
  int cellCount = cells.size();
  gradient_l2_norm.resize(cells.size());
  
  int onePoint = 1;
  MeshTopologyViewPtr meshTopo = soln->mesh()->getTopology();
  int spaceDim = meshTopo->getDimension();
  
  set<GlobalIndexType> cellsAndNeighborsSet;
  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    cellsAndNeighborsSet.insert(cells[cellOrdinal]);
    CellPtr cell = meshTopo->getCell(cells[cellOrdinal]);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    cellsAndNeighborsSet.insert(neighborIDs.begin(),neighborIDs.end());
  }
  vector<GlobalIndexType> cellsAndNeighbors(cellsAndNeighborsSet.begin(),cellsAndNeighborsSet.end());
  
  // get any off-rank solution data we may need:
  soln->importSolutionForOffRankCells(cellsAndNeighborsSet);
  
  int cellsAndNeighborsCount = cellsAndNeighbors.size();
  
  FieldContainer<double> cellValues(cellsAndNeighborsCount,onePoint); // values at cell centers
  FieldContainer<double> cellCenters(cellsAndNeighborsCount,spaceDim);
  FieldContainer<double> cellDiameter(cellsAndNeighborsCount,onePoint); // h-values
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForTopology;
  
  FunctionPtr hFunction = Function::h();
  FunctionPtr solnFunction = Function::solution(u, soln);
  Teuchos::Array<int> cellValueDim;
  cellValueDim.push_back(1);
  cellValueDim.push_back(1);
  
  map<GlobalIndexType,int> cellIDToOrdinal; // lookup table for value access
  
  // setup: compute cell centers, and solution values at those points
  for (int cellOrdinal=0; cellOrdinal<cellsAndNeighborsCount; cellOrdinal++)
  {
    GlobalIndexType cellID = cellsAndNeighbors[cellOrdinal];
    cellIDToOrdinal[cellID] = cellOrdinal;
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    if (basisCacheForTopology.find(cellTopo->getKey()) == basisCacheForTopology.end())
    {
      FieldContainer<double> centroid(onePoint,spaceDim);
      int nodeCount = cellTopo->getNodeCount();
      FieldContainer<double> cellNodes(nodeCount,spaceDim);
      CamelliaCellTools::refCellNodesForTopology(cellNodes, cellTopo);
      for (int node=0; node<nodeCount; node++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          centroid(0,d) += cellNodes(node,d);
        }
      }
      for (int d=0; d<spaceDim; d++)
      {
        centroid(0,d) /= nodeCount;
      }
      basisCacheForTopology[cellTopo->getKey()] = BasisCache::basisCacheForReferenceCell(cellTopo, 0); // 0 cubature degree
      basisCacheForTopology[cellTopo->getKey()]->setRefCellPoints(centroid);
      basisCacheForTopology[cellTopo->getKey()]->setMesh(soln->mesh());
    }
    BasisCachePtr basisCache = basisCacheForTopology[cellTopo->getKey()];
    basisCache->setPhysicalCellNodes(soln->mesh()->physicalCellNodesForCell(cellID), {cellID}, false);
    
    FieldContainer<double> cellValue(cellValueDim,&cellValues(cellOrdinal,0));
    solnFunction->values(cellValue, basisCache);
    for (int d=0; d<spaceDim; d++)
    {
      cellCenters(cellOrdinal,d) = basisCache->getPhysicalCubaturePoints()(0,0,d);
    }
    if (weightWithPowerOfH != 0)
    {
      cellDiameter(cellOrdinal,0) = soln->mesh()->getCellMeasure(cellID);
    }
  }
  
  // now compute the gradients requested
  FieldContainer<double> Y(spaceDim,spaceDim); // the matrix we'll invert to compute the gradient
  FieldContainer<double> b(spaceDim); // RHS for matrix problem
  FieldContainer<double> grad(spaceDim); // LHS for matrix problem
  vector<double> distanceVector(spaceDim);
  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    Y.initialize(0.0);
    b.initialize(0.0);
    GlobalIndexType cellID = cells[cellOrdinal];
    CellPtr cell = meshTopo->getCell(cellID);
    int myOrdinalInCellAndNeighbors = cellIDToOrdinal[cellID];
    double myValue = cellValues(myOrdinalInCellAndNeighbors,0);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    for (GlobalIndexType neighborID : neighborIDs)
    {
      int neighborOrdinalInCellAndNeighbors = cellIDToOrdinal[neighborID];
      double neighborValue = cellValues(neighborOrdinalInCellAndNeighbors,0);
      
      double dist_squared = 0;
      for (int d=0; d<spaceDim; d++)
      {
        distanceVector[d] = cellCenters(neighborOrdinalInCellAndNeighbors,d) - cellCenters(myOrdinalInCellAndNeighbors,d);
        dist_squared += distanceVector[d] * distanceVector[d];
      }
      
      for (int d1=0; d1<spaceDim; d1++)
      {
        b(d1) += distanceVector[d1] * (neighborValue - myValue) / dist_squared;
        for (int d2=0; d2<spaceDim; d2++)
        {
          Y(d1,d2) += distanceVector[d1] * distanceVector[d2] / dist_squared;
        }
      }
    }
    SerialDenseWrapper::solveSystem(grad, Y, b);
    double l2_value_squared = 0;
    for (int d=0; d<spaceDim; d++)
    {
      l2_value_squared += grad(d) * grad(d);
    }
    if (weightWithPowerOfH == 0)
    {
      gradient_l2_norm[cellOrdinal] = sqrt(l2_value_squared);
    }
    else
    {
      gradient_l2_norm[cellOrdinal] = sqrt(l2_value_squared) * pow(cellDiameter(myOrdinalInCellAndNeighbors,0), weightWithPowerOfH);
    }
  }
}

template <typename Scalar>
double TRefinementStrategy<Scalar>::computeTotalEnergyError()
{
  double totalEnergyErrorSquared = 0.0;
  map<GlobalIndexType, double> energyErrorThisRank; // for each cell owned by this rank
  if (_rieszRep.get() != NULL)
  {
    _rieszRep->computeRieszRep();
    energyErrorThisRank = _rieszRep->getNormsSquared();
    // take square roots:
    for (map<GlobalIndexType, double>::iterator energyEntryIt = energyErrorThisRank.begin();
         energyEntryIt != energyErrorThisRank.end(); energyEntryIt++)
    {
      totalEnergyErrorSquared += energyEntryIt->second;
      energyEntryIt->second = sqrt( energyEntryIt->second );
    }
  }
  else
  {
    energyErrorThisRank = _solution->rankLocalEnergyError();
    for (auto energyEntry : energyErrorThisRank)
    {
      totalEnergyErrorSquared += energyEntry.second * energyEntry.second;
    }
  }
  totalEnergyErrorSquared = MPIWrapper::sum(*mesh()->Comm(), totalEnergyErrorSquared);
  return sqrt(totalEnergyErrorSquared);
}

template <typename Scalar>
TRieszRepPtr<Scalar> TRefinementStrategy<Scalar>::getRieszRep()
{
  return _rieszRep;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setMinH(double value)
{
  _min_h = value;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setAnisotropicThreshhold(double value)
{
  _anisotropicThreshhold = value;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setMaxAspectRatio(double value)
{
  _maxAspectRatio = value;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setEnforceOneIrregularity(bool value)
{
  _enforceOneIrregularity = value;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setReportPerCellErrors(bool value)
{
  _reportPerCellErrors = value;
}

template <typename Scalar>
MeshPtr TRefinementStrategy<Scalar>::mesh()
{
  MeshPtr mesh;
  if (_solution.get())
  {
    mesh = _solution->mesh();
  }
  else
  {
    mesh = _rieszRep->mesh();
  }
  return mesh;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::refine(bool printToConsole)
{
  // greedy refinement algorithm - mark cells for refinement
  MeshPtr mesh = this->mesh();

  double totalEnergyError = 0.0;

  // NOTE 2/16/15: Both approaches, the RieszRep and the Solution, really do store on *each* MPI rank
  //               information about *every* active cell in the mesh.  This should be corrected!!
  map<GlobalIndexType, double> energyError;
  if (_rieszRep.get() != NULL)
  {
    _rieszRep->computeRieszRep();
    energyError = _rieszRep->getNormsSquaredGlobal();
    // take square roots:
    for (map<GlobalIndexType, double>::iterator energyEntryIt = energyError.begin();
         energyEntryIt != energyError.end(); energyEntryIt++)
    {
      totalEnergyError += energyEntryIt->second;
      energyEntryIt->second = sqrt( energyEntryIt->second );
    }
    totalEnergyError = sqrt(totalEnergyError);
  }
  else
  {
    energyError = _solution->globalEnergyError();
    totalEnergyError = _solution->energyErrorTotal();
  }
  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();

  double maxError = 0.0;

  map<GlobalIndexType, double> cellMeasures;
  set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIt=cellIDs.begin(); cellIt != cellIDs.end(); cellIt++)
  {
    int cellID = *cellIt;
    cellMeasures[cellID] = mesh->getCellMeasure(cellID);
    maxError = max(maxError,energyError.find(cellID)->second);
  }

  if ( printToConsole && _reportPerCellErrors )
  {
    cout << "per-cell Energy Error Squared for cells with > 0.1% of squared energy error\n";
    for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
         activeElemIt != activeElements.end(); activeElemIt++)
    {
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      GlobalIndexType cellID = current_element->cellID();
      double cellEnergyError = energyError.find(cellID)->second;
//      cout << "cellID " << cellID << " has energy error (not squared) " << cellEnergyError << endl;
      double percent = (cellEnergyError*cellEnergyError) / (totalEnergyError*totalEnergyError) * 100;
      if (percent > 0.1)
      {
        cout << cellID << ": " << cellEnergyError*cellEnergyError << " ( " << percent << " %)\n";
      }
    }
  }

  // record results prior to refinement
  RefinementResults results = setResults(mesh->numActiveElements(), mesh->numGlobalDofs(), totalEnergyError);
  _results.push_back(results);

  vector<GlobalIndexType> cellsToRefine;
  vector<GlobalIndexType> cellsToPRefine;

  // do refinements on cells with error above threshold
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++)
  {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double h = sqrt(cellMeasures[cellID]);
    double cellEnergyError = energyError.find(cellID)->second;
    int p = mesh->cellPolyOrder(cellID);

    if ( cellEnergyError >= maxError * _relativeEnergyThreshold )
    {
      //      cout << "refining cellID " << cellID << endl;
      if (!_preferPRefinements)
      {
        if (h > _min_h)
        {
          cellsToRefine.push_back(cellID);
        }
        else
        {
          cellsToPRefine.push_back(cellID);
        }
      }
      else
      {
        if (p < _max_p)
        {
          cellsToPRefine.push_back(cellID);
        }
        else
        {
          cellsToRefine.push_back(cellID);
        }
      }
    }
  }

  if (printToConsole)
  {
    if (cellsToRefine.size() > 0) Camellia::print("cells for h-refinement", cellsToRefine);
    if (cellsToPRefine.size() > 0) Camellia::print("cells for p-refinement", cellsToPRefine);
  }
  refineCells(cellsToRefine);
  pRefineCells(mesh, cellsToPRefine);

  bool repartitionAndRebuild = false;
  if (_enforceOneIrregularity)
  {
    mesh->enforceOneIrregularity(repartitionAndRebuild);
//    cout << "Enforced one irregularity.\n";
  }

  // now, repartition and rebuild:
  mesh->repartitionAndRebuild();
  
  if (printToConsole)
  {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::getCellsAboveErrorThreshhold(vector<GlobalIndexType> &cellsToRefine)
{
  // greedy refinement algorithm - mark cells for refinement
  MeshPtr mesh = this->mesh();
  const map<GlobalIndexType, double>* energyError = &(_solution->globalEnergyError());
  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();

  double maxError = 0.0;
  double totalEnergyError = 0.0;

  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++)
  {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double cellEnergyError = energyError->find(cellID)->second;
    maxError = max(cellEnergyError,maxError);
    totalEnergyError += cellEnergyError * cellEnergyError;
  }
  totalEnergyError = sqrt(totalEnergyError);

  // do refinements on cells with error above threshold
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++)
  {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double cellEnergyError = energyError->find(cellID)->second;
    if ( cellEnergyError >= maxError * _relativeEnergyThreshold )
    {
      cellsToRefine.push_back(cellID);
    }
  }
}

// defaults to h-refinement
template <typename Scalar>
void TRefinementStrategy<Scalar>::refineCells(vector<GlobalIndexType> &cellIDs)
{
  MeshPtr mesh = this->mesh();
  hRefineCells(mesh, cellIDs);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::pRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs)
{
  bool repartitionAndRebuild = false;
  int pToAdd = 1;
  set<GlobalIndexType> cellIDSet(cellIDs.begin(),cellIDs.end());
  mesh->pRefine(cellIDSet, pToAdd, repartitionAndRebuild);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::hRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs)
{
  map< Camellia::CellTopologyKey, set<GlobalIndexType> > topologyCellsToRefine;

  MeshTopologyViewPtr meshTopology = mesh->getTopology();

  for (vector< GlobalIndexType >::const_iterator cellIDIt = cellIDs.begin();
       cellIDIt != cellIDs.end(); cellIDIt++)
  {
    int cellID = *cellIDIt;

    CellPtr cell = meshTopology->getCell(cellID);
    Camellia::CellTopologyKey topoKey = cell->topology()->getKey();

    topologyCellsToRefine[topoKey].insert(cellID);
  }

  bool repartitionAndRebuild = false;
  for (map< Camellia::CellTopologyKey, set<GlobalIndexType> >::iterator topoEntry = topologyCellsToRefine.begin();
       topoEntry != topologyCellsToRefine.end(); topoEntry++)
  {
    Camellia::CellTopologyKey topoKey = topoEntry->first;
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(topoKey);
    mesh->hRefine(topoEntry->second, refPattern, repartitionAndRebuild);
  }
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::hRefineUniformly(MeshPtr mesh)
{
  set<GlobalIndexType> cellsToRefine = mesh->getTopology()->getActiveCellIndices();
//  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
//  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
//       activeElemIt != activeElements.end(); activeElemIt++)
//  {
//    Teuchos::RCP< Element > current_element = *(activeElemIt);
//    cellsToRefine.push_back(current_element->cellID());
//  }
  vector<GlobalIndexType> cellsToRefineVector(cellsToRefine.begin(),cellsToRefine.end());
  hRefineCells(mesh, cellsToRefineVector);
  mesh->repartitionAndRebuild();
}

template <typename Scalar>
RefinementResults TRefinementStrategy<Scalar>::setResults(GlobalIndexType numElements, GlobalIndexType numDofs, double totalEnergyError)
{
  RefinementResults solnResults;
  solnResults.numElements = numElements;
  solnResults.numDofs = numDofs;
  solnResults.totalEnergyError = totalEnergyError;
  return solnResults;
}

// without variable anisotropic threshholding
template <typename Scalar>
void TRefinementStrategy<Scalar>::refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr)
{
  // greedy refinement algorithm - mark cells for refinement
  MeshPtr mesh = this->mesh();

  vector<GlobalIndexType> xCells, yCells, regCells;
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells);

  // record results prior to refinement
  double totalEnergyError = _solution->energyErrorTotal();
  RefinementResults results = setResults(mesh->numElements(), mesh->numGlobalDofs(), totalEnergyError);
  _results.push_back(results);

  mesh->hRefine(xCells, RefinementPattern::xAnisotropicRefinementPatternQuad());
  mesh->hRefine(yCells, RefinementPattern::yAnisotropicRefinementPatternQuad());
  mesh->hRefine(regCells, RefinementPattern::regularRefinementPatternQuad());

  if (_enforceOneIrregularity)
    //    mesh->enforceOneIrregularity();
    enforceAnisotropicOneIrregularity(xCells,yCells);

  if (printToConsole)
  {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, map<GlobalIndexType,double> &threshMap)
{
  map<GlobalIndexType,bool> hRefMap;
  vector<ElementPtr> elems = _solution->mesh()->activeElements();
  for (vector<ElementPtr>::iterator elemIt = elems.begin(); elemIt!=elems.end(); elemIt++)
  {
    hRefMap[(*elemIt)->cellID()] = true; // default to h-refinement
  }
  refine(printToConsole,xErr,yErr,threshMap,hRefMap);
}

// with variable anisotropic threshholding and p-refinement specification
template <typename Scalar>
void TRefinementStrategy<Scalar>::refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, map<GlobalIndexType,double> &threshMap, map<GlobalIndexType, bool> useHRefMap)
{

  // greedy refinement algorithm - mark cells for refinement
  MeshPtr mesh = this->mesh();

  vector<GlobalIndexType> xCells, yCells, regCells;
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells, threshMap);

  // record results prior to refinement
  double totalEnergyError = _solution->energyErrorTotal();
  RefinementResults results = setResults(mesh->numElements(), mesh->numGlobalDofs(), totalEnergyError);
  _results.push_back(results);

  // check if any cells should be marked for p-refinement
  vector<GlobalIndexType> pCells;
  for (vector<GlobalIndexType>::iterator cellIt = xCells.begin(); cellIt!=xCells.end(); cellIt++)
  {
    int cellID = *cellIt;
    if (!useHRefMap[cellID])
    {
      pCells.push_back(cellID);
      xCells.erase(cellIt);
    }
  }
  for (vector<GlobalIndexType>::iterator cellIt = yCells.begin(); cellIt!=yCells.end(); cellIt++)
  {
    int cellID = *cellIt;
    if (!useHRefMap[cellID])
    {
      pCells.push_back(cellID);
      yCells.erase(cellIt);
    }
  }
  for (vector<GlobalIndexType>::iterator cellIt = regCells.begin(); cellIt!=regCells.end(); cellIt++)
  {
    int cellID = *cellIt;
    if (!useHRefMap[cellID])
    {
      pCells.push_back(cellID);
      regCells.erase(cellIt);
    }
  }

  mesh->pRefine(pCells); // p-refine FIRST
  mesh->hRefine(xCells, RefinementPattern::xAnisotropicRefinementPatternQuad());
  mesh->hRefine(yCells, RefinementPattern::yAnisotropicRefinementPatternQuad());
  mesh->hRefine(regCells, RefinementPattern::regularRefinementPatternQuad());

  if (_enforceOneIrregularity)
  {
    //    mesh->enforceOneIrregularity();
    enforceAnisotropicOneIrregularity(xCells,yCells);
  }

  if (printToConsole)
  {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::getAnisotropicCellsToRefine(map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells, vector<GlobalIndexType> &regCells)
{
  map<GlobalIndexType,double> threshMap;
  vector<ElementPtr> elems = _solution->mesh()->activeElements();
  for (vector<ElementPtr>::iterator elemIt = elems.begin(); elemIt!=elems.end(); elemIt++)
  {
    threshMap[(*elemIt)->cellID()] = _anisotropicThreshhold;
  }
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells,threshMap);
}

// anisotropy with variable threshholding
template <typename Scalar>
void TRefinementStrategy<Scalar>::getAnisotropicCellsToRefine(map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells, vector<GlobalIndexType> &regCells, map<GlobalIndexType,double> &threshMap)
{
  map<GlobalIndexType,double> energyError = _solution->globalEnergyError();
  MeshPtr mesh = this->mesh();
  vector<GlobalIndexType> cellsToRefine;
  getCellsAboveErrorThreshhold(cellsToRefine);
  for (vector<GlobalIndexType>::iterator cellIt = cellsToRefine.begin(); cellIt!=cellsToRefine.end(); cellIt++)
  {
    int cellID = *cellIt;
    double h1 = mesh->getCellXSize(cellID);
    double h2 = mesh->getCellYSize(cellID);
    double min_h = min(h1,h2);

    double thresh = threshMap[cellID];
    double ratio = xErr[cellID]/yErr[cellID];

    /*
    double anisoErr = xErr[cellID] + yErr[cellID];
    double energyErr = energyError[cellID];
    double anisoPercentage = anisoErr/energyErr;
    cout << "aniso percentage = " << anisoPercentage << endl;
    */
    bool doXAnisotropy = ratio > thresh;
    bool doYAnisotropy = ratio < 1.0/thresh;
    double aspectRatio = max(h1/h2,h2/h1); // WARNING: this assumes a *non-squashed/stretched* element (just skewed)
    double maxAspect = _maxAspectRatio; // the conservative aspect ratio from LD's DPG III: Adaptivity paper is 100.
    // don't refine if h is already too small
    bool doAnisotropy = (aspectRatio < maxAspect);
    if (min_h > _min_h)
    {
      if (doXAnisotropy && doAnisotropy)   // if ratio is small = y err bigger than xErr
      {
        xCells.push_back(cellID); // cut along y-axis
      }
      else if (doYAnisotropy && doAnisotropy)     // if ratio is small = y err bigger than xErr
      {
        yCells.push_back(cellID); // cut along x-axis
      }
      else
      {
        regCells.push_back(cellID);
      }
    }
  }
}

// enforcing one-irregularity with anisotropy - ONLY FOR QUADS RIGHT NOW.  ALSO NOT PARALLELIZED
template <typename Scalar>
bool TRefinementStrategy<Scalar>::enforceAnisotropicOneIrregularity(vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells)
{
  bool success = true;
  MeshPtr mesh = this->mesh();
  int maxIters = mesh->numActiveElements(); // should not refine more than the number of elements...

  // build children list - for use in "upgrading" refinements to prevent deadlocking
  vector<GlobalIndexType> xChildren,yChildren;
  for (vector<GlobalIndexType>::iterator cellIt = xCells.begin(); cellIt!=xCells.end(); cellIt++)
  {
    ElementPtr elem = mesh->getElement(*cellIt);
    for (int i = 0; i<elem->numChildren(); i++)
    {
      xChildren.push_back(elem->getChild(i)->cellID());
    }
  }
  // build children list
  for (vector<GlobalIndexType>::iterator cellIt = yCells.begin(); cellIt!=yCells.end(); cellIt++)
  {
    ElementPtr elem = mesh->getElement(*cellIt);
    for (int i = 0; i<elem->numChildren(); i++)
    {
      yChildren.push_back(elem->getChild(i)->cellID());
    }
  }

  bool meshIsNotRegular = true; // assume it's not regular and check elements
  int i = 0;
  while (meshIsNotRegular && i<maxIters)
  {
    vector<GlobalIndexType> irregularQuadCells,xUpgrades,yUpgrades;
    vector< Teuchos::RCP< Element > > newActiveElements = mesh->activeElements();
    vector< Teuchos::RCP< Element > >::iterator newElemIt;

    for (newElemIt = newActiveElements.begin(); newElemIt != newActiveElements.end(); newElemIt++)
    {
      Teuchos::RCP< Element > current_element = *(newElemIt);
      bool isIrregular = false;
      for (int sideIndex=0; sideIndex < current_element->numSides(); sideIndex++)
      {
        int mySideIndexInNeighbor;
        ElementPtr neighbor = current_element->getNeighbor(mySideIndexInNeighbor, sideIndex);
        if (neighbor.get() != NULL)
        {
          int numNeighborsOnSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor).size();
          if (numNeighborsOnSide > 2) isIrregular=true;
        }
      }
      if (isIrregular)
      {
        int cellID = current_element->cellID();
        bool isXRefined = std::find(xChildren.begin(),xChildren.end(),cellID)!=xChildren.end();
        bool isYRefined = std::find(yChildren.begin(),yChildren.end(),cellID)!=yChildren.end();
        bool isPreviouslyRefined = (isXRefined || isYRefined);
        if (!isPreviouslyRefined)   // if the cell to refine has already been refined anisotropically, don't refine it again,
        {
          irregularQuadCells.push_back(cellID);
        }
        else if (isXRefined)
        {
          yUpgrades.push_back(cellID);
        }
        else if (isYRefined)
        {
          xUpgrades.push_back(cellID);
        }
      }
    }
    if (irregularQuadCells.size()>0)
    {
      mesh->hRefine(irregularQuadCells,RefinementPattern::regularRefinementPatternQuad());
      mesh->hRefine(xUpgrades,RefinementPattern::xAnisotropicRefinementPatternQuad());
      mesh->hRefine(yUpgrades,RefinementPattern::yAnisotropicRefinementPatternQuad());
      irregularQuadCells.clear();
      xUpgrades.clear();
      yUpgrades.clear();
    }
    else
    {
      meshIsNotRegular=false;
    }
    ++i;
  }
  if (i>=maxIters)
  {
    success = false;
  }
  return success;
}

template <typename Scalar>
double TRefinementStrategy<Scalar>::getEnergyError(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].totalEnergyError;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

template <typename Scalar>
GlobalIndexType TRefinementStrategy<Scalar>::getNumElements(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].numElements;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

template <typename Scalar>
GlobalIndexType TRefinementStrategy<Scalar>::getNumDofs(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].numDofs;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

namespace Camellia
{
template class TRefinementStrategy<double>;
}
