//
//  HessianErrorIndicator.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/9/16.
//
//

#include "HessianErrorIndicator.h"
#include "CamelliaCellTools.h"
#include "SerialDenseWrapper.h"

using namespace Camellia;
using namespace Intrepid;

// explicitly instantiate double constructors.
template HessianErrorIndicator<double>::HessianErrorIndicator(TSolutionPtr<double> soln, VarPtr varForHessian);
template HessianErrorIndicator<double>::HessianErrorIndicator(TSolutionPtr<double> soln, VarPtr varForHessian, double hPower);

template <typename Scalar>
HessianErrorIndicator<Scalar>::HessianErrorIndicator(TSolutionPtr<Scalar> soln, VarPtr varForHessian) : ErrorIndicator(soln->mesh())
{
  _solution = soln;
  _var = varForHessian;
  
  int spaceDim = _solution->mesh()->getDimension();
  _hPower = 2.0 + spaceDim / 2.0;
}

template <typename Scalar>
HessianErrorIndicator<Scalar>::HessianErrorIndicator(TSolutionPtr<Scalar> soln, VarPtr varForHessian, double hPower) : ErrorIndicator(soln->mesh())
{
  _solution = soln;
  _var = varForHessian;
  _hPower = hPower;
}

//! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
template <typename Scalar>
void HessianErrorIndicator<Scalar>::measureError()
{
  // clear previous measures
  this->_localErrorMeasures.clear();
  
  // imitates https://dealii.org/developer/doxygen/deal.II/namespaceDerivativeApproximation.html
  
  // we require that u be a scalar field variable
  TEUCHOS_TEST_FOR_EXCEPTION(_var->rank() != 0, std::invalid_argument, "varForHessian must be a scalar variable");
  TEUCHOS_TEST_FOR_EXCEPTION(_var->varType() != FIELD, std::invalid_argument, "varForHessian must be a field variable");
  
  auto myCells = &_mesh->cellIDsInPartition();
  
  int onePoint = 1;
  MeshTopologyViewPtr meshTopo = _solution->mesh()->getTopology();
  int spaceDim = meshTopo->getDimension();
  
  set<GlobalIndexType> cellsAndNeighborsSet;
  for (GlobalIndexType myCellID : *myCells)
  {
    cellsAndNeighborsSet.insert(myCellID);
    CellPtr cell = meshTopo->getCell(myCellID);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    cellsAndNeighborsSet.insert(neighborIDs.begin(),neighborIDs.end());
  }
  vector<GlobalIndexType> cellsAndNeighbors(cellsAndNeighborsSet.begin(),cellsAndNeighborsSet.end());
  
  set<GlobalIndexType> remoteNeighbors; // neighbors that are not in myCells
  for (GlobalIndexType cellID : cellsAndNeighbors)
  {
    if (myCells->find(cellID) == myCells->end())
    {
      remoteNeighbors.insert(cellID);
    }
  }
  
  // get any off-rank solution data we may need for gradient computation:
  _solution->importSolutionForOffRankCells(cellsAndNeighborsSet);
  
  int cellsAndNeighborsCount = cellsAndNeighbors.size();
  
  FieldContainer<double> cellValues(cellsAndNeighborsCount,onePoint); // values at cell centers
  FieldContainer<double> cellCenters(cellsAndNeighborsCount,spaceDim);
  FieldContainer<double> cellDiameter(cellsAndNeighborsCount,onePoint); // h-values
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForTopology;
  
  FunctionPtr hFunction = Function::h();
  FunctionPtr solnFunction = Function::solution(_var, _solution);
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
      basisCacheForTopology[cellTopo->getKey()]->setMesh(_solution->mesh());
    }
    BasisCachePtr basisCache = basisCacheForTopology[cellTopo->getKey()];
    basisCache->setPhysicalCellNodes(_solution->mesh()->physicalCellNodesForCell(cellID), {cellID}, false);
    
    FieldContainer<double> cellValue(cellValueDim,&cellValues(cellOrdinal,0));
    solnFunction->values(cellValue, basisCache);
    for (int d=0; d<spaceDim; d++)
    {
      cellCenters(cellOrdinal,d) = basisCache->getPhysicalCubaturePoints()(0,0,d);
    }
    if (_hPower != 0)
    {
      cellDiameter(cellOrdinal,0) = _solution->mesh()->getCellMeasure(cellID);
    }
  }
  
  // compute the gradients for owned cells
  FieldContainer<double> Y(spaceDim,spaceDim); // the matrix we'll invert to compute the gradient
  FieldContainer<double> b(spaceDim); // RHS for matrix problem
  FieldContainer<double> grad(spaceDim); // LHS for matrix problem
  vector<double> distanceVector(spaceDim);
  map<GlobalIndexType,vector<double>> gradients;
  for (GlobalIndexType myCellID : *myCells)
  {
    Y.initialize(0.0);
    b.initialize(0.0);
    CellPtr cell = meshTopo->getCell(myCellID);
    int myOrdinalInCellAndNeighbors = cellIDToOrdinal[myCellID];
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
    vector<double> gradVector(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      gradVector[d] = grad[d];
    }
    gradients[myCellID] = gradVector;
  }
  
  // request remote neighbors' gradients
  Epetra_CommPtr Comm = _mesh->Comm();
  
  int myRank = Comm->MyPID();
  std::map<int,std::vector<std::pair<int,GlobalIndexType>>> requests; // owning PID -> (myPID, cell whose gradient we want)
  for (GlobalIndexType remoteCellID : remoteNeighbors)
  {
    int rank = _mesh->partitionForCellID(remoteCellID);
    requests[rank].push_back({myRank,remoteCellID});
  }
  
  std::vector<std::pair<int,GlobalIndexType>> requestsReceived;
  MPIWrapper::sendDataVectors(Comm, requests, requestsReceived);
  
  std::map<int,std::vector<std::pair<GlobalIndexType,double>>> responsesToSend;
  std::vector<std::pair<GlobalIndexType,double>> responsesReceived;
  for (auto request : requestsReceived)
  {
    int remotePID = request.first;
    GlobalIndexType myCellID = request.second;
    TEUCHOS_TEST_FOR_EXCEPTION(!_mesh->myCellsInclude(myCellID),std::invalid_argument, "request received for non-owned cellID");
    vector<double> gradient = gradients[myCellID];
    for (double gradient_comp : gradient)
    {
      responsesToSend[remotePID].push_back({myCellID,gradient_comp});
    }
  }
  MPIWrapper::sendDataVectors(Comm, responsesToSend, responsesReceived);
  for (auto response : responsesReceived)
  {
    GlobalIndexType remoteCellID = response.first;
    double gradient_comp = response.second;
    gradients[remoteCellID].push_back(gradient_comp);
  }
  
  // now, compute Hessian as Y^{-1} * sum_{K'} (y_K' / norm{y_K'} \tensor (grad u(x_K') - grad u(x_K)) / norm{y_K'})
  b.resize(spaceDim, spaceDim); // RHS for matrix problem -- now matrix-valued
  FieldContainer<double> hessian(spaceDim, spaceDim); // LHS for matrix problem -- now matrix-valued
  for (GlobalIndexType myCellID : *myCells)
  {
    Y.initialize(0.0);
    b.initialize(0.0);
    CellPtr cell = meshTopo->getCell(myCellID);
    int myOrdinalInCellAndNeighbors = cellIDToOrdinal[myCellID];
    vector<double> myGradient = gradients[myCellID];
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    for (GlobalIndexType neighborID : neighborIDs)
    {
      int neighborOrdinalInCellAndNeighbors = cellIDToOrdinal[neighborID];
      vector<double> neighborGradient = gradients[neighborID];
      
      double dist_squared = 0;
      for (int d=0; d<spaceDim; d++)
      {
        distanceVector[d] = cellCenters(neighborOrdinalInCellAndNeighbors,d) - cellCenters(myOrdinalInCellAndNeighbors,d);
        dist_squared += distanceVector[d] * distanceVector[d];
      }
      
      for (int d1=0; d1<spaceDim; d1++)
      {
        for (int d2=0; d2<spaceDim; d2++)
        {
          b(d1,d2) += distanceVector[d1] * (neighborGradient[d2] - myGradient[d2]) / dist_squared;
          
          Y(d1,d2) += distanceVector[d1] * distanceVector[d2] / dist_squared;
        }
      }
    }
    SerialDenseWrapper::solveSystem(hessian, Y, b);
    
    // symmetrize:
    for (int d1=0; d1<spaceDim; d1++)
    {
      for (int d2=d1+1; d2<spaceDim; d2++)
      {
        hessian(d1,d2) = (hessian(d1,d2) + hessian(d2,d1))/2.0;
      }
    }
    
    FieldContainer<double> lambda_real(spaceDim), lambda_imag(spaceDim);
    SerialDenseWrapper::eigenvalues(hessian, lambda_real, lambda_imag);
    
    double hessian_2norm = 0.0; // max eigenvalue
    
    for (int d=0; d<spaceDim; d++)
    {
      double eigenvalue_mag = sqrt(lambda_real(d)*lambda_real(d) + lambda_imag(d)*lambda_imag(d));
      hessian_2norm = max(hessian_2norm,eigenvalue_mag);
    }
    if (_hPower == 0)
    {
      _localErrorMeasures[myCellID] = hessian_2norm;
    }
    else
    {
      _localErrorMeasures[myCellID] = hessian_2norm * pow(cellDiameter(myOrdinalInCellAndNeighbors,0), _hPower);
    }
  }
}