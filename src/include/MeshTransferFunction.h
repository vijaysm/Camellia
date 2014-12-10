#ifndef MESH_TRANSFER_FUNCTION
#define MESH_TRANSFER_FUNCTION

/*@HEADER
 // ***********************************************************************
 //
 //                  Camellia Mesh Transfer Function:
 //
 //  A function that is meant to transfer data from one mesh to another.
 //  Implemented for the purpose of space-time meshes, but might also be
 //  useful in the context of loosely coupled multiphysics implementations.
 //
 
 // ***********************************************************************
 //@HEADER
 */

#include "Function.h"
#include "Mesh.h"
#include "BasisCache.h"

#include "RefinementObserver.h"

//! MeshTransferFunction: Given two meshes with a shared interface...
//
/*!
 
 \author Nathan V. Roberts, ALCF.
 
 \date Last modified on 25-Nov-2014.
 */


class MeshTransferFunction : public Function, public RefinementObserver {
  MeshPtr _originalMesh, _newMesh;
  FunctionPtr _originalFunction;
  double _interface_t;
  
  typedef std::pair<GlobalIndexType,unsigned> CellSide; // cellID, side ordinal
  std::map<CellSide,CellSide> _newToOriginalMap;
  std::map<CellSide,CellSide> _originalToNewMap;
  
  std::map<CellSide,CellSide> _activeSideToAncestralSideInNewMesh;
  
  std::map<CellSide, unsigned> _permutationForNewMeshCellSide; // permutation goes from cell side in _newMesh to that in _originalMesh
  
  void findAncestralPairForNewMeshCellSide(const CellSide &newMeshCellSide, CellSide &newMeshCellSideAncestor,
                                           CellSide &originalMeshCellSideAncestor, unsigned &newCellSideAncestorPermutation);
  
  void rebuildMaps();
public:
  MeshTransferFunction(FunctionPtr originalFunction, MeshPtr originalMesh, MeshPtr newMesh, double interface_t);
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  
  bool boundaryValueOnly();
  
  const std::map<CellSide,CellSide> & mapToOriginalMesh() { return _newToOriginalMap; }
  const std::map<CellSide,CellSide> & mapToNewMesh() { return _originalToNewMap; }
  
  // RefinementObserver method:
  void didRepartition(MeshTopologyPtr meshTopology);
  
  ~MeshTransferFunction();
};

#endif