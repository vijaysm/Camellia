// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  LocalDofMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#ifndef __Camellia_debug__LocalDofMapper__
#define __Camellia_debug__LocalDofMapper__

#include "TypeDefs.h"

#include <iostream>

#include "DofOrdering.h"
#include "SubBasisDofMapper.h"

namespace Camellia
{
class LocalDofMapper   // maps a whole trial ordering
{
  DofOrderingPtr _dofOrdering;
  typedef vector< SubBasisDofMapperPtr > BasisMap; // taken together, these maps map a whole basis
  map< int, BasisMap > _volumeMaps; // keys are var IDs (fields)
  vector< map< int, BasisMap > > _sideMaps; // outer index is side ordinal; map keys are var IDs
  map< GlobalIndexType, unsigned > _globalIndexToOrdinal; // maps from GlobalIndex to the ordinal in our globalData container (on the present implementation, the global indices will always be in numerical order)
  map< GlobalIndexType, vector<int> > _globalIndexToVarIDs; // maps from GlobalIndex to its var IDs (determined by which var ID has a map to a given global index) -- one GlobalIndex may have multiple varIDs in the case of traces that map to fields.

  vector< set<GlobalIndexType> > _fittableGlobalDofOrdinalsOnSides;
  set<GlobalIndexType> _fittableGlobalDofOrdinalsInVolume;
  vector< GlobalIndexType > _fittableGlobalIndices;

  int _sideOrdinalToMap;
  int _varIDToMap;

  void filterData(const vector<int> dofIndices, const Intrepid::FieldContainer<double> &data, Intrepid::FieldContainer<double> &filteredData);
  void addSubBasisMapVectorContribution(int varID, int sideIndex, BasisMap &basisMap, const Intrepid::FieldContainer<double> &localData, Intrepid::FieldContainer<double> &globalData, bool fittableGlobalDofsOnly);
  void addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap &basisMap, const Intrepid::FieldContainer<double> &globalData, Intrepid::FieldContainer<double> &localData);
  void addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap &basisMap, const std::map<GlobalIndexType,double> &globalCoefficients, Intrepid::FieldContainer<double> &localCoefficients);
  Intrepid::FieldContainer<double> mapLocalDataMatrix(const Intrepid::FieldContainer<double> &localData, bool fittableGlobalDofsOnly);
  
  void mapLocalDataVector(const Intrepid::FieldContainer<double> &localData, bool fittableGlobalDofsOnly,
                          Intrepid::FieldContainer<double> &mappedDataVector);
  void mapLocalDataVector(const Intrepid::FieldContainer<double> &localData, bool fittableGlobalDofsOnly,
                          const std::vector<std::pair<int,std::vector<int>>> &varIDsAndSideOrdinalsToMap,
                          Intrepid::FieldContainer<double> &mappedDataVector);
  
  Intrepid::FieldContainer<double> _localCoefficientsFitMatrix; // used for fitLocalCoefficients
  vector<GlobalIndexType> _globalIndices;

  // the following is used by LocalDofMappers that have varIDToMap = -1 when fitLocalCoefficients() is called
  map<pair<int,int>, Teuchos::RCP<LocalDofMapper>> _localDofMapperForVarIDAndSide;
  
  std::map<int, GlobalIndexType> _permutationMap; // lazily computed, returned by getPermutationMap for LDMs that are permuation maps.
public:
  LocalDofMapper(DofOrderingPtr dofOrdering, map< int, BasisMap > volumeMaps,
                 set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume,
                 vector< map< int, BasisMap > > sideMaps,
                 vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides,
                 set<GlobalIndexType> unmappedGlobalDofOrdinals = set<GlobalIndexType>(), // extra dof ordinals which aren't mapped as such but should be included in the mapper (used in GMGOperator)
                 int varIDToMap = -1, int sideOrdinalToMap = -1);

  // ! Returns an STL map from the local dof indices to the global dof indices.  Throws an exception if isPermutation() returns false.
  const std::map<int, GlobalIndexType> &getPermutationMap();
  
  Intrepid::FieldContainer<double> mapLocalData(const Intrepid::FieldContainer<double> &localData, bool fittableGlobalDofsOnly);
  
  Intrepid::FieldContainer<double> mapLocalData(const Intrepid::FieldContainer<double> &localData, bool fittableGlobalDofsOnly,
                                                const std::vector<std::pair<int,std::vector<int>>> &varIDsAndSideOrdinalsToMap);
  
  void mapLocalDataSide(const Intrepid::FieldContainer<double> &localData, Intrepid::FieldContainer<double> &mappedData, bool fittableGlobalDofsOnly, int sideOrdinal);
  void mapLocalDataVolume(const Intrepid::FieldContainer<double> &localData, Intrepid::FieldContainer<double> &mappedData, bool fittableGlobalDofsOnly);

  //! Returns true if the action of the LocalDofMapper is equivalent to a relabeling of the local dofs as global dofs (i.e. it is one-to-one and all weights are unity.)
  bool isPermutation() const;
  
  Intrepid::FieldContainer<double> fitLocalCoefficients(const Intrepid::FieldContainer<double> &localCoefficients); // solves normal equations (if the localCoefficients are in the range of the global-to-local operator, then the returned coefficients will be the preimage of localCoefficients under that operator)
  Intrepid::FieldContainer<double> mapGlobalCoefficients(const Intrepid::FieldContainer<double> &globalCoefficients);
  
  // ! this version of mapGlobalCoefficients should be more efficient when the number of global coefficients in the map is small (i.e. there would be lots of zeroes in the FieldContainer version)
  void mapGlobalCoefficients(const std::map<GlobalIndexType,double> &globalCoefficients, Intrepid::FieldContainer<double> &localCoefficients);
  
  const vector<GlobalIndexType> &fittableGlobalIndices();
  const vector<GlobalIndexType> &globalIndices();
  set<GlobalIndexType> globalIndicesForSubcell(int varID, unsigned d, unsigned subcord);

  void printMappingReport();

  void reverseParity(set<int> fluxVarIDs, set<unsigned> sideOrdinals); // multiplies corresponding sideMaps by -1
};
typedef Teuchos::RCP<LocalDofMapper> LocalDofMapperPtr;
}

#endif /* defined(__Camellia_debug__LocalDofMapper__) */
