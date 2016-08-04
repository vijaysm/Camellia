// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

//
//  BasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

#include "Intrepid_Basis.hpp"
//#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"

namespace Camellia
{
template<class Scalar, class ArrayScalar>
Basis<Scalar,ArrayScalar>::Basis()
{
  _basisTagsAreSet = false;
  _functionSpace = Camellia::FUNCTION_SPACE_UNKNOWN;
}

template<class Scalar, class ArrayScalar>
void Basis<Scalar,ArrayScalar>::CHECK_VALUES_ARGUMENTS(const ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const
{
  // for VALUE, GRAD, and DIV, we can say what happens to the rank:
  // (for CURL, it's different between 2D and 3D--and in 2D it depends on whether you're taking the curl of a scalar or a vector quantity)
  int UNKNOWN_RANK_CHANGE = -2;
  int rankChange=UNKNOWN_RANK_CHANGE;

  if (operatorType == Intrepid::OPERATOR_VALUE)
  {
    rankChange = 0;
  }
  else if (operatorType == Intrepid::OPERATOR_DIV)
  {
    rankChange = -1;
  }
  else if (operatorType == Intrepid::OPERATOR_GRAD)
  {
    rankChange = 1;
  }
  if (operatorType == Intrepid::OPERATOR_CURL)
  {
    if (this->rangeDimension() == 3)
    {
      rankChange = 0;
    }
    else if (this->rangeDimension() == 2)
    {
      if (this->rangeRank() == 0)
      {
        rankChange = 1;
      }
      else if (this->rangeRank() == 1)
      {
        rankChange = -1;
      }
    }
  }

  if (rankChange != UNKNOWN_RANK_CHANGE)
  {
    // values should have shape: (F,P[,D,D,...]) where the # of D's = rank of the basis's range
    if (values.rank() != 2 + rangeRank() + rankChange)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2 + rangeRank() + rankChange, std::invalid_argument, "values should have shape (F,P,[D,D,...]).");
    }
    for (int d=0; d<rangeRank() + rankChange; d++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(2+d) != rangeDimension(), std::invalid_argument, "values should have shape (F,P,[D,D,...]).");
    }
  }
  // refPoints should have shape: (P,D)
  if (refPoints.rank() != 2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(refPoints.rank() != 2, std::invalid_argument, "refPoints should have shape (P,D).");
  }
  if ( refPoints.dimension(1) != domainTopology()->getDimension() )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(refPoints.dimension(1) != domainTopology()->getDimension(), std::invalid_argument, "refPoints should have shape (P,D).");
  }
}

template<class Scalar, class ArrayScalar>
CellTopoPtr Basis<Scalar,ArrayScalar>::domainTopology() const
{
  return _domainTopology;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::getCardinality() const
{
  return this->_basisCardinality;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::getDegree() const
{
  return this->_basisDegree;
}
  
  template<class Scalar, class ArrayScalar>
  void Basis<Scalar, ArrayScalar>::getDofCoords(ArrayScalar & DofCoords) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getDofCoords not implemented by basis!");
  }
  
template<class Scalar, class ArrayScalar>
int Basis<Scalar, ArrayScalar>::getDofOrdinal(const int subcDim,
    const int subcOrd,
    const int subcDofOrd) const
{
//    std::cout << "int Basis<Scalar, ArrayScalar>::getDofOrdinal" << std::endl;
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  // Use .at() for bounds checking
  int dofOrdinal = _tagToOrdinal.at(subcDim).at(subcOrd).at(subcDofOrd);
  TEUCHOS_TEST_FOR_EXCEPTION( (dofOrdinal == -1), std::invalid_argument,
                              ">>> ERROR (Basis): Invalid DoF tag");
  return dofOrdinal;
}

template<class Scalar,class ArrayScalar>
const std::vector<std::vector<std::vector<int> > > & Basis<Scalar, ArrayScalar>::getDofOrdinalData( ) const
{
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  return _tagToOrdinal;
}


template<class Scalar, class ArrayScalar>
const std::vector<int>&  Basis<Scalar, ArrayScalar>::getDofTag(int dofOrd) const
{
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  // Use .at() for bounds checking
  return _ordinalToTag.at(dofOrd);
}

template<class Scalar, class ArrayScalar>
const std::vector<std::vector<int> > & Basis<Scalar, ArrayScalar>::getAllDofTags() const
{
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  return _ordinalToTag;
}

  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSide(int sideOrdinal) const
  {
    int sideDim = this->_domainTopology->getDimension() - 1;
    std::vector<int> dofOrdinals = this->dofOrdinalsForSubcell(sideDim, sideOrdinal, 0);
    std::set<int> dofOrdinalsSet(dofOrdinals.begin(),dofOrdinals.end());
    return dofOrdinalsSet;
  }
  
template<class Scalar, class ArrayScalar>
const std::vector<int> &Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcell(int subcellDim, int subcellIndex) const
{
//  static std::vector<int> emptyVector;
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
//  std::vector<int> dofOrdinals;
//  int firstDofOrdinal = -1;
//  if (_tagToOrdinal.size() > subcellDim)
//  {
//    if (_tagToOrdinal[subcellDim].size() > subcellIndex)
//    {
//      if (_tagToOrdinal[subcellDim][subcellIndex].size() > 0)
//      {
//        firstDofOrdinal = _tagToOrdinal[subcellDim][subcellIndex][0];
//      }
//    }
//  }
//  if (firstDofOrdinal == -1)   // no matching dof ordinals
//  {
//    return emptyVector;
//  }
//  int numDofs = _ordinalToTag.at(firstDofOrdinal)[3];
////  int numDofs = _tagToOrdinal[subcellDim][subcellIndex].size();
//
//  for (int dofIndex=0; dofIndex<numDofs; dofIndex++)
//  {
//    int dofOrdinal = _tagToOrdinal.at(subcellDim).at(subcellIndex).at(dofIndex); // -1 indicates invalid entry...
//    TEUCHOS_TEST_FOR_EXCEPTION(dofOrdinal < 0, std::invalid_argument, "invalid entry encountered in _tagToOrdinal");
//  }
  
  { // DEBUGGING -- sanity checks:
    for (int dofOrdinal : _tagToOrdinal[subcellDim][subcellIndex])
    {
      TEUCHOS_TEST_FOR_EXCEPTION(dofOrdinal < 0, std::invalid_argument, "dofOrdinal can't be < 0");
    }
    int firstDofOrdinal = -1;
    if (_tagToOrdinal.size() > subcellDim)
    {
      if (_tagToOrdinal[subcellDim].size() > subcellIndex)
      {
        if (_tagToOrdinal[subcellDim][subcellIndex].size() > 0)
        {
          firstDofOrdinal = _tagToOrdinal[subcellDim][subcellIndex][0];
        }
      }
    }
    int numDofs;
    
    if (firstDofOrdinal == -1)   // no matching dof ordinals
    {
      numDofs = 0;
    }
    else
    {
      numDofs = _ordinalToTag.at(firstDofOrdinal)[3];
      if (numDofs == 0)
      {
        // inconsistent, but we can survive that
        // (bug in HDIV_HEX_In, at least when n=1; hopefully not otherwise)
        numDofs = 1;
      }
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION(numDofs != _tagToOrdinal[subcellDim][subcellIndex].size(), std::invalid_argument, "wrong number of dofs in trimmed container");
  }
  
  return _tagToOrdinal[subcellDim][subcellIndex];
}
  
//  template<class Scalar, class ArrayScalar>
//  void Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcell(int subcellDim, int subcellIndex, std::vector<int> &dofOrdinals) const
//{
//  if (!_basisTagsAreSet)
//  {
//    initializeTagsAndTrim();
//    _basisTagsAreSet = true;
//  }
//
//  int firstDofOrdinal = -1;
//  if (_tagToOrdinal.size() > subcellDim)
//  {
//    if (_tagToOrdinal[subcellDim].size() > subcellIndex)
//    {
//      if (_tagToOrdinal[subcellDim][subcellIndex].size() > 0)
//      {
//        firstDofOrdinal = _tagToOrdinal[subcellDim][subcellIndex][0];
//      }
//    }
//  }
//  if (firstDofOrdinal == -1)   // no matching dof ordinals
//  {
//    dofOrdinals.resize(0);
//    return;
//  }
//  int numDofs = _ordinalToTag.at(firstDofOrdinal)[3];
////  int numDofs = _tagToOrdinal[subcellDim][subcellIndex].size();
//  dofOrdinals.resize(numDofs);
//  
//  for (int dofIndex=0; dofIndex<numDofs; dofIndex++)
//  {
//    int dofOrdinal = _tagToOrdinal.at(subcellDim).at(subcellIndex).at(dofIndex); // -1 indicates invalid entry...
//    if (dofOrdinal >= 0)
//    {
//      dofOrdinals[dofIndex] = dofOrdinal;
//    }
//    else
//    {
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid entry encountered in _tagToOrdinal");
//    }
//  }
//  std::sort(dofOrdinals.begin(), dofOrdinals.end());
//}

template<class Scalar, class ArrayScalar>
std::vector<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcell(int subcellDim, int subcellIndex, int minimumSubSubcellDimension) const
{
  if (minimumSubSubcellDimension > subcellDim)
  {
    return std::vector<int>();
  }
  // to start, get the ones defined on the subcell itself:
  std::vector<int> dofOrdinals = this->dofOrdinalsForSubcell(subcellDim, subcellIndex);

  CellTopoPtr cellTopo = this->domainTopology();
  CellTopoPtr sideTopo = cellTopo->getSubcell(subcellDim, subcellIndex);

//    const CellTopologyData *cellTopoData = this->domainTopology()->getCellTopologyData();
//    const CellTopologyData *sideTopoData = this->domainTopology()->getCellTopologyData(subcellDim, subcellIndex);

  // Now, we want to look up all the subcells of the "side" of dimension > minimumSubSubcellDimension
  // For each of these, we need to:
  //  - find the appropriate subcell index in the larger topology
  //  - add all dof ordinals associated with that subcell index

  // ssc: sub-subcell
  for (int d_ssc = minimumSubSubcellDimension; d_ssc < subcellDim; d_ssc++)
  {
    int numSubSubcells_d = sideTopo->getSubcellCount(d_ssc); // numSubSubcells of dim d
    for (int ssc=0; ssc<numSubSubcells_d; ssc++)   //
    {
      CellTopoPtr subSubCellTopo = sideTopo->getSubcell(d_ssc,ssc);
      int numNodes_ssc = subSubCellTopo->getNodeCount();
      std::set<int> cellNodeIndices;
      for (int i=0; i < numNodes_ssc; i++)
      {
        unsigned subcellNodeIndex = sideTopo->getNodeMap(d_ssc, ssc, i); //  ->subcell[d_ssc][ssc].node[i];
        unsigned cellNodeIndex = cellTopo->getNodeMap(subcellDim, subcellIndex, subcellNodeIndex); //cellTopoData->subcell[subcellDim][subcellIndex].node[subcellNodeIndex];
        cellNodeIndices.insert(cellNodeIndex);
      }
      // this is a bit involved, because CellTopology doesn't give a way to go from a set of nodes to a subcell defined on those nodes

      // now, examine each of the d_ssc-dimensional subcells of cell to look for one that matches all cellNodeIndices
      unsigned numSubcells = cellTopo->getSubcellCount(d_ssc);
      unsigned matchingSubcellIndex = numSubcells;

      if (d_ssc==0)   // vertex
      {
        // map the nodeIndex ssc in the subcell to the cellNodeIndex in the cell
        matchingSubcellIndex = cellTopo->getNodeMap(subcellDim, subcellIndex, ssc); // cellTopoData->subcell[subcellDim][subcellIndex].node[ssc];
      }
      else
      {
        for (unsigned sc=0; sc<numSubcells; sc++)
        {
          bool matches = true;
          if (numNodes_ssc != subSubCellTopo->getNodeCount())
          {
            matches = false;
          }
          else
          {
            for (int i=0; i < numNodes_ssc; i++)
            {
              unsigned cellNodeIndex = cellTopo->getNodeMap(d_ssc,sc,i); // ->subcell[d_ssc][sc].node[i];
              if (cellNodeIndices.find(cellNodeIndex) == cellNodeIndices.end())
              {
                matches = false;
                break;
              }
            }
          }
          if (matches == true)
          {
            matchingSubcellIndex = sc;
            break;
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION(matchingSubcellIndex >= numSubcells, std::invalid_argument, "matching subcell not found");
      }
      const std::vector<int> dofOrdinals_ssc = this->dofOrdinalsForSubcell(d_ssc,matchingSubcellIndex);
      dofOrdinals.insert(dofOrdinals.end(), dofOrdinals_ssc.begin(), dofOrdinals_ssc.end());
    }
  }
  std::sort(dofOrdinals.begin(), dofOrdinals.end());
  return dofOrdinals;
}

template<class Scalar, class ArrayScalar>
const std::vector<int>& Basis<Scalar,ArrayScalar>::dofOrdinalsForEdge(int edgeIndex) const
{
  static const int edgeDim = 1;
  return dofOrdinalsForSubcell(edgeDim, edgeIndex);
}


template<class Scalar, class ArrayScalar>
std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcells(int subcellDim, bool includeLesserDimensions) const
{
  std::set<int> dofOrdinals;
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  if ((subcellDim > 0) && includeLesserDimensions)
  {
    dofOrdinals = dofOrdinalsForSubcells(subcellDim-1,true);
  }
  if (this->_tagToOrdinal.size() < subcellDim+1)   // none of dimension subcellDim
  {
    return dofOrdinals;
  }

  int numSubcells = this->domainTopology()->getSubcellCount(subcellDim);
  for (int subcellIndex=0; subcellIndex<numSubcells; subcellIndex++)
  {
    std::vector<int> dofOrdinalsForSubcell = this->dofOrdinalsForSubcell(subcellDim,subcellIndex);
    dofOrdinals.insert(dofOrdinalsForSubcell.begin(), dofOrdinalsForSubcell.end());
  }
  return dofOrdinals;
}

template<class Scalar, class ArrayScalar>
std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForEdges(bool includeVertices) const
{
  int edgeDim = 1;
  return dofOrdinalsForSubcells(edgeDim,includeVertices);
}

template<class Scalar, class ArrayScalar>
std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForFaces(bool includeVerticesAndEdges) const
{
  int faceDim = 2;
  return dofOrdinalsForSubcells(faceDim,includeVerticesAndEdges);
}

template<class Scalar, class ArrayScalar>
const std::vector<int> &Basis<Scalar,ArrayScalar>::dofOrdinalsForInterior() const
{
  int interiorDim = this->domainTopology()->getDimension();
  return dofOrdinalsForSubcell(interiorDim, false);
}

template<class Scalar, class ArrayScalar>
const std::vector<int> &Basis<Scalar,ArrayScalar>::dofOrdinalsForVertex(int vertexIndex) const
{
  const static int vertexDim = 0;
  return this->dofOrdinalsForSubcell(vertexDim, vertexIndex);
}

template<class Scalar, class ArrayScalar>
std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForVertices() const
{
  const static int vertexDim = 0;
  return this->dofOrdinalsForSubcells(vertexDim, false);
}

template<class Scalar, class ArrayScalar>
Camellia::EFunctionSpace Basis<Scalar,ArrayScalar>::functionSpace() const
{
  return this->_functionSpace;
}

template<class Scalar, class ArrayScalar>
Camellia::EFunctionSpace Basis<Scalar,ArrayScalar>::functionSpace(int tensorialRank) const
{
  if (tensorialRank==0)
    return this->_functionSpace;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"tensorialRank exceeds the tensorial degree of the basis");
}
  
  template<class Scalar, class ArrayScalar>
  Camellia::EFunctionSpace Basis<Scalar,ArrayScalar>::functionSpace(int subcDim, int subcOrdinal) const
  {
    switch (this->_functionSpace)
    {
      case FUNCTION_SPACE_HGRAD_SPACE_HVOL_TIME:
      case FUNCTION_SPACE_HVOL_SPACE_HGRAD_TIME:
      {
        TEUCHOS_TEST_FOR_EXCEPTION(this->_domainTopology()->getTensorialDegree() == 0, std::invalid_argument, "mixed function spaces only supported for tensorial domains");
        CellTopoPtr subcell= this->_domainTopology()->getSubcell(subcDim,subcOrdinal);
        if (subcell->getTensorialDegree() < this->_domainTopology()->getTensorialDegree())
        {
          // then this is the spatial part
          return this->functionSpace(0); // 0: tensorial rank of spatial part
        }
        else
        {
          // any tensorial subcell other than Node x Line_2 has a mixed function space; Node x Line_2 has the temporal function space
          if (subcell->getNodeCount() == 2)
          {
            return this->functionSpace(1); // 1: tensorial rank of temporal part
          }
          else
          {
            return this->_functionSpace;
          }
        }
      }
        break;
      default:
        return this->_functionSpace;
    }
  }

  template<class Scalar, class ArrayScalar>
  void Basis<Scalar,ArrayScalar>::initializeTagsAndTrim() const
  {
    initializeTags();
    
    // trim so that _tagToOrdinal vectors have the right size (no -1 filling)
    CellTopoPtr domainTopo = this->domainTopology();
    int domainDim = domainTopo->getDimension();
    
    auto tagToOrdinalTrimmed = std::vector<std::vector<std::vector<int> > >(domainDim + 1);
    tagToOrdinalTrimmed.resize(domainDim + 1);
    
    for (int d=0; d<=domainDim; d++)
    {
      int subcellCount = domainTopo->getSubcellCount(d);
      tagToOrdinalTrimmed[d].resize(subcellCount);
      for (int subcOrd=0; subcOrd < subcellCount; subcOrd++)
      {
        vector<int> dofOrdinals;
        // some subclasses don't fill the container for subcells when subcells of a given dimension don't have any dofOrdinals defined...  We'll set it to be an empty vector<int> in that case...
        if ((d < this->_tagToOrdinal.size()) && (subcOrd < this->_tagToOrdinal[d].size()))
        {
          int numDofs = this->_tagToOrdinal[d][subcOrd].size();
          for (int dofIndex = 0; dofIndex < numDofs; dofIndex++)
          {
            int dofOrdinal = this->_tagToOrdinal[d][subcOrd][dofIndex];
            if (dofOrdinal >= 0)
            {
              dofOrdinals.push_back(dofOrdinal);
            }
          }
          std::sort(dofOrdinals.begin(),dofOrdinals.end());
        }
        tagToOrdinalTrimmed[d][subcOrd] = dofOrdinals;
      }
    }
    _tagToOrdinal = tagToOrdinalTrimmed;
  }
  
template<class Scalar, class ArrayScalar>
bool Basis<Scalar,ArrayScalar>::isConforming() const
{
  return false;
}

template<class Scalar, class ArrayScalar>
bool Basis<Scalar,ArrayScalar>::isModal() const
{
  return false;
}
  
template<class Scalar, class ArrayScalar>
bool Basis<Scalar,ArrayScalar>::isNodal() const
{
  return false;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::rangeDimension() const
{
  return _rangeDimension;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::rangeRank() const
{
  return _rangeRank;
}

template<class Scalar, class ArrayScalar>
bool IntrepidBasisWrapper<Scalar,ArrayScalar>::isConforming() const
{
  return true;
}

template<class Scalar, class ArrayScalar>
bool IntrepidBasisWrapper<Scalar,ArrayScalar>::isNodal() const
{
  return true;
}

template<class Scalar, class ArrayScalar>
IntrepidBasisWrapper<Scalar,ArrayScalar>::IntrepidBasisWrapper(Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > intrepidBasis,
    int rangeDimension, int rangeRank,
    Camellia::EFunctionSpace fs)
{
  _intrepidBasis = intrepidBasis;
  this->_rangeDimension = rangeDimension;
  this->_rangeRank = rangeRank;
  this->_functionSpace = fs;
  this->_basisCardinality = _intrepidBasis->getCardinality();
  this->_basisDegree = _intrepidBasis->getDegree();
  this->_domainTopology = CellTopology::cellTopology( _intrepidBasis->getBaseCellTopology() );

  bool isDiscontinuous =  Camellia::functionSpaceIsDiscontinuous(this->_functionSpace);
  
  if (isDiscontinuous)
  {
    bool upgradeHVOL = true;
    Camellia::EFunctionSpace continuousFS = Camellia::continuousSpaceForDiscontinuous(fs,upgradeHVOL);
    this->_continuousBasis = Teuchos::rcp( new IntrepidBasisWrapper(intrepidBasis, rangeDimension,
                                                                    rangeRank, continuousFS));
  }
}

template<class Scalar, class ArrayScalar>
void IntrepidBasisWrapper<Scalar,ArrayScalar>::initializeTags() const
{
  // we leave tag initialization to the _intrepidBasis object, but we'll keep a copy ourselves:
  this->_tagToOrdinal = _intrepidBasis->getDofOrdinalData();
  this->_ordinalToTag = _intrepidBasis->getAllDofTags();

  bool isDiscontinuous =  Camellia::functionSpaceIsDiscontinuous(this->_functionSpace);

  // if this is an L^2 basis (potentially wrapping a non-L^2 Intrepid basis--to date, Intrepid doesn't have any L^2 bases, so we usually use H^1 of one lower degree for L^2),
  // then we should rework the data structures a bit...
  if (isDiscontinuous)
  {
    std::vector<int> tag(4);
    int domainDimension = this->domainTopology()->getDimension();
    this->_ordinalToTag = std::vector< std::vector<int> >(this->_basisCardinality);
    this->_tagToOrdinal = std::vector<std::vector<std::vector<int> > >(domainDimension + 1);
    for (int d=0; d<=domainDimension; d++)
    {
      int subcellCount = this->domainTopology()->getSubcellCount(d);
      this->_tagToOrdinal[d].resize(subcellCount);
    }
    this->_tagToOrdinal[domainDimension] = std::vector<std::vector<int> >(1);
    this->_tagToOrdinal[domainDimension][0] = std::vector<int>(this->_basisCardinality);

//    for (int d=0; d<this->_tagToOrdinal.size(); d++)
//    {
//      for (int subcOrd=0; subcOrd < this->_tagToOrdinal[d].size(); subcOrd++)
//      {
//        for (int subcDofOrd=0; subcDofOrd< this->_tagToOrdinal[d][subcOrd].size(); subcDofOrd++)
//        {
//          this->_tagToOrdinal[d][subcOrd][subcDofOrd] = -1;
//        }
//      }
//    }

    for (int dofOrdinal = 0; dofOrdinal < this->_basisCardinality; dofOrdinal++)
    {
      tag[0] = domainDimension; // dimension of the subcell
      tag[1] = 0; // subcell ordinal
      tag[2] = dofOrdinal; // ordinal of the dof relative to subcell
      tag[3] = this->_basisCardinality; // total # of dofs associated with subcell
      this->_ordinalToTag[dofOrdinal] = tag;
      this->_tagToOrdinal[domainDimension][0][dofOrdinal] = dofOrdinal;
    }
  }
}

template<class Scalar, class ArrayScalar>
Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > IntrepidBasisWrapper<Scalar,ArrayScalar>::intrepidBasis()
{
  return _intrepidBasis;
}
  
  template<class Scalar, class ArrayScalar>
  void IntrepidBasisWrapper<Scalar, ArrayScalar>::getDofCoords(ArrayScalar & DofCoords) const
  {
    Intrepid::DofCoordsInterface<ArrayScalar>* dofCoordsImplementor = dynamic_cast<Intrepid::DofCoordsInterface<ArrayScalar>*>(this->_intrepidBasis.get());
    if (dofCoordsImplementor != NULL)
    {
      dofCoordsImplementor->getDofCoords(DofCoords);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getDofCoords not implemented by basis!");
    }
  }

  template<class Scalar, class ArrayScalar>
  std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForSide(int sideOrdinal) const
  {
    if (this->_continuousBasis == Teuchos::null)
    {
      return this->Basis<Scalar,ArrayScalar>::dofOrdinalsForSide(sideOrdinal);
    }
    else
    {
      return this->_continuousBasis->dofOrdinalsForSide(sideOrdinal);
    }
  }
  
template<class Scalar, class ArrayScalar>
void IntrepidBasisWrapper<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const
{
  this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);
  return _intrepidBasis->getValues(values,refPoints,operatorType);
}

} // namespace Camellia