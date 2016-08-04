// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-Intrepid in the licenses directory.
//
// @HEADER
//
//  Basis_HVOL_TRI_C0_FEMDef.hpp
//  Camellia
//
//  Created by Nate Roberts on 4/18/16.
//

#ifndef Camellia_Basis_HVOL_TRI_C0_FEMDef_hpp
#define Camellia_Basis_HVOL_TRI_C0_FEMDef_hpp

namespace Intrepid {
  
  template<class Scalar, class ArrayScalar>
  Basis_HVOL_TRI_C0_FEM<Scalar, ArrayScalar>::Basis_HVOL_TRI_C0_FEM()
  {
    this -> basisCardinality_  = 1;
    this -> basisDegree_       = 0;    
    this -> basisCellTopology_ = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() );
    this -> basisType_         = BASIS_FEM_DEFAULT;
    this -> basisCoordinates_  = COORDINATES_CARTESIAN;
    this -> basisTagsAreSet_   = false;
  }
  
  template<class Scalar, class ArrayScalar>
  void Basis_HVOL_TRI_C0_FEM<Scalar, ArrayScalar>::initializeTags() {
    
    // Basis-dependent intializations
    int tagSize  = 1;        // size of DoF tag, i.e., number of fields in the tag
    int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim 
    int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
    int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
    
    // An array with local DoF tags assigned to basis functions, in the order of their local enumeration 
    int tags[]  = 
    { 0, 0, 0, 1 };
    
    // Basis-independent function sets tag and enum data in tagToOrdinal_ and ordinalToTag_ arrays:
    Intrepid::setOrdinalTagData(this -> tagToOrdinal_,
                                this -> ordinalToTag_,
                                tags,
                                this -> basisCardinality_,
                                tagSize,
                                posScDim,
                                posScOrd,
                                posDfOrd);
  }
  
  
  
  template<class Scalar, class ArrayScalar>
  void Basis_HVOL_TRI_C0_FEM<Scalar, ArrayScalar>::getValues(ArrayScalar &        outputValues,
                                                               const ArrayScalar &  inputPoints,
                                                               const EOperator      operatorType) const {
    
    // Verify arguments
#ifdef HAVE_INTREPID_DEBUG
    Intrepid::getValues_HGRAD_Args<Scalar, ArrayScalar>(outputValues,
                                                        inputPoints,
                                                        operatorType,
                                                        this -> getBaseCellTopology(),
                                                        this -> getCardinality() );
#endif
    
    // Number of evaluation points = dim 0 of inputPoints
    int dim0 = inputPoints.dimension(0);  
    
    // Temporaries: (x,y) coordinates of the evaluation point
    Scalar x = 0.0;                                    
    Scalar y = 0.0;                                    
    
    switch (operatorType) {
        
      case OPERATOR_VALUE:
        for (int i0 = 0; i0 < dim0; i0++) {
          x = inputPoints(i0, 0);
          y = inputPoints(i0, 1);
          
          // outputValues is a rank-2 array with dimensions (basisCardinality_, dim0)
          outputValues(0, i0) = 1.0;
        }
        break;
        
      case OPERATOR_GRAD:
      case OPERATOR_D1:
        for (int i0 = 0; i0 < dim0; i0++) {
          x = inputPoints(i0,0);
          y = inputPoints(i0,1);
          
          // outputValues is a rank-3 array with dimensions (basisCardinality_, dim0, spaceDim)
          outputValues(0, i0, 0) = 0.0;
          outputValues(0, i0, 1) = 0.0;
        }
        break;
        
      case OPERATOR_CURL:
        for (int i0 = 0; i0 < dim0; i0++) {
          x = inputPoints(i0,0);
          y = inputPoints(i0,1);
          
          // outputValues is a rank-3 array with dimensions (basisCardinality_, dim0, spaceDim)
          outputValues(0, i0, 0) = 0.0;
          outputValues(0, i0, 1) = 0.0;
        }
        break;
        
      case OPERATOR_DIV:
        TEUCHOS_TEST_FOR_EXCEPTION( (operatorType == OPERATOR_DIV), std::invalid_argument,
                           ">>> ERROR (Basis_HGRAD_QUAD_Cr_FEM): DIV is invalid operator for rank-0 (scalar) functions in 2D");
        break;
        
      case OPERATOR_D2:
      case OPERATOR_D3:
      case OPERATOR_D4:
      case OPERATOR_D5:
      case OPERATOR_D6:
      case OPERATOR_D7:
      case OPERATOR_D8:
      case OPERATOR_D9:
      case OPERATOR_D10:
      {
        // outputValues is a rank-3 array with dimensions (basisCardinality_, dim0, DkCardinality)
        int DkCardinality = Intrepid::getDkCardinality(operatorType, 
                                                       this -> basisCellTopology_.getDimension() );
        for(int dofOrd = 0; dofOrd < this -> basisCardinality_; dofOrd++) {
          for (int i0 = 0; i0 < dim0; i0++) {
            for(int dkOrd = 0; dkOrd < DkCardinality; dkOrd++){
              outputValues(dofOrd, i0, dkOrd) = 0.0;
            }
          }
        }
      }
        break;
        
      default:
        TEUCHOS_TEST_FOR_EXCEPTION( !( Intrepid::isValidOperator(operatorType) ), std::invalid_argument,
                           ">>> ERROR (Basis_HVOL_TRI_C0_FEM): Invalid operator type");
    }
  }
  
  
  
  template<class Scalar, class ArrayScalar>
  void Basis_HVOL_TRI_C0_FEM<Scalar, ArrayScalar>::getValues(ArrayScalar&           outputValues,
                                                               const ArrayScalar &    inputPoints,
                                                               const ArrayScalar &    cellVertices,
                                                               const EOperator        operatorType) const {
    TEUCHOS_TEST_FOR_EXCEPTION( (true), std::logic_error,
                       ">>> ERROR (Basis_HVOL_TRI_C0_FEM): FEM Basis calling an FVD member function");
  }
  
  
  
  template<class Scalar, class ArrayScalar>
  void Basis_HVOL_TRI_C0_FEM<Scalar, ArrayScalar>::getDofCoords(ArrayScalar & DofCoords) const {
#ifdef HAVE_INTREPID_DEBUG
    // Verify rank of output array.
    TEUCHOS_TEST_FOR_EXCEPTION( !(DofCoords.rank() == 2), std::invalid_argument,
                       ">>> ERROR: (Intrepid::Basis_HVOL_TRI_C0_FEM::getDofCoords) rank = 2 required for DofCoords array");
    // Verify 0th dimension of output array.
    TEUCHOS_TEST_FOR_EXCEPTION( !( DofCoords.dimension(0) == this -> basisCardinality_ ), std::invalid_argument,
                       ">>> ERROR: (Intrepid::Basis_HVOL_TRI_C0_FEM::getDofCoords) mismatch in number of DoF and 0th dimension of DofCoords array");
    // Verify 1st dimension of output array.
    TEUCHOS_TEST_FOR_EXCEPTION( !( DofCoords.dimension(1) == (int)(this -> basisCellTopology_.getDimension()) ), std::invalid_argument,
                       ">>> ERROR: (Intrepid::Basis_HVOL_TRI_C0_FEM::getDofCoords) incorrect reference cell (1st) dimension in DofCoords array");
    
    DofCoords(0,0) = 0.0;   DofCoords(0,1) = 0.0;
#endif
  }

  
}// namespace Intrepid

#endif