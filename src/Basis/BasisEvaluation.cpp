//
//  BasisEvaluation.cpp
//  DPGTrilinos
//
// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER
//

#include "TypeDefs.h"

#include <iostream>

#include "BasisEvaluation.h"

#include "BasisCache.h"
#include "BasisFactory.h"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

using namespace Intrepid;
using namespace Camellia;

void getD2_ij_FromMultiplicities(int &i, int &j, const Teuchos::Array<int> &multiplicities)
{
  bool i_assigned = false;
  for (int k=0; k<multiplicities.size(); k++)
  {
    if (multiplicities[k] == 2)
    {
      i = k;
      j = k;
      return;
    }
    else if (multiplicities[k] == 1)
    {
      if (! i_assigned)
      {
        i = k;
        i_assigned = true;
      }
      else
      {
        j = k;
        return;
      }
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i,j not identified from multiplicities...")
}

FCPtr BasisEvaluation::getValues(BasisPtr basis, Camellia::EOperator op,
                                 const FieldContainer<double> &refPoints)
{
  int numPoints = refPoints.dimension(0);
  int spaceDim = refPoints.dimension(1);  // points dimensions are (numPoints, spaceDim)
  int basisCardinality = basis->getCardinality();
  int componentOfInterest = -1;
  Camellia::EFunctionSpace fs = basis->functionSpace();
  Intrepid::EOperator relatedOp = relatedOperator(op, fs, spaceDim, componentOfInterest);

  int opCardinality = spaceDim; // for now, we assume basis values are in the same spaceDim as points (e.g. vector 1D has just 1 component)
  
  bool relatedOpIsDkOperator = (relatedOp >= OPERATOR_D1) && (relatedOp <= OPERATOR_D10);
  if (relatedOpIsDkOperator)
  {
    opCardinality = Intrepid::getDkCardinality(relatedOp, spaceDim);
  }
  
  if ((Camellia::EOperator)relatedOp != op)
  {
    // we can assume relatedResults has dimensions (numPoints,basisCardinality,spaceDimOut)
    FCPtr relatedResults = Teuchos::rcp(new FieldContainer<double>(basisCardinality,numPoints,opCardinality));
    basis->getValues(*(relatedResults.get()), refPoints, relatedOp);
    FCPtr result = getComponentOfInterest(relatedResults,op,fs,componentOfInterest);
    if ( result.get() == 0 )
    {
      result = relatedResults;
    }
    return result;
  }
  // if we get here, we should have a standard Intrepid operator, in which case we should
  // be able to: size a FieldContainer appropriately, and then call basis->getValues

  // But let's do just check that we have a standard Intrepid operator
  if ( (op >= Camellia::OP_X) || (op <  Camellia::OP_VALUE) )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown operator.");
  }

  // result dimensions should be either (numPoints,basisCardinality) or (numPoints,basisCardinality,spaceDimOut);
  Teuchos::Array<int> dimensions;
  dimensions.push_back(basisCardinality);
  dimensions.push_back(numPoints);
  int basisRank = basis->rangeRank();
  if (((basisRank == 0) && (op ==  Camellia::OP_VALUE)))
  {
    // scalar; leave as is
  }
  else if ((basisRank == 0) && relatedOpIsDkOperator)
  {
    dimensions.push_back(opCardinality);
  }
  else if ( ( ( basisRank == 1) && (op ==  Camellia::OP_VALUE) ) )
  {
    dimensions.push_back(opCardinality);
  }
  else if (
    ( ( basisRank == 0) && (op == Camellia::OP_GRAD) )
    ||
    ( ( basisRank == 0) && (op == Camellia::OP_CURL) ) )
  {
    dimensions.push_back(spaceDim);
  }
  else if ( (basis->rangeRank() == 1) && (op == Camellia::OP_GRAD) )
  {
    // grad of vector: a tensor
    dimensions.push_back(spaceDim);
    dimensions.push_back(opCardinality);
  }
  FCPtr result = Teuchos::rcp(new FieldContainer<double>(dimensions));
  basis->getValues(*(result.get()), refPoints, (Intrepid::EOperator)op);
  return result;
}

FCPtr BasisEvaluation::getTransformedValues(BasisPtr basis, Camellia::EOperator op,
    const FieldContainer<double> &referencePoints,
    int numCells,
    const FieldContainer<double> &cellJacobian,
    const FieldContainer<double> &cellJacobianInv,
    const FieldContainer<double> &cellJacobianDet)
{
  Camellia::EFunctionSpace fs = basis->functionSpace();
  int spaceDim = referencePoints.dimension(1);
  int componentOfInterest;
  Intrepid::EOperator relatedOp;
  relatedOp = relatedOperator(op, fs, spaceDim, componentOfInterest);

  FCPtr referenceValues = getValues(basis,(Camellia::EOperator) relatedOp, referencePoints);
  return getTransformedValuesWithBasisValues(basis,op,spaceDim,referenceValues,
                                             numCells,cellJacobian,cellJacobianInv,cellJacobianDet);
}

FCPtr BasisEvaluation::getTransformedValues(BasisPtr basis, Camellia::EOperator op,
                                            const FieldContainer<double> &referencePoints,
                                            int numCells,
                                            BasisCache* basisCache)
{
  Camellia::EFunctionSpace fs = basis->functionSpace();
  int spaceDim = referencePoints.dimension(1);
  int componentOfInterest;
  Intrepid::EOperator relatedOp;
  relatedOp = relatedOperator(op, fs, spaceDim, componentOfInterest);
  
  FCPtr referenceValues = getValues(basis,(Camellia::EOperator) relatedOp, referencePoints);
  return getTransformedValuesWithBasisValues(basis,op,referenceValues,numCells,basisCache);
}

FCPtr BasisEvaluation::getTransformedVectorValuesWithComponentBasisValues(VectorBasisPtr basis, Camellia::EOperator op,
    constFCPtr componentReferenceValuesTransformed)
{
  Camellia::EFunctionSpace fs = basis->functionSpace();
  bool vectorizedBasis = Camellia::functionSpaceIsVectorized(fs);
  if ( !vectorizedBasis ||
       ((op !=  Camellia::OP_VALUE) && (op != Camellia::OP_CROSS_NORMAL) ))
  {
    TEUCHOS_TEST_FOR_EXCEPTION( !vectorizedBasis || (op !=  Camellia::OP_VALUE),
                                std::invalid_argument, "Only Vector HGRAD/HVOL with OP_VALUE supported by getTransformedVectorValuesWithComponentBasisValues.  Please use getTransformedValuesWithBasisValues instead.");
  }
  BasisPtr componentBasis = basis->getComponentBasis();
  Teuchos::Array<int> dimensions;
  componentReferenceValuesTransformed->dimensions(dimensions);
  dimensions[1] = basis->getCardinality(); // dimensions are (C,F,P,D)
  dimensions.push_back(basis->getNumComponents());
  Teuchos::RCP<FieldContainer<double> > transformedValues = Teuchos::rcp(new FieldContainer<double>(dimensions));
  int fieldIndex = 1;
  basis->getVectorizedValues(*transformedValues, *componentReferenceValuesTransformed,fieldIndex);
  return transformedValues;
}

FCPtr BasisEvaluation::getTransformedValuesWithBasisValues(BasisPtr basis, Camellia::EOperator op, int spaceDim,
                                                           constFCPtr referenceValues, int numCells,
                                                           const FieldContainer<double> &cellJacobian,
                                                           const FieldContainer<double> &cellJacobianInv,
                                                           const FieldContainer<double> &cellJacobianDet)
{
  typedef FunctionSpaceTools fst;
//  int numCells = cellJacobian.dimension(0);
  
//  int spaceDim = basis->domainTopology()->getDimension(); // changed 2/18/15
//  // 6-16-14 NVR: getting the spaceDim from cellJacobian's dimensioning is the way we've historically done it.
//  // I think it might be better to do this using basis->domainTopology() generally, but for now we only make the
//  // switch in case the domain topology is a Node.
//  if (basis->domainTopology()->getDimension() == 0) {
//    spaceDim = 0;
//  } else {
//    spaceDim = cellJacobian.dimension(2);
//  }

  int componentOfInterest;
  Camellia::EFunctionSpace fs = basis->functionSpace();
  Intrepid::EOperator relatedOp = relatedOperator(op,fs,spaceDim, componentOfInterest);
  Teuchos::Array<int> dimensions;
  referenceValues->dimensions(dimensions);
  dimensions.insert(dimensions.begin(), numCells);
  Teuchos::RCP<FieldContainer<double> > transformedValues = Teuchos::rcp(new FieldContainer<double>(dimensions));
  bool vectorizedBasis = functionSpaceIsVectorized(fs);
  if (vectorizedBasis && (op ==  Camellia::OP_VALUE))
  {
    TEUCHOS_TEST_FOR_EXCEPTION( vectorizedBasis && (op ==  Camellia::OP_VALUE),
                                std::invalid_argument, "Vector HGRAD/HVOL with OP_VALUE not supported by getTransformedValuesWithBasisValues.  Please use getTransformedVectorValuesWithComponentBasisValues instead.");
  }
  switch(relatedOp)
  {
  case(Intrepid::OPERATOR_VALUE):
    switch(fs)
    {
    case Camellia::FUNCTION_SPACE_REAL_SCALAR:
    //          cout << "Reference values for FUNCTION_SPACE_REAL_SCALAR: " << *referenceValues;
    case Camellia::FUNCTION_SPACE_HGRAD:
    case Camellia::FUNCTION_SPACE_HGRAD_DISC:
      fst::HGRADtransformVALUE<double>(*transformedValues,*referenceValues);
      break;
    case Camellia::FUNCTION_SPACE_HCURL:
    case Camellia::FUNCTION_SPACE_HCURL_DISC:
      fst::HCURLtransformVALUE<double>(*transformedValues,cellJacobianInv,*referenceValues);
      break;
    case Camellia::FUNCTION_SPACE_HDIV:
    case Camellia::FUNCTION_SPACE_HDIV_DISC:
    case Camellia::FUNCTION_SPACE_HDIV_FREE:
      fst::HDIVtransformVALUE<double>(*transformedValues,cellJacobian,cellJacobianDet,*referenceValues);
        break;
      case Camellia::FUNCTION_SPACE_HVOL:
      case Camellia::FUNCTION_SPACE_HVOL_SPACE_HGRAD_TIME:
      case Camellia::FUNCTION_SPACE_HGRAD_SPACE_HVOL_TIME:
//        {
//          static bool haveWarned = false;
//          if (!haveWarned) {
//            cout << "WARNING: for the moment, switching to the standard HVOLtransformVALUE method.\n";
//            haveWarned = true;
//          }
//        }
//          fst::HVOLtransformVALUE<double>(*transformedValues, cellJacobianDet, *referenceValues);
      // for the moment, use the fact that we know the HVOL basis is always an HGRAD basis:
      // (I think using the below amounts to solving for the HVOL variables scaled by Jacobian)
      fst::HGRADtransformVALUE<double>(*transformedValues,*referenceValues);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
      break;
    }
    break;
  case(Intrepid::OPERATOR_GRAD):
    switch(fs)
    {
    case Camellia::FUNCTION_SPACE_HVOL:
    case Camellia::FUNCTION_SPACE_HGRAD:
    case Camellia::FUNCTION_SPACE_HGRAD_DISC:
      fst::HGRADtransformGRAD<double>(*transformedValues,cellJacobianInv,*referenceValues);
      break;
    case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
    case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
    case Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC:
      // referenceValues has dimensions (F,P,D1,D2).  D1 is our component dimension, and D2 is the one that came from the gradient.
      // HGRADtransformGRAD expects (F,P,D) for input, and (C,F,P,D) for output.
      // If we split referenceValues into (F,P,D1=0,D2) and (F,P,D1=1,D2), then we can transform each of those, and then interleave the results…
    {
      // block off so we can create new stuff inside the switch case
      Teuchos::Array<int> dimensions;
      referenceValues->dimensions(dimensions);
      int numFields = dimensions[0];
      int numPoints = dimensions[1];
      int D1 = dimensions[dimensions.size()-2];
      int D2 = dimensions[dimensions.size()-1];
      dimensions[dimensions.size()-2] = D2; // put D2 in the D1 spot
      dimensions.pop_back(); // get rid of original D2
      FieldContainer<double> refValuesSlice(dimensions);
      dimensions.insert(dimensions.begin(),numCells);
      FieldContainer<double> transformedValuesSlice(dimensions);

//          int numEntriesPerSlice = refValuesSlice.size();
//          int numEntriesPerTransformedSlice = transformedValuesSlice.size();

      for (int compIndex1=0; compIndex1<D1; compIndex1++)
      {
        // could speed the following along by doing the enumeration arithmetic in place...
        for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++)
        {
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
          {
            for (int compIndex2=0; compIndex2<D2; compIndex2++)
            {
              refValuesSlice(fieldIndex,ptIndex,compIndex2) = (*referenceValues)(fieldIndex,ptIndex,compIndex1,compIndex2);
            }
          }
        }

//            for (int i=0; i<numEntriesPerSlice; i++) {
//              refValuesSlice[i] = (*referenceValues)[i*D2 + compIndex];
//            }
        fst::HGRADtransformGRAD<double>(transformedValuesSlice,cellJacobianInv,refValuesSlice);
        // could speed the following along by doing the enumeration arithmetic in place...
        for (int cellIndex=0; cellIndex<numCells; cellIndex++)
        {
          for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++)
          {
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
            {
              for (int compIndex2=0; compIndex2<D2; compIndex2++)
              {
                (*transformedValues)(cellIndex,fieldIndex,ptIndex,compIndex1,compIndex2) = transformedValuesSlice(cellIndex,fieldIndex,ptIndex,compIndex2);
              }
            }
          }
        }
//            for (int i=0; i<numEntriesPerTransformedSlice; i++) {
//              (*transformedValues)[i*D2 + compIndex] = transformedValuesSlice[i];
//            }
      }
    }

    break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
      break;
    }
    break;
  case(Intrepid::OPERATOR_CURL):
    switch(fs)
    {
    case Camellia::FUNCTION_SPACE_HCURL:
    case Camellia::FUNCTION_SPACE_HCURL_DISC:
      if (spaceDim == 2)
      {
        // TODO: confirm that this is correct
//            static bool warningIssued = false;
//            if ( ! warningIssued ) {
//              cout << "WARNING: for HCURL in 2D, transforming with HVOLtransformVALUE. Need to confirm this is correct.\n";
//              warningIssued = true;
//            }
        fst::HVOLtransformVALUE<double>(*transformedValues, cellJacobianDet, *referenceValues);
      }
      else
      {
        fst::HCURLtransformCURL<double>(*transformedValues,cellJacobian,cellJacobianDet,*referenceValues);
      }
      break;
    case Camellia::FUNCTION_SPACE_HGRAD:
    case Camellia::FUNCTION_SPACE_HGRAD_DISC:
      // in 2D, HGRADtransformCURL == HDIVtransformVALUE (because curl(H1) --> H(div))
      fst::HDIVtransformVALUE<double>(*transformedValues,cellJacobian,cellJacobianDet,*referenceValues);
      break;
    case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
    case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
    case Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC: // shouldn't take the transform so late
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
      break;
    }
    break;
  case(Intrepid::OPERATOR_DIV):
    switch(fs)
    {
    case Camellia::FUNCTION_SPACE_HDIV:
    case Camellia::FUNCTION_SPACE_HDIV_DISC:
    case Camellia::FUNCTION_SPACE_HDIV_FREE:
      fst::HDIVtransformDIV<double>(*transformedValues,cellJacobianDet,*referenceValues);
      break;
    case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
    case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
    case Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC: // shouldn't take the transform so late
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
      break;
    }
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
    break;
  }

  FCPtr result = getComponentOfInterest(transformedValues,op,fs,componentOfInterest);
  if ( result.get() == 0 )
  {
    result = transformedValues;
  }
  return result;
}

FCPtr BasisEvaluation::getTransformedValuesWithBasisValues(BasisPtr basis, Camellia::EOperator op,
                                                           constFCPtr referenceValues, int numCells,
                                                           BasisCache* basisCache)
{
  typedef FunctionSpaceTools fst;
  //  int numCells = cellJacobian.dimension(0);
  
  int spaceDim = basisCache->getSpaceDim(); // changed 3-21-16
  
  int componentOfInterest;
  Camellia::EFunctionSpace fs = basis->functionSpace();
  Intrepid::EOperator relatedOp = relatedOperator(op,fs,spaceDim, componentOfInterest);
  Teuchos::Array<int> dimensions;
  referenceValues->dimensions(dimensions);
  dimensions.insert(dimensions.begin(), numCells);
  Teuchos::RCP<FieldContainer<double> > transformedValues = Teuchos::rcp(new FieldContainer<double>(dimensions));
  bool vectorizedBasis = functionSpaceIsVectorized(fs);
  if (vectorizedBasis && (op ==  Camellia::OP_VALUE))
  {
    TEUCHOS_TEST_FOR_EXCEPTION( vectorizedBasis && (op ==  Camellia::OP_VALUE),
                               std::invalid_argument, "Vector HGRAD/HVOL with OP_VALUE not supported by getTransformedValuesWithBasisValues.  Please use getTransformedVectorValuesWithComponentBasisValues instead.");
  }
  switch(relatedOp)
  {
    case(Intrepid::OPERATOR_VALUE):
      switch(fs)
    {
      case Camellia::FUNCTION_SPACE_REAL_SCALAR:
        //          cout << "Reference values for FUNCTION_SPACE_REAL_SCALAR: " << *referenceValues;
      case Camellia::FUNCTION_SPACE_HGRAD:
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
        fst::HGRADtransformVALUE<double>(*transformedValues,*referenceValues);
        break;
      case Camellia::FUNCTION_SPACE_HCURL:
      case Camellia::FUNCTION_SPACE_HCURL_DISC:
        fst::HCURLtransformVALUE<double>(*transformedValues,basisCache->getJacobianInv(),*referenceValues);
        break;
      case Camellia::FUNCTION_SPACE_HDIV:
      case Camellia::FUNCTION_SPACE_HDIV_DISC:
      case Camellia::FUNCTION_SPACE_HDIV_FREE:
        fst::HDIVtransformVALUE<double>(*transformedValues,basisCache->getJacobian(),basisCache->getJacobianDet(),*referenceValues);
        break;
      case Camellia::FUNCTION_SPACE_HVOL:
      case Camellia::FUNCTION_SPACE_HVOL_SPACE_HGRAD_TIME:
      case Camellia::FUNCTION_SPACE_HGRAD_SPACE_HVOL_TIME:
        //        {
        //          static bool haveWarned = false;
        //          if (!haveWarned) {
        //            cout << "WARNING: for the moment, switching to the standard HVOLtransformVALUE method.\n";
        //            haveWarned = true;
        //          }
        //        }
        //          fst::HVOLtransformVALUE<double>(*transformedValues, cellJacobianDet, *referenceValues);
        // for the moment, use the fact that we know the HVOL basis is always an HGRAD basis:
        // (I think using the below amounts to solving for the HVOL variables scaled by Jacobian)
        fst::HGRADtransformVALUE<double>(*transformedValues,*referenceValues);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
        break;
    }
      break;
    case(Intrepid::OPERATOR_GRAD):
    case(Intrepid::OPERATOR_D1):
      switch(fs)
    {
      case Camellia::FUNCTION_SPACE_HVOL:
      case Camellia::FUNCTION_SPACE_HGRAD:
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
        fst::HGRADtransformGRAD<double>(*transformedValues,basisCache->getJacobianInv(),*referenceValues);
        break;
      case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
      case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
      case Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC:
        // referenceValues has dimensions (F,P,D1,D2).  D1 is our component dimension, and D2 is the one that came from the gradient.
        // HGRADtransformGRAD expects (F,P,D) for input, and (C,F,P,D) for output.
        // If we split referenceValues into (F,P,D1=0,D2) and (F,P,D1=1,D2), then we can transform each of those, and then interleave the results…
      {
        // block off so we can create new stuff inside the switch case
        Teuchos::Array<int> dimensions;
        referenceValues->dimensions(dimensions);
        int numFields = dimensions[0];
        int numPoints = dimensions[1];
        int D1 = dimensions[dimensions.size()-2];
        int D2 = dimensions[dimensions.size()-1];
        dimensions[dimensions.size()-2] = D2; // put D2 in the D1 spot
        dimensions.pop_back(); // get rid of original D2
        FieldContainer<double> refValuesSlice(dimensions);
        dimensions.insert(dimensions.begin(),numCells);
        FieldContainer<double> transformedValuesSlice(dimensions);
        
        //          int numEntriesPerSlice = refValuesSlice.size();
        //          int numEntriesPerTransformedSlice = transformedValuesSlice.size();
        
        for (int compIndex1=0; compIndex1<D1; compIndex1++)
        {
          // could speed the following along by doing the enumeration arithmetic in place...
          for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++)
          {
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
            {
              for (int compIndex2=0; compIndex2<D2; compIndex2++)
              {
                refValuesSlice(fieldIndex,ptIndex,compIndex2) = (*referenceValues)(fieldIndex,ptIndex,compIndex1,compIndex2);
              }
            }
          }
          
          //            for (int i=0; i<numEntriesPerSlice; i++) {
          //              refValuesSlice[i] = (*referenceValues)[i*D2 + compIndex];
          //            }
          fst::HGRADtransformGRAD<double>(transformedValuesSlice,basisCache->getJacobianInv(),refValuesSlice);
          // could speed the following along by doing the enumeration arithmetic in place...
          for (int cellIndex=0; cellIndex<numCells; cellIndex++)
          {
            for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++)
            {
              for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
              {
                for (int compIndex2=0; compIndex2<D2; compIndex2++)
                {
                  (*transformedValues)(cellIndex,fieldIndex,ptIndex,compIndex1,compIndex2) = transformedValuesSlice(cellIndex,fieldIndex,ptIndex,compIndex2);
                }
              }
            }
          }
          //            for (int i=0; i<numEntriesPerTransformedSlice; i++) {
          //              (*transformedValues)[i*D2 + compIndex] = transformedValuesSlice[i];
          //            }
        }
      }
        
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
        break;
    }
      break;
    case(Intrepid::OPERATOR_CURL):
      switch(fs)
    {
      case Camellia::FUNCTION_SPACE_HCURL:
      case Camellia::FUNCTION_SPACE_HCURL_DISC:
        if (spaceDim == 2)
        {
          // TODO: confirm that this is correct
          //            static bool warningIssued = false;
          //            if ( ! warningIssued ) {
          //              cout << "WARNING: for HCURL in 2D, transforming with HVOLtransformVALUE. Need to confirm this is correct.\n";
          //              warningIssued = true;
          //            }
          fst::HVOLtransformVALUE<double>(*transformedValues,basisCache->getJacobianDet(), *referenceValues);
        }
        else
        {
          fst::HCURLtransformCURL<double>(*transformedValues,basisCache->getJacobian(),basisCache->getJacobianDet(),*referenceValues);
        }
        break;
      case Camellia::FUNCTION_SPACE_HGRAD:
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
        // in 2D, HGRADtransformCURL == HDIVtransformVALUE (because curl(H1) --> H(div))
        fst::HDIVtransformVALUE<double>(*transformedValues,basisCache->getJacobian(),basisCache->getJacobianDet(),*referenceValues);
        break;
      case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
      case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
      case Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC: // shouldn't take the transform so late
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
        break;
    }
      break;
    case(Intrepid::OPERATOR_DIV):
      switch(fs)
    {
      case Camellia::FUNCTION_SPACE_HDIV:
      case Camellia::FUNCTION_SPACE_HDIV_DISC:
      case Camellia::FUNCTION_SPACE_HDIV_FREE:
        fst::HDIVtransformDIV<double>(*transformedValues,basisCache->getJacobianDet(),*referenceValues);
        break;
      case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
      case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
      case Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC: // shouldn't take the transform so late
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
        break;
    }
      break;
    case(Intrepid::OPERATOR_D2):
      switch(fs)
    {
      case Camellia::FUNCTION_SPACE_HVOL:
      case Camellia::FUNCTION_SPACE_HGRAD:
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
      {
        // first term in the sum is:
        //      J^{-1} * OPD2 * J^{-T}
        // where J^{-1} is the inverse Jacabian, and OPD2 is the matrix of reference-space values formed from OP_D2
        
        const FieldContainer<double> *J_inv = &basisCache->getJacobianInv();
        transformedValues->initialize(0.0);
        int basisCardinality = basis->getCardinality();
        
        Teuchos::Array<int> multiplicities(spaceDim);
        Teuchos::Array<int> multiplicitiesTransformed(spaceDim);
        int dkCardinality = Intrepid::getDkCardinality(Intrepid::OPERATOR_D2, spaceDim);
        int numPoints = referenceValues->dimension(1);
        
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          for (int fieldOrdinal=0; fieldOrdinal<basisCardinality; fieldOrdinal++)
          {
            for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
            {
              for (int dkOrdinal=0; dkOrdinal<dkCardinality; dkOrdinal++) // ref values dkOrdinal
              {
                double refValue = (*referenceValues)(fieldOrdinal,pointOrdinal,dkOrdinal);
                Intrepid::getDkMultiplicities(multiplicities, dkOrdinal, Intrepid::OPERATOR_D2, spaceDim);
                int k,l; // ref space coordinate directions for refValue (columns in J_inv)
                getD2_ij_FromMultiplicities(k,l,multiplicities);
                
                for (int dkOrdinalTransformed=0; dkOrdinalTransformed<dkCardinality; dkOrdinalTransformed++) // physical values dkOrdinal
                {
                  Intrepid::getDkMultiplicities(multiplicitiesTransformed, dkOrdinalTransformed, Intrepid::OPERATOR_D2, spaceDim);
                  int i,j; // physical space coordinate directions for refValue (rows in J_inv)
                  getD2_ij_FromMultiplicities(i,j,multiplicitiesTransformed);
                  double J_inv_ik = (*J_inv)(cellOrdinal,pointOrdinal,i,k);
                  double J_inv_jl = (*J_inv)(cellOrdinal,pointOrdinal,j,l);
                  (*transformedValues)(cellOrdinal,fieldOrdinal,pointOrdinal,dkOrdinalTransformed) += refValue * J_inv_ik * J_inv_jl;
                }
              }
            }
          }
        }
        
        // do we need the second term in the sum?  (So far, we don't support this)
        if (!basisCache->neglectHessian())
        {
          // then we need to do include the gradient of the basis times the (inverse) of the Hessian
          // this seems complicated, and we don't need it for computing the Laplacian in physical space
          // (at least not unless we have curvilinear geometry) so we don't support it for now...
          TEUCHOS_TEST_FOR_EXCEPTION(!basisCache->neglectHessian(), std::invalid_argument, "Support for the Hessian of the reference-to-physical mapping not yet implemented.");
        }
      }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
        break;
    }
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
      break;
  }
  
  FCPtr result = getComponentOfInterest(transformedValues,op,fs,componentOfInterest);
  if ( result.get() == 0 )
  {
    result = transformedValues;
  }
  return result;
}

Intrepid::EOperator BasisEvaluation::relatedOperator(Camellia::EOperator op, Camellia::EFunctionSpace fs,
                                                     int spaceDim, int &componentOfInterest)
{
  Intrepid::EOperator relatedOp = (Intrepid::EOperator) op;
  componentOfInterest = -1;

  bool vectorizedBasis = Camellia::functionSpaceIsVectorized(fs);

  if (   vectorizedBasis
         && ( (op == Camellia::OP_CURL) || (op == Camellia::OP_DIV) ) )
  {
    relatedOp = Intrepid::OPERATOR_GRAD;
  }
  else if ((op==OP_X) || (op==OP_Y) || (op==OP_Z))
  {
    componentOfInterest = op - OP_X;
    relatedOp = Intrepid::OPERATOR_VALUE;
  }
  else if ((op==OP_DX) || (op==OP_DY) || (op==OP_DZ))
  {
    componentOfInterest = op - OP_DX;
    relatedOp = Intrepid::OPERATOR_GRAD;
  }
  else if (   (op==OP_CROSS_NORMAL)   || (op==OP_DOT_NORMAL)     || (op==OP_TIMES_NORMAL)
              || (op==OP_TIMES_NORMAL_X) || (op==OP_TIMES_NORMAL_Y) || (op==OP_TIMES_NORMAL_Z) || (op==OP_TIMES_NORMAL_T)
              || (op==OP_VECTORIZE_VALUE))
  {
    relatedOp = Intrepid::OPERATOR_VALUE;
  }
  else if (op==OP_LAPLACIAN)
  {
    relatedOp = Intrepid::OPERATOR_D2;
    componentOfInterest = -1; // this is a special case; we shouldn't compute Laplacians in reference space.  Instead keep the whole operator D2 result; in general, we will also want the gradient in reference space to compute the transformed guy (if the second derivatives of the reference-to-physical transformation are nonzero).
  }
  else if ((op==OP_DXDX) || (op==OP_DYDY) || (op==OP_DZDZ))
  {
    relatedOp = Intrepid::OPERATOR_D2;
    if (spaceDim == 1)
    {
      if (op==OP_DXDX)
      {
        componentOfInterest = Intrepid::getDkEnumeration(2);
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "op incompatible with spaceDim");
      }
    }
    else if (spaceDim == 2)
    {
      if (op==OP_DXDX)
      {
        componentOfInterest = Intrepid::getDkEnumeration(2,0);
      }
      else if (op==OP_DYDY)
      {
        componentOfInterest = Intrepid::getDkEnumeration(0,2);
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "op incompatible with spaceDim");
      }
    }
    else if (spaceDim == 3)
    {
      if (op==OP_DXDX)
      {
        componentOfInterest = Intrepid::getDkEnumeration(2,0,0);
      }
      else if (op==OP_DYDY)
      {
        componentOfInterest = Intrepid::getDkEnumeration(0,2,0);
      }
      else if (op==OP_DZDZ)
      {
        componentOfInterest = Intrepid::getDkEnumeration(0,0,2);
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "op incompatible with spaceDim");
      }
    }
  }
  return relatedOp;
}

FCPtr BasisEvaluation::getComponentOfInterest(constFCPtr values, Camellia::EOperator op, Camellia::EFunctionSpace fs, int componentOfInterest)
{
  FCPtr result;
  bool vectorizedBasis = functionSpaceIsVectorized(fs);
  if (   vectorizedBasis
         && ( (op == Camellia::OP_CURL) || (op == Camellia::OP_DIV)) )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(values->rank() != 5, std::invalid_argument, "rank of values must be 5 for VECTOR_HGRAD_GRAD");
    int numCells  = values->dimension(0);
    int numFields = values->dimension(1);
    int numPoints = values->dimension(2);
    result = Teuchos::rcp(new FieldContainer<double>(numCells,numFields,numPoints));
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int field=0; field<numFields; field++)
      {
        for (int point=0; point<numPoints; point++)
        {
          if (op==Camellia::OP_DIV)
          {
            (*result)(cellIndex,field,point) = (*values)(cellIndex,field,point,0,0) + (*values)(cellIndex,field,point,1,1);
          }
          else if (op==Camellia::OP_CURL)
          {
            // curl of 2D vector: d(Fx)/dy - d(Fy)/dx
            (*result)(cellIndex,field,point) = (*values)(cellIndex,field,point,1,0) - (*values)(cellIndex,field,point,0,1);
          }
        }
      }
    }
    return result;
  }
  else if ((componentOfInterest < 0) && (op != Camellia::OP_LAPLACIAN))     // then just return values
  {
    // the copy is a bit unfortunate, but can't be avoided unless we change a bunch of constFCPtrs to FCPtrs (or vice versa)
    // in the API...
//    cout << "values:\n" << *values;
    return Teuchos::rcp( new FieldContainer<double>(*values));
  }
  Teuchos::Array<int> dimensions;
  values->dimensions(dimensions);
  int opCardinality = dimensions[dimensions.size()-1];
  dimensions.pop_back(); // get rid of last, spatial dimension
  result = Teuchos::rcp(new FieldContainer<double>(dimensions));
  
  if (op == Camellia::OP_LAPLACIAN)
  {
    // back out spaceDim from opCardinality
    int spaceDim = -1;
    if (opCardinality == 1) spaceDim = 1;
    if (opCardinality == 3) spaceDim = 2;
    if (opCardinality == 6) spaceDim = 3;
//    if (opCardinality == 10) spaceDim = 4;
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim == -1, std::invalid_argument, "unhandled opCardinality and/or spaceDim");
    vector<int> dkEnumeration(spaceDim);
    if (spaceDim == 1)
    {
      dkEnumeration[0] = Intrepid::getDkEnumeration(2);
    }
    else if (spaceDim == 2)
    {
      dkEnumeration[0] = Intrepid::getDkEnumeration(2,0);
      dkEnumeration[1] = Intrepid::getDkEnumeration(0,2);
    }
    else if (spaceDim == 3)
    {
      dkEnumeration[0] = Intrepid::getDkEnumeration(2,0,0);
      dkEnumeration[1] = Intrepid::getDkEnumeration(0,2,0);
      dkEnumeration[2] = Intrepid::getDkEnumeration(0,0,2);
    }
    for (int comp : dkEnumeration)
    {
      int size = result->size();
      int enumeratedLocation;
      if (values->rank() == 3) // (F,P,D)
      {
        enumeratedLocation = values->getEnumeration(0,0,comp);
      }
      else if (values->rank() == 4) // (C,F,P,D)
      {
        enumeratedLocation = values->getEnumeration(0,0,0,comp);
      }
      else
      {
        // TODO: consider computing the enumerated location in a rank-independent way.
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported values container rank.");
      }
      for (int i=0; i<size; i++)
      {
        (*result)[i] += (*values)[enumeratedLocation];
        enumeratedLocation += opCardinality;
      }
    }
    return result;
  }
  
  TEUCHOS_TEST_FOR_EXCEPTION(componentOfInterest >= opCardinality, std::invalid_argument, "componentOfInterest is out of bounds!");
//  int numPoints = dimensions[0];
//  int basisCardinality = dimensions[1];
  int size = result->size();
  int enumeratedLocation;
  if (values->rank() == 3)
  {
    enumeratedLocation = values->getEnumeration(0,0,componentOfInterest);
  }
  else if (values->rank() == 4)
  {
    enumeratedLocation = values->getEnumeration(0,0,0,componentOfInterest);
  }
  else if (values->rank() == 5)
  {
    enumeratedLocation = values->getEnumeration(0,0,0,0,componentOfInterest);
  }
  else
  {
    // TODO: consider computing the enumerated location in a rank-independent way.
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported values container rank.");
  }
  for (int i=0; i<size; i++)
  {
    (*result)[i] = (*values)[enumeratedLocation];
    enumeratedLocation += opCardinality;
  }
  return result;
}

FCPtr BasisEvaluation::getValuesCrossedWithNormals(constFCPtr values,const FieldContainer<double> &sideNormals)
{
  // values should have dimensions (C,basisCardinality,P,D)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);

  if (spaceDim != 2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "crossing with normal only supported for 2D right now");
  }
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
       || (numPoints != values->dimension(2)) || (spaceDim != values->dimension(3))
     )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P,D)");
  }

  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
      {
        double n1 = sideNormals(cellIndex,pointIndex,0);
        double n2 = sideNormals(cellIndex,pointIndex,1);
        double xValue = (*values)(cellIndex,basisOrdinal,pointIndex,0);
        double yValue = (*values)(cellIndex,basisOrdinal,pointIndex,1);
        /*cout << "(n1,n2) = (" << n1 << ", " << n2 << ")" << endl;
        cout << "(x,y) = (" << xValue << ", " << yValue << ")" << endl;
        cout << "(x,y) x n = " << xValue*n2 - yValue*n1 << endl;*/
//        cout << "WARNING: pretty sure that we've reversed the sign of OP_CROSS_NORMAL.\n";
        (*result)(cellIndex,basisOrdinal,pointIndex) = yValue*n1 - xValue*n2;
      }
    }
  }
  return result;
}

FCPtr BasisEvaluation::getValuesDottedWithNormals(constFCPtr values,const FieldContainer<double> &sideNormals)
{
  // values should have dimensions (C,basisCardinality,P,D)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);

  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
       || (numPoints != values->dimension(2)) || (spaceDim != values->dimension(3))
     )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P,D)");
  }

  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
      {
        (*result)(cellIndex,basisOrdinal,pointIndex) = 0;
        for (int d=0; d<spaceDim; d++)
        {
          double nd = sideNormals(cellIndex,pointIndex,d);
          double dValue = (*values)(cellIndex,basisOrdinal,pointIndex,d);
          (*result)(cellIndex,basisOrdinal,pointIndex) += nd * dValue;
        }
      }
    }
  }
  return result;
}

FCPtr BasisEvaluation::getValuesTimesNormals(constFCPtr values,const FieldContainer<double> &sideNormals, int normalComponent)
{
  // TODO: write tests against this version of getValuesTimesNormals
  // values should have dimensions (C,basisCardinality,P)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
//  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);

  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
       || (numPoints != values->dimension(2)))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
  }

  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
      {
        double n_i = sideNormals(cellIndex,pointIndex,normalComponent);
        double value = (*values)(cellIndex,basisOrdinal,pointIndex);
        (*result)(cellIndex,basisOrdinal,pointIndex) = value * n_i;
      }
    }
  }
  return result;
}

FCPtr BasisEvaluation::getValuesTimesNormals(constFCPtr values,const FieldContainer<double> &sideNormals)
{
  // values should have dimensions (C,basisCardinality,P)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);

  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
       || (numPoints != values->dimension(2)))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
  }

  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints,spaceDim));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
      {
        for (int dim=0; dim<spaceDim; dim++)
        {
          double n_dim = sideNormals(cellIndex,pointIndex,dim);
          double value = (*values)(cellIndex,basisOrdinal,pointIndex);
          (*result)(cellIndex,basisOrdinal,pointIndex,dim) = value * n_dim;
        }
      }
    }
  }
  return result;
}

FCPtr BasisEvaluation::getVectorizedValues(constFCPtr values, int spaceDim)
{
  // values should have dimensions (C,basisCardinality,P)
  int numCells = values->dimension(0);
  int basisCardinality = values->dimension(1);
  int numPoints = values->dimension(2);

  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1)) || (numPoints != values->dimension(2)))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
  }

  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints,spaceDim));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
      {
        for (int dim=0; dim<spaceDim; dim++)
        {
          (*result)(cellIndex,basisOrdinal,pointIndex,dim) = (*values)(cellIndex,basisOrdinal,pointIndex);
        }
      }
    }
  }
  return result;
}
