//
//  TIP.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Intrepid_FunctionSpaceTools.hpp"

#include "IP.h"
#include "SerialDenseMatrixUtility.h"
#include "SerialDenseWrapper.h"
#include "VarFactory.h"
#include "BasisCache.h"
#include "CellTopology.h"

using namespace Intrepid;
using namespace Camellia;

template <typename Scalar>
TIP<Scalar>::TIP()
{
  _isLegacySubclass = false;
}
// if the terms are a1, a2, ..., then the inner product is (a1,a1) + (a2,a2) + ...

template <typename Scalar>
TIP<Scalar>::TIP(TBFPtr<Scalar> bfs)
{
  _bilinearForm = bfs;
  _isLegacySubclass = true;
}

// added by Nate
template <typename Scalar>
TLinearTermPtr<Scalar> TIP<Scalar>::evaluate(const map< int, TFunctionPtr<Scalar>> &varFunctions)
{
  // include both the boundary and non-boundary parts
  return evaluate(varFunctions,true) + evaluate(varFunctions,false);
}

// added by Jesse - evaluate inner product at given varFunctions
template <typename Scalar>
TLinearTermPtr<Scalar> TIP<Scalar>::evaluate(const map< int, TFunctionPtr<Scalar>> &varFunctions, bool boundaryPart)
{
  TLinearTermPtr<Scalar> ltEval = Teuchos::rcp(new LinearTerm);
  for (typename vector< TLinearTermPtr<Scalar> >:: const_iterator ltIt = _linearTerms.begin(); ltIt != _linearTerms.end(); ltIt++)
  {
    TLinearTermPtr<Scalar> lt = *ltIt;
    TFunctionPtr<Scalar> weight = lt->evaluate(varFunctions,boundaryPart);
    ltEval->addTerm(weight*lt);
  }
  return ltEval;
}

template <typename Scalar>
void TIP<Scalar>::addTerm( TLinearTermPtr<Scalar> a )
{
  _linearTerms.push_back(a);
}

template <typename Scalar>
void TIP<Scalar>::addTerm( VarPtr v )
{
  _linearTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

template <typename Scalar>
void TIP<Scalar>::addZeroMeanTerm( TLinearTermPtr<Scalar> a)
{
  _zeroMeanTerms.push_back(a);
}

template <typename Scalar>
void TIP<Scalar>::addZeroMeanTerm( VarPtr v )
{
  _zeroMeanTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

template <typename Scalar>
void TIP<Scalar>::addBoundaryTerm( TLinearTermPtr<Scalar> a )
{
  _boundaryTerms.push_back(a);
}

template <typename Scalar>
void TIP<Scalar>::addBoundaryTerm( VarPtr v )
{
  _boundaryTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

template <typename Scalar>
void TIP<Scalar>::applyInnerProductData(FieldContainer<Scalar> &testValues1,
                                        FieldContainer<Scalar> &testValues2,
                                        int testID1, int testID2, int operatorIndex,
                                        Teuchos::RCP<BasisCache> basisCache)
{
  applyInnerProductData(testValues1, testValues2, testID1, testID2, operatorIndex, basisCache->getPhysicalCubaturePoints());
}

template <typename Scalar>
void TIP<Scalar>::applyInnerProductData(FieldContainer<Scalar> &testValues1,
                                        FieldContainer<Scalar> &testValues2,
                                        int testID1, int testID2, int operatorIndex,
                                        const FieldContainer<double>& physicalPoints)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override some version of applyInnerProductData!");
}

template <typename Scalar>
void TIP<Scalar>::computeInnerProductMatrix(FieldContainer<Scalar> &innerProduct,
    Teuchos::RCP<DofOrdering> dofOrdering, shards::CellTopology &cellTopo,
    FieldContainer<double>& physicalCellNodes)
{
  if (_isLegacySubclass)
  {
    CellTopoPtr cellTopoPtr = CellTopology::cellTopology(cellTopo);
    Teuchos::RCP<ElementType> elemTypePtr = Teuchos::rcp( new ElementType(dofOrdering,dofOrdering, cellTopoPtr) );
    Teuchos::RCP<Mesh> nullMeshPtr = Teuchos::rcp( (Mesh*) NULL );
    BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr, nullMeshPtr,true));
    ipBasisCache->setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(), false);
    computeInnerProductMatrix(innerProduct,dofOrdering,ipBasisCache);
  }
}

template <typename Scalar>
void TIP<Scalar>::computeInnerProductMatrix(FieldContainer<Scalar> &innerProduct,
    Teuchos::RCP<DofOrdering> dofOrdering,
    Teuchos::RCP<BasisCache> basisCache)
{
  if (_isLegacySubclass)
  {
    // much of this code is the same as what's in the volume integration in computeStiffness...
    FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();

    unsigned numCells = physicalCubaturePoints.dimension(0);

    vector<int> testIDs = _bilinearForm->testIDs();
    vector<int>::iterator testIterator1;
    vector<int>::iterator testIterator2;

    BasisPtr test1Basis, test2Basis;

    innerProduct.initialize(0.0);

    for (testIterator1= testIDs.begin(); testIterator1 != testIDs.end(); testIterator1++)
    {
      int testID1 = *testIterator1;
      for (testIterator2= testIDs.begin(); testIterator2 != testIDs.end(); testIterator2++)
      {
        int testID2 = *testIterator2;

        vector<Camellia::EOperator> test1Operators;
        vector<Camellia::EOperator> test2Operators;

        operators(testID1,testID2,test1Operators,test2Operators);

        // check dimensions
        TEUCHOS_TEST_FOR_EXCEPTION( ( test1Operators.size() != test2Operators.size() ),
                                    std::invalid_argument,
                                    "test1Operators.size() and test2Operators.size() do not match.");

        vector<Camellia::EOperator>::iterator op1It;
        vector<Camellia::EOperator>::iterator op2It = test2Operators.begin();
        int operatorIndex = 0;
        for (op1It=test1Operators.begin(); op1It != test1Operators.end(); op1It++)
        {
          Camellia::EOperator op1 = *(op1It);
          Camellia::EOperator op2 = *(op2It);
          FieldContainer<Scalar> test1Values; // these will be resized inside applyOperator..
          FieldContainer<Scalar> test2Values; // derivative values

          test1Basis = dofOrdering->getBasis(testID1);
          test2Basis = dofOrdering->getBasis(testID2);

          int numDofs1 = test1Basis->getCardinality();
          int numDofs2 = test2Basis->getCardinality();

          FieldContainer<Scalar> miniMatrix( numCells, numDofs1, numDofs2 );

          Teuchos::RCP< const FieldContainer<Scalar> > test1ValuesTransformedWeighted, test2ValuesTransformed;

          test1ValuesTransformedWeighted = basisCache->getTransformedWeightedValues(test1Basis,op1);
          test2ValuesTransformed = basisCache->getTransformedValues(test2Basis,op2);

          FieldContainer<Scalar> innerProductDataAppliedToTest2 = *test2ValuesTransformed; // copy first
          FieldContainer<Scalar> innerProductDataAppliedToTest1 = *test1ValuesTransformedWeighted; // copy first

          //cout << "rank of test2ValuesTransformed: " << test2ValuesTransformed->rank() << endl;
          applyInnerProductData(innerProductDataAppliedToTest1, innerProductDataAppliedToTest2,
                                testID1, testID2, operatorIndex, basisCache);

          Intrepid::FunctionSpaceTools::integrate<Scalar>(miniMatrix,innerProductDataAppliedToTest1,
              innerProductDataAppliedToTest2,COMP_BLAS);

          int test1DofOffset = dofOrdering->getDofIndex(testID1,0);
          int test2DofOffset = dofOrdering->getDofIndex(testID2,0);

          // there may be a more efficient way to do this copying:
          for (int i=0; i < numDofs1; i++)
          {
            for (int j=0; j < numDofs2; j++)
            {
              for (unsigned k=0; k < numCells; k++)
              {
                innerProduct(k,i+test1DofOffset,j+test2DofOffset) += miniMatrix(k,i,j);
              }
            }
          }

          op2It++;
          operatorIndex++;
        }

      }
    }
  }
  else     // _isLegacySubclass is false
  {

    // innerProduct FC is sized as (C,F,F)
    const FieldContainer<double>* physicalCubaturePoints = &basisCache->getPhysicalCubaturePoints();

    unsigned numCells = physicalCubaturePoints->dimension(0);
    unsigned numDofs = dofOrdering->totalDofs();

    innerProduct.initialize(0.0);

//    int totalBasisCardinality = dofOrdering->getTotalBasisCardinality();
    // want to fit all basis values in 256K with a little room to spare -- 32768 doubles would be no room to spare
    // 256K is the size of Sandy Bridge's L2 cache.  Blue Gene Q has 32 MB shared L2 cache, so maybe there we could go bigger
    // (fitting within L1 is likely a non-starter for 3D DPG)
//    int maxValuesAllowed = 30000;
//    int maxPointsPerPhase = std::max(2, maxValuesAllowed / totalBasisCardinality); // minimally, compute 2 points at once

//    basisCache->setMaxPointsPerCubaturePhase(maxPointsPerPhase);
//    basisCache->setMaxPointsPerCubaturePhase(-1); // old behavior

//    for (int phase=0; phase < basisCache->getCubaturePhaseCount(); phase++) {
//      basisCache->setCubaturePhase(phase);
    for (typename vector< TLinearTermPtr<Scalar> >:: iterator ltIt = _linearTerms.begin();
         ltIt != _linearTerms.end(); ltIt++)
    {
      TLinearTermPtr<Scalar> lt = *ltIt;
      // integrate lt against itself
      lt->integrate(innerProduct,dofOrdering,lt,dofOrdering,basisCache,basisCache->isSideCache());
    }
//    }

//    basisCache->setMaxPointsPerCubaturePhase(-1); // infinite (BasisCache doesn't yet properly support phased *side* caches)

    bool enforceNumericalSymmetry = false;
    if (enforceNumericalSymmetry)
    {
      for (unsigned int c=0; c < numCells; c++)
        for (unsigned int i=0; i < numDofs; i++)
          for (unsigned int j=i+1; j < numDofs; j++)
          {
            innerProduct(c,i,j) = (innerProduct(c,i,j) + innerProduct(c,j,i)) / 2.0;
            innerProduct(c,j,i) = innerProduct(c,i,j);
          }
    }

    // boundary terms:
    for (typename vector< TLinearTermPtr<Scalar> >:: iterator btIt = _boundaryTerms.begin();
         btIt != _boundaryTerms.end(); btIt++)
    {
      TLinearTermPtr<Scalar> bt = *btIt;
      bool forceBoundary = true; // force interpretation of this as a term on the element boundary
      bt->integrate(innerProduct,dofOrdering,bt,dofOrdering,basisCache,forceBoundary);
    }

    // zero mean terms:
    for (typename vector< TLinearTermPtr<Scalar> >:: iterator ztIt = _zeroMeanTerms.begin();
         ztIt != _zeroMeanTerms.end(); ztIt++)
    {
      TLinearTermPtr<Scalar> zt = *ztIt;
      FieldContainer<Scalar> avgVector(numCells, numDofs);
      // Integrate against 1
      zt->integrate(avgVector, dofOrdering, basisCache);


      // cout << numDofs << avgVector << endl;

      // Sum into innerProduct
      for (unsigned int c=0; c < numCells; c++)
        for (unsigned int i=0; i < numDofs; i++)
          for (unsigned int j=0; j < numDofs; j++)
          {
            Scalar valAdd = avgVector(c, i) * avgVector(c, j);
            // cout << "(" << innerProduct(c, i, j) << ", " << valAdd << ") ";
            innerProduct(c, i, j) += valAdd;
          }
    }
  }
}

template <typename Scalar>
double TIP<Scalar>::computeMaxConditionNumber(DofOrderingPtr testSpace, BasisCachePtr basisCache)
{
  int testDofs = testSpace->totalDofs();
  int numCells = basisCache->cellIDs().size();
  FieldContainer<Scalar> innerProduct(numCells,testDofs,testDofs);
  this->computeInnerProductMatrix(innerProduct, testSpace, basisCache);
  double maxConditionNumber = -1;
  Teuchos::Array<int> cellIP_dim;
  cellIP_dim.push_back(testDofs);
  cellIP_dim.push_back(testDofs);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    FieldContainer<Scalar> cellIP = FieldContainer<Scalar>(cellIP_dim,&innerProduct(cellIndex,0,0) );
    double conditionNumber = SerialDenseMatrixUtility::estimate2NormConditionNumber(cellIP);
    maxConditionNumber = std::max(maxConditionNumber,conditionNumber);
  }
  return maxConditionNumber;
}

// compute TIP vector when var==fxn
template <typename Scalar>
void TIP<Scalar>::computeInnerProductVector(FieldContainer<Scalar> &ipVector,
    VarPtr var, TFunctionPtr<Scalar> fxn,
    Teuchos::RCP<DofOrdering> dofOrdering,
    Teuchos::RCP<BasisCache> basisCache)
{
  // ipVector FC is sized as (C,F)
  FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();

  if (!fxn.get())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fxn cannot be null!");
  }

  ipVector.initialize(0.0);

  for (typename vector< TLinearTermPtr<Scalar> >:: iterator ltIt = _linearTerms.begin();
       ltIt != _linearTerms.end(); ltIt++)
  {
    TLinearTermPtr<Scalar> lt = *ltIt;
    // integrate lt against itself, evaluated at var = fxn
    lt->integrate(ipVector,dofOrdering,lt,var,fxn,basisCache);
  }

  // boundary terms:
  for (typename vector< TLinearTermPtr<Scalar> >:: iterator btIt = _boundaryTerms.begin();
       btIt != _boundaryTerms.end(); btIt++)
  {
    TLinearTermPtr<Scalar> bt = *btIt;
    bool forceBoundary = true; // force interpretation of this as a term on the element boundary
    bt->integrate(ipVector,dofOrdering,bt,var,fxn,basisCache,forceBoundary);
  }

  // zero mean terms:
  for (typename vector< TLinearTermPtr<Scalar> >:: iterator ztIt = _zeroMeanTerms.begin();
       ztIt != _zeroMeanTerms.end(); ztIt++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zero mean terms not yet supported in TIP vector computation");
  }
}

template <typename Scalar>
bool TIP<Scalar>::hasBoundaryTerms()
{
  if (_isLegacySubclass) return false;
  else return _boundaryTerms.size() > 0;
}

// ! returns the number of potential nonzeros for the given trial ordering and test ordering
template <typename Scalar>
int TIP<Scalar>::nonZeroEntryCount(DofOrderingPtr testOrdering)
{
  int nonZeros = 0;
  
  set<pair<int,int>> testInteractions;
  
  for (TLinearTermPtr<Scalar> lt : _linearTerms)
  {
    set<int> varIDs = lt->varIDs();
    for (int test1 : varIDs)
    {
      for (int test2 : varIDs)
      {
        testInteractions.insert({test1,test2});
      }
    }
  }
  for (TLinearTermPtr<Scalar> lt : _boundaryTerms)
  {
    set<int> varIDs = lt->varIDs();
    for (int test1 : varIDs)
    {
      for (int test2 : varIDs)
      {
        testInteractions.insert({test1,test2});
      }
    }
  }
  for (TLinearTermPtr<Scalar> lt : _zeroMeanTerms)
  {
    set<int> varIDs = lt->varIDs();
    for (int test1 : varIDs)
    {
      for (int test2 : varIDs)
      {
        testInteractions.insert({test1,test2});
      }
    }
  }

  for (pair<int,int> testPair : testInteractions)
  {
    int test1Cardinality = testOrdering->getBasis(testPair.first)->getCardinality();
    int test2Cardinality = testOrdering->getBasis(testPair.second)->getCardinality();
    nonZeros += test1Cardinality * test2Cardinality;
  }
  
  return nonZeros;
}

template <typename Scalar>
void TIP<Scalar>::operators(int testID1, int testID2,
                            vector<Camellia::EOperator> &testOp1,
                            vector<Camellia::EOperator> &testOp2)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TIP<Scalar>::operators() not implemented.");
}

template <typename Scalar>
void TIP<Scalar>::printInteractions()
{
  if (_isLegacySubclass)
  {
    cout << "Inner product: test interactions\n";
    vector<int> testIDs = _bilinearForm->testIDs();
    for (vector<int>::iterator testIt = testIDs.begin(); testIt != testIDs.end(); testIt++)
    {
      int testID = *testIt;
      cout << endl << "****** Interactions with test variable " << _bilinearForm->testName(testID) << " ******* " << endl;
      bool first = true;
      for (vector<int>::iterator testIt2 = testIDs.begin(); testIt2 != testIDs.end(); testIt2++)
      {
        int testID2 = *testIt2;
        vector<Camellia::EOperator> ops1, ops2;
        operators(testID, testID2, ops1, ops2);
        int numOps = ops1.size();
        for (int i=0; i<numOps; i++)
        {
          if ( ! first) cout << " + ";
          cout << Camellia::operatorName(ops1[i]) << " " << _bilinearForm->testName(testID) << " ";
          cout << Camellia::operatorName(ops2[i]) << " " << _bilinearForm->testName(testID2);
          first = false;
        }
      }
      cout << endl;
    }
    return;
  }
  else
  {
    cout << "_linearTerms:\n";
    for (typename vector< TLinearTermPtr<Scalar> >::iterator ltIt = _linearTerms.begin();
         ltIt != _linearTerms.end(); ltIt++)
    {
      cout << (*ltIt)->displayString() << endl;
    }
    cout << "_boundaryTerms:\n";
    for (typename vector< TLinearTermPtr<Scalar> >::iterator ltIt = _boundaryTerms.begin();
         ltIt != _boundaryTerms.end(); ltIt++)
    {
      cout << (*ltIt)->displayString() << endl;
    }
    cout << "_zeroMeanTerms:\n";
    for (typename vector< TLinearTermPtr<Scalar> >::iterator ltIt = _zeroMeanTerms.begin();
         ltIt != _zeroMeanTerms.end(); ltIt++)
    {
      cout << (*ltIt)->displayString() << endl;
    }
  }
}

template <typename Scalar>
pair<TIPPtr<Scalar>, VarPtr> TIP<Scalar>::standardInnerProductForFunctionSpace(Camellia::EFunctionSpace fs, bool useTraceVar, int spaceDim)
{
  TIPPtr<Scalar> ip = Teuchos::rcp( new TIP<Scalar> );
  VarFactoryPtr vf = VarFactory::varFactory();
  Camellia::Space space = Camellia::spaceForEFS(fs);
  VarPtr var = useTraceVar ? vf->traceVar("v",space) : vf->testVar("v", space);

  ip->addTerm(var);

  switch (fs)
  {
  case Camellia::FUNCTION_SPACE_HVOL:
  case Camellia::FUNCTION_SPACE_HVOL_DISC:
  case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
    break;
  case Camellia::FUNCTION_SPACE_HGRAD:
  case Camellia::FUNCTION_SPACE_HGRAD_DISC:
  case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
    ip->addTerm(var->grad());
    break;
  case Camellia::FUNCTION_SPACE_HCURL:
  case Camellia::FUNCTION_SPACE_HCURL_DISC:
    ip->addTerm(var->curl(spaceDim));
    break;
  case Camellia::FUNCTION_SPACE_HDIV:
  case Camellia::FUNCTION_SPACE_HDIV_DISC:
    ip->addTerm(var->div());
    break;
      // space-time:
    case Camellia::FUNCTION_SPACE_HGRAD_SPACE_HVOL_TIME:
      ip->addTerm(var->grad()); // by default, grad means spatial gradient
      // space-time:
    case Camellia::FUNCTION_SPACE_HVOL_SPACE_HGRAD_TIME:
      ip->addTerm(var->dt());
  case Camellia::FUNCTION_SPACE_REAL_SCALAR:
    break;

  default:
    cout << "Error: unhandled function space.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled function space");
    break;
  }
  return make_pair(ip,var);
}

template <typename Scalar>
TIPPtr<Scalar> TIP<Scalar>::ip()
{
  return Teuchos::rcp( new TIP<Scalar> );
}

namespace Camellia
{
template class TIP<double>;
}
