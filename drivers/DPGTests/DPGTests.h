//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#ifndef DPG_TESTS
#define DPG_TESTS

using namespace Camellia;
using namespace Intrepid;
using namespace std;

class Epetra_SerialDenseMatrix;

class DPGTests
{
public:
  static void runExceptionThrowingTest();

  static void runTests();
  static void createBases();
  static bool testDofOrdering();
  static bool testAnalyticBoundaryIntegral(bool);
  static bool testComputeStiffnessConformingVertices();
  static bool testComputeStiffnessTrace();
  static bool testComputeStiffnessFlux();
  static bool testMathInnerProductDx();
  static bool testOptimalStiffnessByMultiplying();
  static bool testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices();
  static bool testWeightBasis();
  static bool checkOptTestWeights(FieldContainer<double> &optWeights,
                                  FieldContainer<double> &ipMatrix,
                                  FieldContainer<double> &preStiffness,
                                  double tol);
  static bool fcsAgree(string &testName, FieldContainer<double> &expected,
                       FieldContainer<double> &actual, double tol);
  static bool fcEqualsSDM(FieldContainer<double> &fc, int cellIndex,
                          Epetra_SerialDenseMatrix &sdm, double tol, bool transpose);
  static bool fcIsSymmetric(FieldContainer<double> &fc, double tol,
                            int &cellOfAsymmetry,
                            int &rowOfAsymmetry, int &colOfAsymmetry);
};

#endif
