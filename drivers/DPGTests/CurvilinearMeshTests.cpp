//
//  CurvilinearMeshTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/21/13.
//
//

#include "CurvilinearMeshTests.h"

#include "MeshFactory.h"
#include "Mesh.h"
#include "Function.h"

#include "GnuPlotUtil.h"

#include "StokesFormulation.h"

const static double PI  = 3.141592653589793238462;

void CurvilinearMeshTests::setup() {
  
}

void CurvilinearMeshTests::teardown() {
  
}

void CurvilinearMeshTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testStraightEdgeMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testCylinderMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool CurvilinearMeshTests::testCylinderMesh() {
  bool success = true;
  
  FunctionPtr one = Function::constant(1.0);
  
  double width = 3.0;
  double height = 3.0;
  double r = 1.0;
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  double trueArea = width * height - PI * r * r;
  
  int H1Order = 1;
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  
  MeshPtr mesh = MeshFactory::flowPastCylinderMesh(width, height, r, bf, H1Order, pToAdd);
  
  double approximateArea = one->integrate(mesh);
  
//  cout << setprecision(15);
//  cout << "Exact area:" << trueArea << endl;
//  cout << "Approximate area on straight-line mesh: " << approximateArea << endl;
//  
  double tol = 1e-10;
  // test p-convergence of mesh area
  double previousError = abs(trueArea - approximateArea);
  for (int i=0; i<3; i++) {
    H1Order++;
    mesh = MeshFactory::flowPastCylinderMesh(width, height, r, bf, H1Order, pToAdd);
    approximateArea = one->integrate(mesh);
//    cout << "Area with H1Order " << H1Order << ": " << approximateArea << endl;
    double error = abs(trueArea - approximateArea);
    if ((error > previousError) && (error > tol)) { // non-convergence
      success = false;
      cout << "Error with H1Order = " << H1Order << " is greater than with H1Order = " << H1Order - 1 << endl;
      cout << "Current error = " << error << "; previous = " << previousError << endl;
    }
//    ostringstream filePath;
//    filePath << "/tmp/cylinderFlowMesh" << H1Order << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
    previousError = error;
  }
  
//  GnuPlotUtil::writeExactMeshSkeleton("/tmp/cylinderFlowExactMesh.dat", mesh, 10);
  
  // TODO: add tests against h-refinements and p-refinements (instead of simply using finer initial mesh)
  return success;
}

bool CurvilinearMeshTests::testStraightEdgeMesh() {
  bool success = true;
  
  // to begin, a very simple test: do we compute the correct area for a square?
  FunctionPtr one = Function::constant(1.0);
  
  double width = 1.0;
  double height = 1.0;
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  double trueArea = width * height;
  
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  

    // make a single-element mesh:
  int H1Order = 1;
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
  double approximateArea = one->integrate(mesh);
  
  // sanity check on the test: regular mesh gets the area right:
  double tol = 1e-14;
  double err = abs(trueArea-approximateArea);
  if (err > tol) {
    success = false;
    cout << "Error: even regular mesh (no curves set) gets the area wrong.\n";
  }
  
//  GnuPlotUtil::writeExactMeshSkeleton("/tmp/unitMesh.dat", mesh, 2);
  
  for (int i=0; i<4; i++) {
    H1Order = i+1;
    mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
  
    // now, set curves for each edge:
    map< pair<int, int>, ParametricFunctionPtr > edgeToCurveMap;
    
    int cellID = 0; // the only cell
    vector< ParametricFunctionPtr > lines = mesh->parametricEdgesForCell(cellID);
    vector< int > vertices = mesh->vertexIndicesForCell(cellID);
    
    for (int i=0; i<vertices.size(); i++) {
      int vertex = vertices[i];
      int nextVertex = vertices[(i+1) % vertices.size()];
      pair< int, int > edge = make_pair(vertex,nextVertex);
      edgeToCurveMap[edge] = lines[i];
    }
    
    mesh->setEdgeToCurveMap(edgeToCurveMap);
    
    // now repeat with our straight-edge curves:
    approximateArea = one->integrate(mesh);
    tol = 1e-14;
    err = abs(trueArea-approximateArea);
    if (err > tol) {
      success = false;
      cout << "Error: mesh with straight-edge 'curves' and H1Order " << H1Order;
      cout << " has area " << approximateArea << "; should be " << trueArea << "." << endl;
    }
    
//    ostringstream filePath;
//    filePath << "/tmp/unitMesh" << H1Order << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
  }

  return success;
}

std::string CurvilinearMeshTests::testSuiteName() {
  return "CurvilinearMeshTests";
}