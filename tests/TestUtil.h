#include "Intrepid_FieldContainer.hpp"
using namespace Intrepid;
using namespace std;

static bool fcsAgree(const FieldContainer<double> &fc1, const FieldContainer<double> &fc2, double tol, double &maxDiff) {
  if (fc1.size() != fc2.size()) {
    maxDiff = -1.0; // a signal something's wrong…
    return false;
  }
  maxDiff = 0.0;
  for (int i=0; i<fc1.size(); i++) {
    maxDiff = max(maxDiff, abs(fc1[i] - fc2[i]));
  }
  return (maxDiff <= tol);
}

