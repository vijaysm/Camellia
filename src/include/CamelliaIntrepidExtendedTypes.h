#ifndef CAMELLIA_INTREPID_EXTENDED_TYPES
#define CAMELLIA_INTREPID_EXTENDED_TYPES

#include <set>
#include <string>

namespace Camellia
{
enum EOperator   // first 13 simply copied from Intrepid::EOperator
{
  OP_VALUE = 0,
  OP_GRAD,      // 1
  OP_CURL,      // 2
  OP_DIV,       // 3
  OP_D1,        // 4
  OP_D2,        // 5
  OP_D3,        // 6
  OP_D4,        // 7
  OP_D5,        // 8
  OP_D6,        // 9
  OP_D7,        // 10
  OP_D8,        // 11
  OP_D9,        // 12
  OP_D10,       // 13
  OP_X,         // 14 (pick up where EOperator left off...)
  OP_Y,         // 15
  OP_Z,         // 16
  OP_T,         // 17
  OP_DX,        // 18
  OP_DY,        // 19
  OP_DZ,        // 20
  OP_DT,        // 21
  OP_CROSS_NORMAL,    // 22
  OP_DOT_NORMAL,      // 23
  OP_TIMES_NORMAL,    // 24
  OP_TIMES_NORMAL_X,  // 25
  OP_TIMES_NORMAL_Y,  // 26
  OP_TIMES_NORMAL_Z,  // 27
  OP_TIMES_NORMAL_T,  // 28
  OP_VECTORIZE_VALUE, // 29
  OP_LAPLACIAN,       // 30
  OP_DXDX,            // 31
  OP_DYDY,            // 32
  OP_DZDZ             // 33
};

enum EFunctionSpace   // the first four copied from Intrepid::EFunctionSpace
{
  FUNCTION_SPACE_HGRAD = 0,
  FUNCTION_SPACE_HCURL,
  FUNCTION_SPACE_HDIV,
  FUNCTION_SPACE_HVOL,
  FUNCTION_SPACE_VECTOR_HGRAD,
  FUNCTION_SPACE_TENSOR_HGRAD,
  FUNCTION_SPACE_VECTOR_HVOL,
  FUNCTION_SPACE_TENSOR_HVOL,
  FUNCTION_SPACE_HGRAD_DISC,
  FUNCTION_SPACE_HCURL_DISC,
  FUNCTION_SPACE_HDIV_DISC,
  FUNCTION_SPACE_HVOL_DISC,
  FUNCTION_SPACE_VECTOR_HGRAD_DISC,
  FUNCTION_SPACE_TENSOR_HGRAD_DISC,
  FUNCTION_SPACE_VECTOR_HVOL_DISC,
  FUNCTION_SPACE_TENSOR_HVOL_DISC,
  FUNCTION_SPACE_REAL_SCALAR,
  FUNCTION_SPACE_HDIV_FREE,
  FUNCTION_SPACE_HGRAD_SPACE_HVOL_TIME,
  FUNCTION_SPACE_HVOL_SPACE_HGRAD_TIME,
  FUNCTION_SPACE_UNKNOWN
};

bool functionSpaceIsVectorized(EFunctionSpace fs);
bool functionSpaceIsDiscontinuous(EFunctionSpace fs);
EFunctionSpace continuousSpaceForDiscontinuous(EFunctionSpace fs_disc, bool upgradeHVOL = true);
EFunctionSpace discontinuousSpaceForContinuous(EFunctionSpace fs_continuous);

const std::set<EOperator> & normalOperators();

const std::string & operatorName(EOperator op);
int operatorRank(EOperator op, EFunctionSpace fs);
}

#endif