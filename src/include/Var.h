//
//  Var.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Var_h
#define Camellia_Var_h

#include "TypeDefs.h"

#include "Teuchos_RCP.hpp"

#include "CamelliaIntrepidExtendedTypes.h"

namespace Camellia
{
enum Space { HGRAD, HCURL, HDIV, HGRAD_DISC, HCURL_DISC, HDIV_DISC, HDIV_FREE, L2, CONSTANT_SCALAR, VECTOR_HGRAD, VECTOR_HGRAD_DISC, VECTOR_L2, HGRAD_SPACE_L2_TIME, L2_SPACE_HGRAD_TIME, UNKNOWN_FS };
enum VarType { TEST, FIELD, TRACE, FLUX, UNKNOWN_TYPE, MIXED_TYPE };

Camellia::EFunctionSpace efsForSpace(Space space);
Space spaceForEFS(Camellia::EFunctionSpace efs);
int rankForSpace(Space space);

class Var   // really Var x Operator
{
  int _rank;
  int _id;
  std::string _name;
  Camellia::Space _fs;
  Camellia::EOperator _op; // default is OP_VALUE
  Camellia::VarType _varType;
  LinearTermPtr _termTraced; // for trace variables, optionally allows identification with fields
  //  map< Camellia::EOperator, VarPtr > _relatedVars; // grad, div, etc. could be cached here
  bool _definedOnTemporalInterfaces;
public:
  Var(int ID, int rank, std::string name, Camellia::EOperator op =  Camellia::OP_VALUE,
      Camellia::Space fs = Camellia::UNKNOWN_FS, Camellia::VarType varType = Camellia::UNKNOWN_TYPE, LinearTermPtr termTraced = Teuchos::rcp((LinearTerm*) NULL),
      bool definedOnTemporalInterfaces = true);

  int ID() const;
  const std::string & name() const;
  std::string displayString() const;
  Camellia::EOperator op() const;
  int rank() const;  // 0 for scalar, 1 for vector, etc.
  Camellia::Space space() const;
  Camellia::VarType varType() const;

  VarPtr grad() const;
  VarPtr div() const;
  VarPtr curl(int spaceDim) const; // 3D curl differs from 2D
  VarPtr dx() const;
  VarPtr dy() const;
  VarPtr dz() const;
  VarPtr dt() const;
  VarPtr x() const;
  VarPtr y() const;
  VarPtr z() const;
  VarPtr t() const;
  VarPtr laplacian() const;
  
  // ! i is 1-based ordinal; 1 for x, 2 for y, 3 for z
  VarPtr spatialComponent(int i) const;
  
  // ! i is 1-based ordinal; 1 for dx, 2 for dy, 3 for dz
  VarPtr di(int i) const;

  LinearTermPtr cross_normal(int spaceDim) const;
  LinearTermPtr dot_normal() const;
  LinearTermPtr times_normal() const;
  LinearTermPtr times_normal_x() const;
  LinearTermPtr times_normal_y() const;
  LinearTermPtr times_normal_z() const;
  LinearTermPtr times_normal_t() const;

  LinearTermPtr termTraced() const;

  /** \brief  Used for space-time elements; returns whether the Var is defined on temporal interfaces.  (Variables which are traces of terms involving a spatial normal only may degenerate on such interfaces.)
   */
  bool isDefinedOnTemporalInterface() const;

  template <typename Scalar>
  static VarPtr varForTrialID(int trialID, TBFPtr<Scalar> bf);
};
}

#endif
