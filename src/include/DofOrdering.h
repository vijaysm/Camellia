#ifndef DOF_ORDERING
#define DOF_ORDERING

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

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Basis.h"

#include "CellTopology.h"

namespace Camellia
{
  const static int VOLUME_INTERIOR_SIDE_ORDINAL = -1;
class DofOrdering
{
  int _indexNeedsToBeRebuilt;
  int _nextIndex;
  
  int _volumeIndex; // where the volume indices are stored in _indices
  
  //  vector<int> varIDs;
  std::set<int> varIDs;
  //  std::vector<int> varIDsVector;
  std::map< std::pair<int, std::pair<int, int> >, std::pair<int, int> > dofIdentifications; // keys: <varID, <sideIndex, dofOrdinal> >
  // values: <sideIndex, dofOrdinal>
  std::map< int, std::vector<int> > _sidesForVarID; // to replace numSidesForVarID
  std::map<int,int> numSidesForVarID;
//  std::map< std::pair<int,int>, std::vector<int> > indices; // keys for indices are <varID, sideIndex >, where sideIndex = VOLUME_INTERIOR_SIDE_ORDINAL for field (volume) variables
  std::vector<map<int,vector<int>>> _indices; // outer index is sideOrdinal (or _volumeIndex); next is varID.
  // values for indices: list of the indices used in the DofOrdering by this <varID, sideIndex> pair's basis, ordered according to that basis's ordering
  std::map< std::pair<int,int>, BasisPtr > bases; // keys are <varID, sideIndex>
  std::map< int, int > basisRanks; // keys are varIDs; values are 0,1,2,... (scalar, vector, tensor)

  std::map< int, CellTopoPtr > _cellTopologyForSide; // -1 is field variable
public:
  DofOrdering(CellTopoPtr cellTopo = Teuchos::null); // constructor

  void addEntry(int varID, BasisPtr basis, int basisRank, int sideIndex = VOLUME_INTERIOR_SIDE_ORDINAL);

  bool hasBasisEntry(int varID, int sideIndex) const;
  bool hasSideVarIDs();

  void copyLikeCoefficients( Intrepid::FieldContainer<double> &newValues, Teuchos::RCP<DofOrdering> oldDofOrdering,
                             const Intrepid::FieldContainer<double> &oldValues );

  // get the varIndex variable's dof with basis ordinal dofId in the Dof ordering:
  int getDofIndex(int varID, int basisDofOrdinal, int sideIndex=VOLUME_INTERIOR_SIDE_ORDINAL, int subSideIndex = -1);

  const std::vector<int> & getDofIndices(int varID, int sideIndex=VOLUME_INTERIOR_SIDE_ORDINAL) const;

  const std::set<int> & getVarIDs() const;

  std::set<int> getTraceDofIndices(); // returns dof indices corresponding to variables with numSides > 1.

  int getBasisCardinality(int varID, int sideIndex);

  BasisPtr getBasis(int varID, int sideIndex = VOLUME_INTERIOR_SIDE_ORDINAL) const;

  int getBasisRank(int varID)
  {
    return basisRanks[varID];
  }

  bool hasEntryForVarID( int varID ); // returns true if we have any basis on any side for varID

  int getNumSidesForVarID(int varID); // will be deprecated soon.  Use getSidesForVarID instead

  const vector<int> & getSidesForVarID(int varID) const;

  int getTotalBasisCardinality(); // sum of all the *distinct* bases' cardinalities

  void addIdentification(int varID, int side1, int basisDofOrdinal1,
                         int side2, int basisDofOrdinal2);

  CellTopoPtr cellTopology(int sideIndex = -1) const;

  int minimumSubcellDimensionForContinuity() const; // across all bases
  int maxBasisDegree();
  int maxBasisDegreeForVolume();

  void print(std::ostream& os);
  
  int totalDofs() const
  {
    return _nextIndex;
  }

  void rebuildIndex();
  
  // ! Returns vector containing (varID,vector<sideOrdinal>) entries corresponding to variables with nonzero coefficients in the provided container
  vector<pair<int,vector<int>>> variablesWithNonZeroEntries(const Intrepid::FieldContainer<double> &localCoefficients, double tol = 0.0) const;
};
}

std::ostream& operator << (std::ostream& os, const Camellia::DofOrdering& dofOrdering);

#endif
