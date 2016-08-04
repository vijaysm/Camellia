// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER
//
//  BasisReconciliation.h
//  Camellia
//
//  Created by Nate Roberts on 11/19/13.
//
//

#ifndef Camellia_debug_BasisReconciliation_h
#define Camellia_debug_BasisReconciliation_h

#include "Intrepid_FieldContainer.hpp"

#include "RefinementPattern.h"

#include "Basis.h"

#include "LinearTerm.h"

namespace Camellia
{
struct SubBasisReconciliationWeights
{
  Intrepid::FieldContainer<double> weights; // indices are (fine, coarse)
  std::set<int> fineOrdinals;
  std::set<int> coarseOrdinals;
  
  bool isIdentity = false; // fineOrdinals == coarseOrdinals; weights is the identity matrix (not stored)
};

class BasisReconciliation
{
  bool _cacheResults;

  // TODO: simplify this: eliminate simple reconciliation weights, and the h/p distinction.  Everything can happen in terms of subcell reconciliation.  (Simple is just subcdim = domain dimension, subcord = 0.  The non-h variant is just an empty RefinementBranch.)

  typedef pair< Camellia::Basis<>*, pair<unsigned, unsigned> > SubcellBasisRestriction;  // second pair is (subcdim, subcord)
  typedef pair< Camellia::Basis<>*, int > SideBasisRestriction;
  // cached values:
  typedef unsigned Permutation;
  typedef pair< Camellia::Basis<>*, Camellia::Basis<>*> BasisPair; // fineBasis first.
  map< pair<BasisPair, Permutation>, Intrepid::FieldContainer<double> > _simpleReconciliationWeights; // simple: no sides involved
  map< pair< pair< SideBasisRestriction, SideBasisRestriction >, Permutation >, SubBasisReconciliationWeights > _sideReconciliationWeights;
private:
  typedef pair< BasisPair, RefinementBranch > RefinedBasisPair; // fineBasis (the one on the refined element) is first in the BasisPair
  typedef pair< pair< SideBasisRestriction, SideBasisRestriction >, RefinementBranch > SideRefinedBasisPair;
  typedef pair< pair< SubcellBasisRestriction, SubcellBasisRestriction >, RefinementBranch > SubcellRefinedBasisPair;
  map< pair<RefinedBasisPair, Permutation>, Intrepid::FieldContainer<double> > _simpleReconcilationWeights_h;
  map< pair< SideRefinedBasisPair, Permutation> , SubBasisReconciliationWeights > _sideReconcilationWeights_h;

  // this is the only map that actually needs to remain, after the code simplification described above...
  map< pair< SubcellRefinedBasisPair, Permutation> , SubBasisReconciliationWeights > _subcellReconcilationWeights;

//  // trace to field reconciliation:
//  // we do need a separate container for maps from fields to traces, because each can have a distinct LinearTerm describing
//  // the relationship of the trace to the fields
//  // in the _termTracedSubcellReconcilationWeights map below, notes:
//  // - key to outer map is the pointer contained in termTraced.
//  // - key to the inner map:
//  //   - key.first: the variable ID of the field variable in the LinearTerm.
//  //   - key.second.first: the SubcellRefinedBasisPair
//  //   - key.second.second: the permutation of the fine cell's ancestor relative to what is seen by the coarse basis
//  map< LinearTerm*, map< pair<int, pair< SubcellRefinedBasisPair, Permutation > >, SubBasisReconciliationWeights > > _termTracedSubcellReconcilationWeights;

  // trying something else for trace to field, to allow more reuse when multiple linear terms are in the same relationship to fields and traces:
  typedef pair< SubcellRefinedBasisPair, Permutation> PermutedRefinedBasisPair;
  typedef pair<unsigned, unsigned> FineCoarseDomainOrdinalPair;
  typedef pair< PermutedRefinedBasisPair, FineCoarseDomainOrdinalPair > PermutedRefinedBasisPairDomainOrdinals;
  typedef vector<pair<Function*, Camellia::EOperator>> FieldOps; // the Function* thing is *NOT* perfectly safe; this is a reason that BasisReconciliation's cache should not live too long -- Function could change underneath (as with Solution functions) or could even be deleted and replaced by a different function in the same memory location.
  typedef pair<PermutedRefinedBasisPairDomainOrdinals, FieldOps> TermTracedCacheKey;
  map<TermTracedCacheKey, SubBasisReconciliationWeights> _termsTraced;
  
  static Intrepid::FieldContainer<double> filterBasisValues(const Intrepid::FieldContainer<double> &basisValues, std::set<int> &filter);

  static SubBasisReconciliationWeights filterToInclude(std::set<int> &rowOrdinals, std::set<int> &colOrdinals, const SubBasisReconciliationWeights &weights);
public:
  BasisReconciliation(bool cacheResults = true)
  {
    _cacheResults = cacheResults;
  }

  // p
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, unsigned vertexNodePermutation); // requires these to be defined on the same topology
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation); // requires the sides to have the same topology

  // h
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, unsigned vertexNodePermutation);
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, RefinementBranch &domainRefinements, BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation); // vertexPermutation is for the fine basis's ancestral orientation (how to permute side as seen by fine's ancestor to produce side as seen by coarse)...

  // the new bottleneck method (the others can be reimplemented to call this one, or simply eliminated)
  const SubBasisReconciliationWeights &constrainedWeights(unsigned subcellDimension,
      BasisPtr finerBasis, unsigned finerBasisSubcellOrdinal,
      RefinementBranch &refinements,
      BasisPtr coarserBasis, unsigned coarserBasisSubcellOrdinal,
      unsigned vertexNodePermutation);  // vertexNodePermutation: how to permute the subcell vertices as seen by finerBasis to get the one seen by coarserBasis.
  
  const SubBasisReconciliationWeights &constrainedWeightsForTermTraced(LinearTermPtr termTraced, int fieldID,
                                                                       unsigned fineSubcellDimension,
                                                                       BasisPtr finerBasis,
                                                                       unsigned fineSubcellOrdinalInFineDomain,
                                                                       RefinementBranch &cellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
                                                                       unsigned fineDomainOrdinalInRefinementLeaf,
                                                                       CellTopoPtr coarseCellTopo,
                                                                       unsigned coarseSubcellDimension,
                                                                       BasisPtr coarserBasis, unsigned coarseSubcellOrdinalInCoarseDomain,
                                                                       unsigned coarseDomainOrdinalInCoarseCellTopo, // we use the coarserBasis's domain topology to determine the domain's space dimension
                                                                       unsigned coarseSubcellPermutation);

  // static workhorse methods:
  static SubBasisReconciliationWeights computeConstrainedWeights(unsigned subcellDimension,
      BasisPtr finerBasis, unsigned finerBasisSubcellOrdinal,
      RefinementBranch &refinements,
      BasisPtr coarserBasis, unsigned coarserBasisSubcellOrdinal,
      unsigned vertexNodePermutation);  // vertexNodePermutation: how to permute the subcell vertices as seen by finerBasis to get the one seen by coarserBasis.

  static SubBasisReconciliationWeights computeConstrainedWeights(unsigned fineSubcellDimension,
      BasisPtr finerBasis, unsigned fineSubcellOrdinalInFineDomain,
      RefinementBranch &fineCellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
      unsigned fineDomainOrdinalInRefinementLeaf,
      CellTopoPtr coarseCellTopo,
      unsigned coarseSubcellDimension,
      BasisPtr coarserBasis, unsigned coarseSubcellOrdinalInCoarseDomain,
      unsigned coarseDomainOrdinalInCoarseCellTopo, // we use the coarserBasis's domain topology to determine the domain's space dimension
      unsigned coarseSubcellPermutation);  // coarseSubcellPermutation: how to permute the nodes of the refinement root to get the subcell seen by the coarse cell.  (This is DIFFERENT from the one in the other computeConstrainedWeights, which deals with the view from the ancestral and coarse *domains*.)

  static SubBasisReconciliationWeights computeConstrainedWeightsForTermTraced(LinearTermPtr termTraced, int fieldID,
      unsigned fineSubcellDimension,
      BasisPtr finerBasis,
      unsigned fineSubcellOrdinalInFineDomain,
      RefinementBranch &cellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
      unsigned fineDomainOrdinalInRefinementLeaf,
      CellTopoPtr coarseCellTopo,
      unsigned coarseSubcellDimension,
      BasisPtr coarserBasis, unsigned coarseSubcellOrdinalInCoarseDomain,
      unsigned coarseDomainOrdinalInCoarseCellTopo, // we use the coarserBasis's domain topology to determine the domain's space dimension
      unsigned coarseSubcellPermutation);  // coarseSubcellPermutation: how to permute the nodes of the refinement root to get the subcell seen by the coarse cell.  (This is DIFFERENT from the one in the side-centric computeConstrainedWeights, which deals with the view from the ancestral and coarse *domains*.)

  static SubBasisReconciliationWeights weightsForCoarseSubcell(const SubBasisReconciliationWeights &weights, BasisPtr constrainingBasis, unsigned subcdim, unsigned subcord, bool includeSubsubcells);

  static SubBasisReconciliationWeights composedSubBasisReconciliationWeights(const SubBasisReconciliationWeights &aWeights, const SubBasisReconciliationWeights &bWeights);

  // !equalWeights is intended primarily for debugging.  May be inefficient.
  static bool equalWeights(const SubBasisReconciliationWeights &aWeights, const SubBasisReconciliationWeights &bWeights, double tol = 1e-15);

  static SubBasisReconciliationWeights filterOutZeroRowsAndColumns(SubBasisReconciliationWeights &weights);

  static SubBasisReconciliationWeights sumWeights(const SubBasisReconciliationWeights &aWeights, const SubBasisReconciliationWeights &bWeights);

//  static std::set<int> interiorDofOrdinalsForBasis(BasisPtr basis);

  static set<unsigned> internalDofOrdinalsForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements); // which degrees of freedom in the finer basis have empty support on the boundary of the coarser basis's reference element? -- these are the ones for which the constrained weights are determined in computeConstrainedWeights.
  static set<unsigned> internalDofOrdinalsForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements, unsigned subcdim, unsigned subcord);

  static unsigned minimumSubcellDimension(BasisPtr basis); // for continuity enforcement

public:
  // !! this method exposed publicly primarily for testing purposes.
  static void mapFineSubcellPointsToCoarseDomain(Intrepid::FieldContainer<double> &coarseDomainPoints, const Intrepid::FieldContainer<double> &fineSubcellPoints,
      unsigned fineSubcellDimension,
      unsigned fineSubcellOrdinalInFineDomain,
      unsigned fineDomainDim,
      unsigned fineDomainOrdinalInRefinementLeaf,
      RefinementBranch &cellRefinementBranch,
      CellTopoPtr coarseCellTopo,
      unsigned coarseSubcellDimension,
      unsigned coarseSubcellOrdinalInCoarseDomain,
      unsigned coarseDomainDim,
      unsigned coarseDomainOrdinalInCoarseCellTopo,
      unsigned coarseSubcellPermutation); // coarseSubcellPermutation: how to permute the nodes of the refinement root to get the subcell seen by the coarse cell.  (This is DIFFERENT from the one in the side-centric computeConstrainedWeights, which deals with the view from the ancestral and coarse *domains*.)
  // !! this method exposed publicly primarily for testing purposes.
  static void setupFineAndCoarseBasisCachesForReconciliation(BasisCachePtr &fineBasisCache, BasisCachePtr &coarseBasisCache,
      unsigned fineSubcellDimension,
      BasisPtr finerBasis,
      unsigned fineSubcellOrdinalInFineDomain,
      RefinementBranch &fineCellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
      unsigned fineDomainOrdinalInRefinementLeaf,
      CellTopoPtr coarseCellTopo,
      unsigned coarseSubcellDimension,
      BasisPtr coarserBasis, unsigned coarseSubcellOrdinalInCoarseDomain,
      unsigned coarseDomainOrdinalInCoarseCellTopo, // we use the coarserBasis's domain topology to determine the domain's space dimension
      unsigned coarseSubcellPermutation); // coarseSubcellPermutation: how to permute the nodes of the refinement root to get the subcell seen by the coarse cell.  (This is DIFFERENT from the one in the side-centric computeConstrainedWeights, which deals with the view from the ancestral and coarse *domains*.)
};
}

#endif
