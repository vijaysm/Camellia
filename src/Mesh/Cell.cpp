//
//  Cell.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/22/14.
//
//

#include "Cell.h"
#include "RefinementPattern.h"

vector< pair<GlobalIndexType, unsigned> > Cell::childrenForSide(unsigned sideIndex) {
  vector< pair<GlobalIndexType, unsigned> > childIndicesForSide;
  
  if (_refPattern.get() != NULL) {
    vector< pair<unsigned, unsigned> > refinementChildIndicesForSide = _refPattern->childrenForSides()[sideIndex];
    
    for( vector< pair<unsigned, unsigned> >::iterator entryIt = refinementChildIndicesForSide.begin();
        entryIt != refinementChildIndicesForSide.end(); entryIt++) {
      unsigned childIndex = _children[entryIt->first]->cellIndex();
      unsigned childSide = entryIt->second;
      childIndicesForSide.push_back(make_pair(childIndex,childSide));
    }
  }
  
  return childIndicesForSide;
}

vector< pair< GlobalIndexType, unsigned> > Cell::getDescendantsForSide(int sideIndex, bool leafNodesOnly) {
  // if leafNodesOnly == true,  returns a flat list of leaf nodes (descendants that are not themselves parents)
  // if leafNodesOnly == false, returns a list in descending order: immediate children, then their children, and so on.
  
  // guarantee is that if a child and its parent are both in the list, the parent will come first
  
  // pair (descendantCellIndex, descendantSideIndex)
  vector< pair< GlobalIndexType, unsigned> > descendantsForSide;
  if ( ! isParent() ) {
    descendantsForSide.push_back( make_pair( _cellIndex, sideIndex) );
    return descendantsForSide;
  }
  
  vector< pair<unsigned,unsigned> > childIndices = _refPattern->childrenForSides()[sideIndex];
  vector< pair<unsigned,unsigned> >::iterator entryIt;
  
  for (entryIt=childIndices.begin(); entryIt != childIndices.end(); entryIt++) {
    unsigned childIndex = (*entryIt).first;
    unsigned childSideIndex = (*entryIt).second;
    if ( (! _children[childIndex]->isParent()) || (! leafNodesOnly ) ) {
      // (            leaf node              ) || ...
      descendantsForSide.push_back( make_pair( _children[childIndex]->cellIndex(), childSideIndex) );
    }
    if ( _children[childIndex]->isParent() ) {
      vector< pair<GlobalIndexType,unsigned> > childDescendants = _children[childIndex]->getDescendantsForSide(childSideIndex,leafNodesOnly);
//      descendantsForSide.insert(descendantsForSide.end(), childDescendants.begin(), childDescendants.end());
      vector< pair<GlobalIndexType,unsigned> >::iterator childEntryIt;
      for (childEntryIt=childDescendants.begin(); childEntryIt != childDescendants.end(); childEntryIt++) {
        descendantsForSide.push_back(*childEntryIt);
      }
    }
  }
  return descendantsForSide;
}

Cell::Cell(CellTopoPtr cellTopo, const vector<unsigned> &vertices, const vector< map< unsigned, unsigned > > &subcellPermutations,
     unsigned cellIndex, MeshTopology* meshTopo) {
  _cellTopo = cellTopo;
  _vertices = vertices;
  _subcellPermutations = subcellPermutations;
  _cellIndex = cellIndex;
  _meshTopo = meshTopo;
  _neighbors = vector< pair<GlobalIndexType, unsigned> >(_cellTopo->getSideCount(),make_pair(-1,-1));
}
unsigned Cell::cellIndex() {
  return _cellIndex;
}

const vector< Teuchos::RCP< Cell > > &Cell::children() {
  return _children;
}
void Cell::setChildren(vector< Teuchos::RCP< Cell > > children) {
  _children = children;
  Teuchos::RCP< Cell > thisPtr = Teuchos::rcp( this, false ); // doesn't own memory
  for (vector< Teuchos::RCP< Cell > >::iterator childIt = children.begin(); childIt != children.end(); childIt++) {
    (*childIt)->setParent(thisPtr);
  }
}
vector<unsigned> Cell::getChildIndices() {
  vector<unsigned> indices(_children.size());
  for (unsigned childOrdinal=0; childOrdinal<_children.size(); childOrdinal++) {
    indices[childOrdinal] = _children[childOrdinal]->cellIndex();
  }
  return indices;
}

unsigned Cell::entityIndex(unsigned subcdim, unsigned subcord) {
  set< unsigned > nodes;
  if (subcdim != 0) {
    int entityNodeCount = _cellTopo->getNodeCount(subcdim, subcord);
    for (int node=0; node<entityNodeCount; node++) {
      unsigned nodeIndexInCell = _cellTopo->getNodeMap(subcdim, subcord, node);
      nodes.insert(_vertices[nodeIndexInCell]);
    }
  } else {
    nodes.insert(_vertices[subcord]);
  }
  return _meshTopo->getEntityIndex(subcdim, nodes);
}

vector<unsigned> Cell::getEntityIndices(unsigned subcdim) {
  int entityCount = _cellTopo->getSubcellCount(subcdim);
  vector<unsigned> cellEntityIndices(entityCount);
  for (int j=0; j<entityCount; j++) {
    unsigned entityIndex;
    set< unsigned > nodes;
    if (subcdim != 0) {
      int entityNodeCount = _cellTopo->getNodeCount(subcdim, j);
      for (int node=0; node<entityNodeCount; node++) {
        unsigned nodeIndexInCell = _cellTopo->getNodeMap(subcdim, j, node);
        nodes.insert(_vertices[nodeIndexInCell]);
      }
    } else {
      nodes.insert(_vertices[j]);
    }
    
    entityIndex = _meshTopo->getEntityIndex(subcdim, nodes);
    cellEntityIndices[j] = entityIndex;
  }
  return cellEntityIndices;
}

Teuchos::RCP<Cell> Cell::getParent() {
  return _parent;
}

void Cell::setParent(Teuchos::RCP<Cell> parent) {
  _parent = parent;
}

bool Cell::isParent() { return _children.size() > 0; }

RefinementPatternPtr Cell::refinementPattern() {
  return _refPattern;
}
void Cell::setRefinementPattern(RefinementPatternPtr refPattern) {
  _refPattern = refPattern;
}

CellTopoPtr Cell::topology() {
  return _cellTopo;
}

pair<GlobalIndexType, unsigned> Cell::getNeighbor(unsigned sideOrdinal) {
  if (sideOrdinal >= _cellTopo->getSideCount()) {
    cout << "sideOrdinal " << sideOrdinal << " >= sideCount " << _cellTopo->getSideCount() << endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideOrdinal must be less than sideCount!");
  }
  return _neighbors[sideOrdinal];
}
void Cell::setNeighbor(unsigned sideOrdinal, GlobalIndexType neighborCellIndex, unsigned neighborSideOrdinal) {
  if (neighborCellIndex == _cellIndex) {
    cout << "ERROR: neighborCellIndex == _cellIndex.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: neighborCellIndex == _cellIndex.\n");
  }
  if (sideOrdinal >= _cellTopo->getSideCount()) {
    cout << "sideOrdinal " << sideOrdinal << " >= sideCount " << _cellTopo->getSideCount() << endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideOrdinal must be less than sideCount!");
  }
  _neighbors[sideOrdinal] = make_pair(neighborCellIndex, neighborSideOrdinal);
}

const vector< unsigned > & Cell::vertices() {return _vertices;}