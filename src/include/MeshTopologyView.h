//
//  MeshTopologyView.h
//  Camellia
//
//  Created by Nate Roberts on 6/23/15.
//
//

//! MeshTopologyView: a class that defines a minimal interface for MeshTopology objects used by Mesh, GlobalDofAssignment, and
//! subclasses of GlobalDofAssignment.

/*!
 \author Nathan V. Roberts, ALCF.
 
 \date Last modified on 23-June-2015.
 */


#ifndef Camellia_MeshTopologyView_h
#define Camellia_MeshTopologyView_h

#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"

namespace Camellia {
  
  class MeshTransformationFunction;

  class MeshTopologyView
  {
    MeshTopologyPtr _meshTopo; // null when subclass constructor is used
    std::set<IndexType> _allKnownCells; // empty when subclass constructor is used
    mutable std::set<GlobalIndexType> _ownedCellIndices; // depends on base MeshTopology's _ownedCellIndices.
    mutable int _ownedCellIndicesPruningOrdinal = -1; // what the pruningOrdinal was when _ownedCellIndices was last determined.  If _meshTopo->pruningOrdinal is different, we need to rebuild.
    
    IndexType _globalCellCount;
    IndexType _globalActiveCellCount;
    
    void buildLookups(); // _rootCellIndices and _ancestralCells
  protected:
    std::set<IndexType> _activeCells;
    std::set<IndexType> _rootCells; // filled during construction when meshTopoPtr is not null; otherwise responsibility belongs to subclass.
    GlobalDofAssignment* _gda; // for cubature degree lookups
    
    std::vector<IndexType> getActiveCellsForSide(IndexType sideEntityIndex);
  public:
    // ! Constructor for use by MeshTopology and any other subclasses
    MeshTopologyView();
    
    // ! Constructor that defines a view in terms of an existing MeshTopology and a set of cells selected to be active.
    MeshTopologyView(MeshTopologyPtr meshTopoPtr, const std::set<IndexType> &activeCellIDs);
    
    // ! Destructor
    virtual ~MeshTopologyView() {}
    
    // ! This method only gets within a factor of 2 or so, but can give a rough estimate
    virtual long long approximateMemoryFootprint();
    
    virtual std::vector<IndexType> cellIDsForPoints(const Intrepid::FieldContainer<double> &physicalPoints);
    virtual IndexType cellCount() const;
    
    // ! Returns the global active cell count.
    virtual IndexType activeCellCount() const;
    
    // ! If the base MeshTopology is distributed, returns the Comm object used.  Otherwise, returns Teuchos::null, which is meant to indicate that the MeshTopology is replicated on every MPI rank on which it is used.
    virtual Epetra_CommPtr Comm() const;
    
    // ! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs).  Not supported for MeshTopologyViews with _meshTopo defined (i.e. those that are themselves defined in terms of another MeshTopology object).
    virtual Teuchos::RCP<MeshTopology> deepCopy() const;
    
    virtual bool entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent);
    virtual bool entityIsGeneralizedAncestor(unsigned ancestorDimension, IndexType ancestor,
                                             unsigned descendentDimension, IndexType descendent);
    
    virtual IndexType getActiveCellCount(unsigned d, IndexType entityIndex);
    virtual const std::set<IndexType> &getLocallyKnownActiveCellIndices();
    virtual const std::set<IndexType> &getMyActiveCellIndices() const;
    virtual std::set<IndexType> getActiveCellIndicesForAncestorsOfMyCellsInBaseMeshTopology() const;
    virtual std::vector< std::pair<IndexType,unsigned> > getActiveCellIndices(unsigned d, IndexType entityIndex); // first entry in pair is the cellIndex, the second is the index of the entity in that cell (the subcord).
    
    virtual MeshTopology* baseMeshTopology();
    
    virtual CellPtr getCell(IndexType cellIndex) const;
    virtual std::vector<double> getCellCentroid(IndexType cellIndex);
    virtual std::set< std::pair<IndexType, unsigned> > getCellsContainingEntity(unsigned d, unsigned entityIndex);
    virtual std::vector<IndexType> getCellsForSide(IndexType sideEntityIndex);

    virtual std::pair<IndexType, unsigned> getConstrainingEntity(unsigned d, IndexType entityIndex);
    virtual IndexType getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex);
    virtual std::vector< std::pair<IndexType,unsigned> > getConstrainingSideAncestry(unsigned int sideEntityIndex);
    
    virtual unsigned getDimension() const;
    
    virtual std::vector<IndexType> getEntityVertexIndices(unsigned d, IndexType entityIndex);
    
    virtual const std::set<IndexType> &getRootCellIndicesLocal();
    
    virtual std::vector< IndexType > getSidesContainingEntity(unsigned d, IndexType entityIndex);
    
    virtual bool isDistributed() const;
    
    virtual bool isParent(IndexType cellIndex);
    
    virtual bool isValidCellIndex(IndexType cellIndex) const;
    
    virtual const std::vector<double>& getVertex(IndexType vertexIndex) const;
    
    virtual bool getVertexIndex(const std::vector<double> &vertex, IndexType &vertexIndex, double tol=1e-14);
    
    virtual std::vector<IndexType> getVertexIndicesMatching(const std::vector<double> &vertexInitialCoordinates, double tol=1e-14);

    virtual Intrepid::FieldContainer<double> physicalCellNodesForCell(unsigned cellIndex, bool includeCellDimension = false);
    
    virtual Teuchos::RCP<MeshTransformationFunction> transformationFunction();
    
    virtual std::pair<IndexType,IndexType> owningCellIndexForConstrainingEntity(unsigned d, unsigned constrainingEntityIndex);
    
    virtual void setGlobalDofAssignment(GlobalDofAssignment* gda); // for cubature degree lookups
    
    virtual void verticesForCell(Intrepid::FieldContainer<double>& vertices, IndexType cellID);
    
    virtual MeshTopologyViewPtr getView(const std::set<IndexType> &activeCellIndices);
    
    virtual MeshTopologyViewPtr getGatheredViewCopy() const; // all-to-all gather MeshTopology info, and create a new non-distributed copy on each rank.
    
    void printAllEntitiesInBaseMeshTopology();
    
    void printActiveCellAncestors();
    void printCellAncestors(IndexType cellIndex);
    
    // distributed read/write methods (for HDF5 support, e.g.)
    // ! returns the size, in bytes, of the serialization of this rank's view of the MeshTopologyView object.  Includes the base MeshTopology's serialization.  (While potentially inefficient, this makes exported MeshTopologyViews self-contained.)
    virtual int dataSize() const;
    // ! reads a distributed MeshTopologyView
    static MeshTopologyViewPtr read(Epetra_CommPtr comm, const char* &dataLocation, int size);
    // ! writes a distributed MeshTopologyView
    virtual void write(char* &dataLocation, int size) const;
  };

}
#endif
