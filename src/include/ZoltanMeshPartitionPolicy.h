// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  MeshPartitionPolicy.h
//  Camellia
//
//  Created by Nathan Roberts on 11/18/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#ifndef Zoltan_MeshPartitionPolicy_h
#define Zoltan_MeshPartitionPolicy_h

#include "MeshPartitionPolicy.h"
#include "TypeDefs.h"
#include <zoltan_cpp.h>
#include <string>

using namespace std;

namespace Camellia
{
class ZoltanMeshPartitionPolicy : public MeshPartitionPolicy
{
private:
  string _ZoltanPartitioner; // default to block
  string _debug_level;

  struct MigrationData
  {
    Mesh* mesh;
    vector<pair<GlobalIndexType,PartitionIndexType>> migratedCells; // cellID, owner pairs
  };
  
  //helper functions for query functions
  //  int getNextActiveIndex(Intrepid::FieldContainer<int> &partitionedActiveCells);
  //  static GlobalIndexType getIndexOfGID(int myNode, Intrepid::FieldContainer<GlobalIndexType> &partitionedActiveCells,GlobalIndexType globalID);
  //  vector<GlobalIndexType> getListOfActiveGlobalIDs(Intrepid::FieldContainer<GlobalIndexType> partitionedActiveCells);

  static set<GlobalIndexType> getRankLocalCellIDs(Mesh* mesh);

  //Zoltan query functions
  static int get_number_of_objects(void *data, int *ierr);
  static void get_object_list(void *data, int sizeGID, int sizeLID,ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,int wgt_dim, float *obj_wgts, int *ierr);
  static int get_num_geom(void *data, int *ierr);
  static void get_geom_list(void *data,int num_gid_entries, int num_lid_entries, int num_obj,ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec, int *ierr);

  //Reftree refinement pattern query functions
  //  static int get_num_coarse_elem(void *data, int *ierr);
  //  static void get_coarse_elem_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, int *in_order, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr);
  //  static int get_num_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);
  //  static void get_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR parent_gid, ZOLTAN_ID_PTR parent_lid, ZOLTAN_ID_PTR child_gids, ZOLTAN_ID_PTR child_lids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, ZOLTAN_REF_TYPE *ref_type, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr);
  //  static void get_child_weight(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim, float *obj_wgt, int *ierr);

  // data packing/unpacking:
  static int get_elem_data_size(void *data,
                                int num_gid_entries,
                                int num_lid_entries,
                                ZOLTAN_ID_PTR global_id,
                                ZOLTAN_ID_PTR local_id,
                                int *ierr); // returns size in bytes
  static void pack_elem_data(void *data,
                             int num_gid_entries,
                             int num_lid_entries,
                             ZOLTAN_ID_PTR global_id,
                             ZOLTAN_ID_PTR local_id,
                             int dest,
                             int size,
                             char *buf,
                             int *ierr);
  static void unpack_elem_data(void *data,
                               int num_gid_entries,
                               ZOLTAN_ID_PTR global_id,
                               int size,
                               char *buf,
                               int *ierr);
public:
  ZoltanMeshPartitionPolicy(Epetra_CommPtr Comm);
  ZoltanMeshPartitionPolicy(Epetra_CommPtr Comm, string partitionerName);
  virtual void partitionMesh(Mesh *mesh, PartitionIndexType numPartitions);
};
}

#endif
