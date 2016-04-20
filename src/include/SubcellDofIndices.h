//
//  SubcellDofIndices.h
//  Camellia
//
//  Created by Nate Roberts on 4/18/16.
//
//

#ifndef Camellia_SubcellDofIndices_h
#define Camellia_SubcellDofIndices_h

#include <map>
#include <set>
#include <vector>

#include "Var.h"
#include "TypeDefs.h"

namespace Camellia {
  struct SubcellDofIndices
  {
    typedef std::map<int, std::vector<GlobalIndexType> > VarIDToDofIndices; // key: varID
    typedef std::map<unsigned, VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
    std::vector<SubCellOrdinalToMap> subcellDofIndices; // index to vector: subcell dimension
    
    std::set<GlobalIndexType> allDofIndices()
    {
      std::set<GlobalIndexType> dofIndices;
      for (auto dimMapIt = subcellDofIndices.begin(); dimMapIt != subcellDofIndices.end(); dimMapIt++)
      {
        for (auto scordMapIt = dimMapIt->begin(); scordMapIt != dimMapIt->end(); scordMapIt++)
        {
          for (auto varIDMapIt = scordMapIt->second.begin(); varIDMapIt != scordMapIt->second.end(); varIDMapIt++)
          {
            dofIndices.insert(varIDMapIt->second.begin(),varIDMapIt->second.end());
          }
        }
      }
      return dofIndices;
    }
    
    std::set<GlobalIndexType> dofIndicesForVariable(VarPtr var)
    {
      std::set<GlobalIndexType> dofIndices;
      for (auto dimMapIt = subcellDofIndices.begin(); dimMapIt != subcellDofIndices.end(); dimMapIt++)
      {
        for (auto scordMapIt = dimMapIt->begin(); scordMapIt != dimMapIt->end(); scordMapIt++)
        {
          if (scordMapIt->second.find(var->ID()) != scordMapIt->second.end())
          {
            std::vector<GlobalIndexType>* scordDofIndicesForVariable = &scordMapIt->second.find(var->ID())->second;
            dofIndices.insert(scordDofIndicesForVariable->begin(), scordDofIndicesForVariable->end());
          }
        }
      }
      return dofIndices;
    }
    
    int dataSize() const
    {
      int size = 0;
      int numDimensions = subcellDofIndices.size();
      size += sizeof(numDimensions);
      for (int d=0; d<numDimensions; d++)
      {
        int numSubcellEntries = subcellDofIndices[d].size();
        size += sizeof(numSubcellEntries);
        for (auto subcellEntry : subcellDofIndices[d])
        {
          int subcord = subcellEntry.first;
          size += sizeof(subcord);
          int numVarEntries = subcellEntry.second.size();
          size += sizeof(numVarEntries);
          for (auto varEntry : subcellEntry.second)
          {
            int varID = varEntry.first;
            size += sizeof(varID);
            int numGlobalDofIndices = varEntry.second.size();
            size += sizeof(numGlobalDofIndices);
            int vectorSize = numGlobalDofIndices*sizeof(GlobalIndexType);
            size += vectorSize;
          }
        }
      }
      return size;
    }
    
    void read(const char* &dataLocation, int size)
    {
      subcellDofIndices.clear();
      int numDimensions;
      memcpy(&numDimensions, dataLocation, sizeof(numDimensions));
      dataLocation += sizeof(numDimensions);
      subcellDofIndices.resize(numDimensions);
      for (int d=0; d<numDimensions; d++)
      {
        int numSubcellEntries;
        memcpy(&numSubcellEntries, dataLocation, sizeof(numSubcellEntries));
        dataLocation += sizeof(numSubcellEntries);
        for (int i=0; i<numSubcellEntries; i++)
        {
          int subcord;
          memcpy(&subcord, dataLocation, sizeof(subcord));
          dataLocation += sizeof(subcord);
          int numVarEntries;
          memcpy(&numVarEntries, dataLocation, sizeof(numVarEntries));
          dataLocation += sizeof(numVarEntries);
          for (int j=0; j<numVarEntries; j++)
          {
            int varID;
            memcpy(&varID, dataLocation, sizeof(varID));
            dataLocation += sizeof(varID);
            int numGlobalDofIndices;
            memcpy(&numGlobalDofIndices, dataLocation, sizeof(numGlobalDofIndices));
            dataLocation += sizeof(numGlobalDofIndices);
            std::vector<GlobalIndexType> globalDofIndices(numGlobalDofIndices);
            int vectorSize = numGlobalDofIndices*sizeof(GlobalIndexType);
            memcpy(&globalDofIndices[0], dataLocation, vectorSize);
            dataLocation += vectorSize;
            subcellDofIndices[d][subcord][varID] = globalDofIndices;
          }
        }
      }
    }
    
    void write(char* &dataLocation, int bufferSize) const
    {
      int numDimensions = subcellDofIndices.size();
      memcpy(dataLocation, &numDimensions, sizeof(numDimensions));
      dataLocation += sizeof(numDimensions);
      for (int d=0; d<numDimensions; d++)
      {
        int numSubcellEntries = subcellDofIndices[d].size();
        memcpy(dataLocation, &numSubcellEntries, sizeof(numSubcellEntries));
        dataLocation += sizeof(numSubcellEntries);
        for (auto subcellEntry : subcellDofIndices[d])
        {
          int subcord = subcellEntry.first;
          memcpy(dataLocation, &subcord, sizeof(subcord));
          dataLocation += sizeof(subcord);
          int numVarEntries = subcellEntry.second.size();
          memcpy(dataLocation, &numVarEntries, sizeof(numVarEntries));
          dataLocation += sizeof(numVarEntries);
          for (auto varEntry : subcellEntry.second)
          {
            int varID = varEntry.first;
            memcpy(dataLocation, &varID, sizeof(varID));
            dataLocation += sizeof(varID);
            int numGlobalDofIndices = varEntry.second.size();
            memcpy(dataLocation, &numGlobalDofIndices, sizeof(numGlobalDofIndices));
            dataLocation += sizeof(numGlobalDofIndices);
            int vectorSize = numGlobalDofIndices*sizeof(GlobalIndexType);
            memcpy(dataLocation, &varEntry.second[0], vectorSize);
            dataLocation += vectorSize;
          }
        }
      }
    }
  };
}
#endif