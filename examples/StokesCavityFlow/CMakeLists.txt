project(StokesCavityFlowExamples)

add_executable(StokesCavityFlowExample StokesCavityFlowExample.cpp)
target_link_libraries(StokesCavityFlowExample 
  Camellia
)

add_executable(StokesCavityFlowSpaceTimeExample StokesCavityFlowSpaceTimeExample.cpp)
target_link_libraries(StokesCavityFlowSpaceTimeExample
  Camellia
)

