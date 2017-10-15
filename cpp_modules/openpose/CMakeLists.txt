add_subdirectory(core)
add_subdirectory(utilities)
add_subdirectory(face)
set_target_properties(core utilities face PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})

