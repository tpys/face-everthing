set(SOURCES tracker.cpp)
add_library(tracker ${SOURCES})
target_link_libraries(tracker alignment_lib ${OpenCV_LIBS})
