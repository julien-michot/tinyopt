set(TINYOPT_CPP_EXAMPLES
    finance
    computer_vision
    biology
    astronomy
    physics
    robotics
    signal_processing)

foreach(example IN LISTS TINYOPT_CPP_EXAMPLES)
  add_executable(tinyopt_example_${example} ${example}.cpp)
  target_link_libraries(tinyopt_example_${example} PRIVATE tinyopt)
endforeach()