add_executable (tests test.c test_functions.c test_layer.c test_matrix.c test_network.c test_utils.c test_train_mnist.c utils.h utils.c test_train_xor.c)
set_property(TARGET tests PROPERTY C_STANDARD 11)
set_property(TARGET tests PROPERTY ENABLE_EXPORTS 1)
target_link_libraries(tests m lnn)