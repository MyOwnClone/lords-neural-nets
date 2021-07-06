add_library (lib layer.c matrix.c network.c utils.c dataset.h dataset.c metrics.h train_opts.h train_log_opts.h ../just_build_all/proxy.c activations.h activations.c cost.h cost.c)
set_property(TARGET lib PROPERTY C_STANDARD 11)

target_link_libraries(lib m)
target_include_directories (lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})