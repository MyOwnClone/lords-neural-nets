add_library (lnn layer.c network.c matrix.c utils.c dataset.h dataset.c metrics.h train_opts.h train_log_opts.h ../just_build_all/proxy.c activations.h activations.c cost.h cost.c)
set_property(TARGET lnn PROPERTY C_STANDARD 11)

target_link_libraries(lnn m)
target_include_directories (lnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})