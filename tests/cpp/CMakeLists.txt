# MIT License
#
# Copyright (c) 2025 Meher V.R. Malladi.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Minimal stand-alone C++ tests. We avoid pulling in a third-party test framework
# so the existing build invariants (no extra dependencies; works with the same
# CMake invocation that builds the core) are preserved.

add_executable(test_camera_residual test_camera_residual.cpp)
target_link_libraries(test_camera_residual PRIVATE rko_lio::core)
target_compile_features(test_camera_residual PRIVATE cxx_std_20)

if(COMMAND add_test)
  add_test(NAME camera_residual_recovery COMMAND test_camera_residual)
endif()
