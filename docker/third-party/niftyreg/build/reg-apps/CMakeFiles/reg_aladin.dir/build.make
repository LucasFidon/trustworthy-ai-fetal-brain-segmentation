# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.14.5-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.14.5-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lucasf/workspace/niftyreg_stable/niftyreg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lucasf/workspace/niftyreg_stable/build

# Include any dependencies generated for this target.
include reg-apps/CMakeFiles/reg_aladin.dir/depend.make

# Include the progress variables for this target.
include reg-apps/CMakeFiles/reg_aladin.dir/progress.make

# Include the compile flags for this target's objects.
include reg-apps/CMakeFiles/reg_aladin.dir/flags.make

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o: reg-apps/CMakeFiles/reg_aladin.dir/flags.make
reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o: /home/lucasf/workspace/niftyreg_stable/niftyreg/reg-apps/reg_aladin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucasf/workspace/niftyreg_stable/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o"
	cd /home/lucasf/workspace/niftyreg_stable/build/reg-apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o -c /home/lucasf/workspace/niftyreg_stable/niftyreg/reg-apps/reg_aladin.cpp

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reg_aladin.dir/reg_aladin.cpp.i"
	cd /home/lucasf/workspace/niftyreg_stable/build/reg-apps && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucasf/workspace/niftyreg_stable/niftyreg/reg-apps/reg_aladin.cpp > CMakeFiles/reg_aladin.dir/reg_aladin.cpp.i

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reg_aladin.dir/reg_aladin.cpp.s"
	cd /home/lucasf/workspace/niftyreg_stable/build/reg-apps && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucasf/workspace/niftyreg_stable/niftyreg/reg-apps/reg_aladin.cpp -o CMakeFiles/reg_aladin.dir/reg_aladin.cpp.s

# Object files for target reg_aladin
reg_aladin_OBJECTS = \
"CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o"

# External object files for target reg_aladin
reg_aladin_EXTERNAL_OBJECTS =

reg-apps/reg_aladin: reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o
reg-apps/reg_aladin: reg-apps/CMakeFiles/reg_aladin.dir/build.make
reg-apps/reg_aladin: reg-lib/lib_reg_aladin.a
reg-apps/reg_aladin: reg-lib/lib_reg_localTrans.a
reg-apps/reg_aladin: reg-lib/lib_reg_blockMatching.a
reg-apps/reg_aladin: reg-lib/lib_reg_resampling.a
reg-apps/reg_aladin: reg-lib/lib_reg_globalTrans.a
reg-apps/reg_aladin: reg-io/lib_reg_ReadWriteImage.a
reg-apps/reg_aladin: reg-io/png/libreg_png.a
reg-apps/reg_aladin: reg-lib/lib_reg_tools.a
reg-apps/reg_aladin: reg-lib/lib_reg_maths.a
reg-apps/reg_aladin: reg-io/nifti/libreg_nifti.a
reg-apps/reg_aladin: /usr/lib/x86_64-linux-gnu/libpng.so
reg-apps/reg_aladin: reg-apps/CMakeFiles/reg_aladin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lucasf/workspace/niftyreg_stable/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reg_aladin"
	cd /home/lucasf/workspace/niftyreg_stable/build/reg-apps && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reg_aladin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
reg-apps/CMakeFiles/reg_aladin.dir/build: reg-apps/reg_aladin

.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/build

reg-apps/CMakeFiles/reg_aladin.dir/clean:
	cd /home/lucasf/workspace/niftyreg_stable/build/reg-apps && $(CMAKE_COMMAND) -P CMakeFiles/reg_aladin.dir/cmake_clean.cmake
.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/clean

reg-apps/CMakeFiles/reg_aladin.dir/depend:
	cd /home/lucasf/workspace/niftyreg_stable/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lucasf/workspace/niftyreg_stable/niftyreg /home/lucasf/workspace/niftyreg_stable/niftyreg/reg-apps /home/lucasf/workspace/niftyreg_stable/build /home/lucasf/workspace/niftyreg_stable/build/reg-apps /home/lucasf/workspace/niftyreg_stable/build/reg-apps/CMakeFiles/reg_aladin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/depend

