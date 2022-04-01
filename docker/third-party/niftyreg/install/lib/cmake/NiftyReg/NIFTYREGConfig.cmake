# Author: Richard Brown
# Copyright 2019 University College London
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# This file sets NIFTYREG_INCLUDE_DIRS, NIFTYREG_LIBRARY_DIRS and NIFTYREG_LIBRARIES.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was NIFTYREGConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

#################################################################################### 

# add folder where this file resides to the cmake path such that it can use our find_package modules and .cmake files
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR};${CMAKE_MODULE_PATH}")

set_and_check(NIFTYREG_INCLUDE_DIRS "/home/lucasf/workspace/niftyreg_stable/install/include")
set_and_check(NIFTYREG_LIBRARY_DIRS "/home/lucasf/workspace/niftyreg_stable/install/lib")
set(NIFTYREG_LIBRARIES "reg_nifti;z;reg_png;_reg_ReadWriteImage;_reg_maths;_reg_tools;_reg_globalTrans;_reg_localTrans;_reg_measure;_reg_resampling;_reg_blockMatching;_reg_femTrans;_reg_aladin;_reg_f3d")

# NIFTYREG built with various components?
# CUDA
if (OFF)
  set(NIFTYREG_BUILT_WITH_CUDA TRUE)
  mark_as_advanced(NIFTYREG_BUILT_WITH_CUDA)
endif()
# NII_NAN
if (ON)
  set(NIFTYREG_BUILT_WITH_NII_NAN TRUE)
  mark_as_advanced(NIFTYREG_BUILT_WITH_NII_NAN)
endif()
# NRRD
if (OFF)
  set(NIFTYREG_BUILT_WITH_NRRD TRUE)
  mark_as_advanced(NIFTYREG_BUILT_WITH_NRRD)
endif()
# OPENCL
if (OFF)
  set(NIFTYREG_BUILT_WITH_OPENCL TRUE)
  mark_as_advanced(NIFTYREG_BUILT_WITH_OPENCL)
endif()
# OPENMP
if (ON)
  set(NIFTYREG_BUILT_WITH_OPENMP TRUE)
  mark_as_advanced(NIFTYREG_BUILT_WITH_OPENMP)
endif()
# SSE
if (ON)
  set(NIFTYREG_BUILT_WITH_SSE TRUE)
  mark_as_advanced(NIFTYREG_BUILT_WITH_SSE)
endif()
# THROW_EXCEP
if (OFF)
  set(NIFTYREG_BUILT_WITH_THROW_EXCEP TRUE)
  mark_as_advanced(NIFTYREG_BUILT_WITH_THROW_EXCEP)
endif()
