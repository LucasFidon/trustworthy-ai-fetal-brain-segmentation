# Install script for directory: /home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/lucasf/workspace/niftyreg_stable/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_maths.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_maths.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_maths_eigen.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_tools.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_tools.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_globalTrans.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_globalTrans.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_localTrans.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_localTrans.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_splineBasis.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_localTrans_regul.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_localTrans_jac.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_measure.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_measure.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_nmi.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_ssd.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_kld.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_lncc.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_dti.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_mind.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_resampling.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_resampling.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_blockMatching.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_blockMatching.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_femTrans.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_femTrans.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_aladin.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_macros.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/_reg_aladin.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/_reg_aladin_sym.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/_reg_aladin.cpp"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/_reg_aladin_sym.cpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/AladinContent.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/Platform.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/Kernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/AffineDeformationFieldKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/BlockMatchingKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/ConvolutionKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/OptimiseKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/ResampleImageKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/CPUAffineDeformationFieldKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/CPUBlockMatchingKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/CPUConvolutionKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/CPUOptimiseKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/CPUResampleImageKernel.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/KernelFactory.h"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/CPUKernelFactory.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/lucasf/workspace/niftyreg_stable/build/reg-lib/lib_reg_f3d.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/_reg_base.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/_reg_f3d.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/_reg_f3d2.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/_reg_f3d_sym.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_optimiser.cpp"
    "/home/lucasf/workspace/niftyreg_stable/niftyreg/reg-lib/cpu/_reg_optimiser.h"
    )
endif()

