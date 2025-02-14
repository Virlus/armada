//==============================================================================
/*
    Software License Agreement (BSD License)
    Copyright (c) 2003-2016, CHAI3D.
    (www.chai3d.org)

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

    * Neither the name of CHAI3D nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

*/
//==============================================================================

//------------------------------------------------------------------------------
#ifndef CGlobalsH
#define CGlobalsH
//------------------------------------------------------------------------------

//==============================================================================
/*!
    \file       CGlobals.h
    \ingroup    system

    \brief  
    Implements option settings for CHAI3D.

    \details

    __CGlobal.h__ also contains a list of compiler settings that control the 
    features included in CHAI3D. Default settings are provided for
    __Windows__, __Linux__ and __Mac OS-X__ operating systems. \n\n

    CHAI3D compiling options are the following:\n

    - __C_USE_OPEN_GL__: Enable or disable support for OpenGL. Disabling OpenGL
    allows you to compile CHAI3D on embedded real-time operating systems for 
    instance, or with applications that may use a different graphics library 
    (i.e. DirectX). 
    
    - __C_USE_EIGEN__: Enable or disable support of external Eigen libraries.
                     If Eigen is disabled, CHAI3D uses its own internal 
                     matrix library and representation.

    - __C_USE_FILE_3DS__: Enable of disable external support for 3DS files.\n
    - __C_USE_FILE_GIF__: Enable of disable external support for GIF files.\n
    - __C_USE_FILE_JPG__: Enable of disable external support for JPG files.\n
    - __C_USE_FILE_PNG__: Enable of disable external support for PNG files.\n
                        
    Disabling one or more features will reduce the overall capabilities of 
    CHAI3D and may affect some of the examples provided with the framework.
    
    For more information about the different compilation settings, please
    review the inline comments in the __CGlobal.h__ header file.
*/
//==============================================================================

//------------------------------------------------------------------------------
#ifndef DOXYGEN_SHOULD_SKIP_THIS
//------------------------------------------------------------------------------

//==============================================================================
// GENERAL SETTINGS
//==============================================================================

//------------------------------------------------------------------------------
// STANDARD LIBRARIES
//------------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>
#include <memory>

//==============================================================================
// OPERATING SYSTEM SPECIFIC
//==============================================================================

//------------------------------------------------------------------------------
// WIN32 / WIN64 OS
//------------------------------------------------------------------------------
#if defined(WIN32) | defined(WIN64)

    //--------------------------------------------------------------------
    // GENERAL
    //--------------------------------------------------------------------

    // OS specific
    #if !defined(NOMINMAX)
        #define NOMINMAX
    #endif
    #include <windows.h>

	// if building with .NET, we need to explicitly tell MSVC that we are using unmanaged code
    // http://msdn.microsoft.com/en-us/library/4ex65770(v=vs.110).aspx
    #if (_MANAGED == 1)
      #define nullptr __nullptr
    #endif

    // printf
    #include <conio.h>
    #define cPrint _cprintf

    // math
    #if !defined(C_USE_EIGEN)
        #define _USE_MATH_DEFINES
        #include <cmath>
    #endif

    // OpenGL GLEW library
    #if defined(C_USE_OPENGL)
        #ifndef GLEW_STATIC
        #define GLEW_STATIC
        #endif
    #endif

    //--------------------------------------------------------------------
    // HAPTIC DEVICES
    //--------------------------------------------------------------------
    #define C_ENABLE_CUSTOM_DEVICE_SUPPORT
    #define C_ENABLE_DELTA_DEVICE_SUPPORT
    #define C_ENABLE_PHANTOM_DEVICE_SUPPORT
    // #define C_ENABLE_SIXENSE_DEVICE_SUPPORT

    //--------------------------------------------------------------------
    // SYSTEM LIBRARIES
    //--------------------------------------------------------------------
    #pragma comment (lib, "winmm.lib")

#endif


//------------------------------------------------------------------------------
// LINUX OS
//------------------------------------------------------------------------------
#if defined(LINUX)

    //--------------------------------------------------------------------
    // GENERAL
    //--------------------------------------------------------------------

    // OS specific
    #include <ctime>
    #include <pthread.h>
    #include <dlfcn.h>

    // printf
    #define cPrint printf

    //--------------------------------------------------------------------
    // HAPTIC DEVICES
    //--------------------------------------------------------------------
    #define C_ENABLE_CUSTOM_DEVICE_SUPPORT
    #define C_ENABLE_SIXENSE_DEVICE_SUPPORT
    #define C_ENABLE_DELTA_DEVICE_SUPPORT
    #define C_ENABLE_PHANTOM_DEVICE_SUPPORT

#endif


//------------------------------------------------------------------------------
// MAC OS
//------------------------------------------------------------------------------
#if defined(MACOSX)

    //--------------------------------------------------------------------
    // GENERAL
    //--------------------------------------------------------------------

    // OS specific
    #include <mach/mach_time.h>
    #include <pthread.h>
    #include <dlfcn.h>

    // printf
    #define cPrint printf

    //--------------------------------------------------------------------
    // HAPTIC DEVICES
    //--------------------------------------------------------------------
    #define C_ENABLE_CUSTOM_DEVICE_SUPPORT
    #define C_ENABLE_DELTA_DEVICE_SUPPORT

#endif


//------------------------------------------------------------------------------
#endif  // DOXYGEN_SHOULD_SKIP_THIS
//------------------------------------------------------------------------------


//==============================================================================
// GENERAL PURPOSE FUNCTIONS:
//==============================================================================

//------------------------------------------------------------------------------
namespace chai3d {
//------------------------------------------------------------------------------

//! This function suspends the execution of the current thread for a specified interval.
void cSleepMs(const unsigned int a_interval);

//------------------------------------------------------------------------------
} // namespace chai3d
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif
//------------------------------------------------------------------------------
