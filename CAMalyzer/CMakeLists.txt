cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(CAMalyzer)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/CAMalyzer")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Benjamin Rodriguez  (B3MAT)")
set(EXTENSION_DESCRIPTION "This is extension allows to segment automicly 3T MRI images from the hip joint, creating a 3D Model from this segmentation, and analyze the cam morpholofy presence.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/CAMalyzer.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/CAMalyzer/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CAMalyzer)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
