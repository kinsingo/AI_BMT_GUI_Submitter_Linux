cmake_minimum_required(VERSION 3.14)

project(AI_BMT_GUI_Submitter_App VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_SOURCES main.cpp)

# Create the executable
add_executable(AI_BMT_GUI_Submitter ${PROJECT_SOURCES})

# Open CV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(AI_BMT_GUI_Submitter PRIVATE ${OpenCV_LIBS})


# Mobilint (maccel) — adjust paths as needed if the SDK is not installed in a standard location
# 1. Optionally add the include directory if maccel headers are not found automatically
#    target_include_directories(AI_BMT_GUI_Submitter PRIVATE /path/to/maccel/include)

# 2. Find the maccel library; you can specify search paths if necessary
find_library(MACCEL_LIBRARY NAMES maccel PATHS /usr/local/lib /usr/lib)
if(NOT MACCEL_LIBRARY)
    message(FATAL_ERROR "Could not find the maccel library")
endif()

# 3. Link it to your target (this is the equivalent of `-lmaccel`)
target_link_libraries(AI_BMT_GUI_Submitter PRIVATE ${MACCEL_LIBRARY})

# Add the directory where the headers are located
target_include_directories(AI_BMT_GUI_Submitter PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/include
)


# Link the libraries to the executable
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/build/libAI_BMT_GUI_Library.so)

# Set RPATH to include the lib directory during the build and install phases
set_target_properties(AI_BMT_GUI_Submitter PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Ensure RPATH is always used (not stripped out)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)