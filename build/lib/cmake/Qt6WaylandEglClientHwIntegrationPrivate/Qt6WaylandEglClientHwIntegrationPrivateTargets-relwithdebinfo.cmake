#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Qt6::WaylandEglClientHwIntegrationPrivate" for configuration "RelWithDebInfo"
set_property(TARGET Qt6::WaylandEglClientHwIntegrationPrivate APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(Qt6::WaylandEglClientHwIntegrationPrivate PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libQt6WaylandEglClientHwIntegration.so.6.7.3"
  IMPORTED_SONAME_RELWITHDEBINFO "libQt6WaylandEglClientHwIntegration.so.6"
  )

list(APPEND _cmake_import_check_targets Qt6::WaylandEglClientHwIntegrationPrivate )
list(APPEND _cmake_import_check_files_for_Qt6::WaylandEglClientHwIntegrationPrivate "${_IMPORT_PREFIX}/lib/libQt6WaylandEglClientHwIntegration.so.6.7.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
