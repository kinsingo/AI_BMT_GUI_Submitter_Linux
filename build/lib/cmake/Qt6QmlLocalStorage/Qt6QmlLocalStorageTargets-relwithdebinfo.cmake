#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Qt6::QmlLocalStorage" for configuration "RelWithDebInfo"
set_property(TARGET Qt6::QmlLocalStorage APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(Qt6::QmlLocalStorage PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELWITHDEBINFO "Qt6::Qml"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libQt6QmlLocalStorage.so.6.7.3"
  IMPORTED_SONAME_RELWITHDEBINFO "libQt6QmlLocalStorage.so.6"
  )

list(APPEND _cmake_import_check_targets Qt6::QmlLocalStorage )
list(APPEND _cmake_import_check_files_for_Qt6::QmlLocalStorage "${_IMPORT_PREFIX}/lib/libQt6QmlLocalStorage.so.6.7.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
