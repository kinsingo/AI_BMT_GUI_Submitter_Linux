# Copyright (C) 2024 The Qt Company Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# Find "ModuleTools" dependencies, which are other ModuleTools packages.
set(Qt6WaylandScannerTools_FOUND FALSE)
set(__qt_WaylandScannerTools_tool_deps "WaylandScanner\;")
foreach(__qt_WaylandScannerTools_target_dep ${__qt_WaylandScannerTools_tool_deps})
    list(GET __qt_WaylandScannerTools_target_dep 0 __qt_WaylandScannerTools_pkg)
    list(GET __qt_WaylandScannerTools_target_dep 1 __qt_WaylandScannerTools_version)

    if (NOT ${__qt_WaylandScannerTools_pkg}_FOUND)
        find_dependency(${__qt_WaylandScannerTools_pkg} ${__qt_WaylandScannerTools_version})
    endif()
endforeach()

set(Qt6WaylandScannerTools_FOUND TRUE)
