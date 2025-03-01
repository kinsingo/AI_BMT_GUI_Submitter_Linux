# Copyright (C) 2024 The Qt Company Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# TODO: Ideally this should look for each Qt module separately, with each module's specific version,
# bypassing the Qt6 Config file, aka find_package(Qt6SpecificFoo) repated x times. But it's not
# critical.
find_package(Qt6 6.7.3
             COMPONENTS QmlToolingSettingsPrivate QmlTypeRegistrarPrivate QmlIntegration QmlBuiltins Qml QmlModels QmlCore QmlNetwork QmlWorkerScript QmlLocalStorage QmlXmlListModel Quick QuickLayouts QuickTest QuickTestUtilsPrivate QuickParticlesPrivate QuickEffectsPrivate QuickShapesPrivate QuickWidgets QuickTemplates2 QuickControls2Impl QuickControls2 QuickControls2BasicStyleImpl QuickControls2Basic QuickControls2FusionStyleImpl QuickControls2Fusion QuickControls2ImagineStyleImpl QuickControls2Imagine QuickControls2MaterialStyleImpl QuickControls2Material QuickControls2UniversalStyleImpl QuickControls2Universal QuickDialogs2Utils QuickDialogs2QuickImpl QuickDialogs2 QuickControlsTestUtilsPrivate LabsSettings LabsQmlModels LabsFolderListModel LabsAnimation LabsWavefrontMesh LabsSharedImage QmlCompiler QmlDomPrivate QmlLSPrivate PacketProtocolPrivate QmlDebugPrivate)
