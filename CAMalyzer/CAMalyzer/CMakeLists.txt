#-----------------------------------------------------------------------------
set(MODULE_NAME CAMalyzer)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  Lib/camalyzer_core/__init__.py
  Lib/camalyzer_core/config.py
  Lib/camalyzer_core/params.py
  Lib/camalyzer_core/deps.py
  Lib/camalyzer_core/io.py
  Lib/camalyzer_core/inference.py
  Lib/camalyzer_core/vtkconv.py
  Lib/camalyzer_core/profiling.py
  Lib/camalyzer_core/post/__init__.py
  Lib/camalyzer_core/post/surface.py
  Lib/camalyzer_core/post/cluster.py
  Lib/camalyzer_core/post/poisson.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
