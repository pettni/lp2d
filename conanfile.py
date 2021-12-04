from conans import ConanFile


class CppStarterProject(ConanFile):
    # Note: options are copied from CMake boolean options.
    # When turned off, CMake sometimes passes them as empty strings.
    options = {
    }
    name = "lp2d"
    version = "0.1"
    requires = (
        "catch2/2.13.7",
    )
    generators = "cmake", "gcc", "txt", "cmake_find_package"

    def requirements(self):
      pass

