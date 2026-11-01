set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_ARCHITECTURES arm64)
# zed-mac: Apple clang 21 adds [[nodiscard]] on std::future::get(); Pangolin builds with -Werror
set(VCPKG_C_FLAGS "${VCPKG_C_FLAGS} -Wno-error=unused-result")
set(VCPKG_CXX_FLAGS "${VCPKG_CXX_FLAGS} -Wno-error=unused-result")
