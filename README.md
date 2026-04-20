# Minecraft Vulkan Renderer
Minecraft Vulkan Renderer (MCVR) is a C++ implemented render framework for minecraft. It should be used with the Radiance mod.



# Build

First, make sure submodules are cloned with git.

```
git submodule update --init --recursive
```

## Linux

Use `cmake` to configure the project.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJAVA_PROJECT_ROOT_DIR=${PATH_TO_RADIANCE_JAVA_PROJECT} -DUSE_AMD=ON -DMCVR_ENABLE_NRD=ON
```

Build and install.

```
cmake --build build -j
cmake --install build
```

## Windows

Use `cmake` to configure the project.

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DJAVA_PROJECT_ROOT_DIR=${PATH_TO_RADIANCE_JAVA_PROJECT} -DMCVR_ENABLE_NRD=ON -DUSE_AMD=ON
```

Build and install.

```
cmake --build build -j --config Release
cmake --install build --config Release
```

# Fork Status

## Implemented / actively used in this fork

1. Sun-elevation-based celestial light gating
2. Volumetric clouds and cloud shadows
3. Ray-traced rain droplets
4. Ray-traced volumetric fog
5. HDR

## Not successfully implemented

1. SER (Shader Execution Reordering): multiple attempts were made, including enabling primary-hit SER and fallback experiments, but stability issues and random all-zero returns prevented it from becoming a shippable feature in this fork.

## Statement

All changes currently carried by this fork were produced through a GitHub Copilot vibe-coding workflow, including exploration, implementation, debugging, rollback, and final integration.
