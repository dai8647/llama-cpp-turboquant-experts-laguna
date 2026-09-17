# ROCm 7.2 local toolchain (extracted from repo.radeon.com wheels)
$core = "C:\Users\dai86\rocm-sdk-core\_rocm_sdk_core"
$env:ROCM_PATH = $core
$env:HIP_PATH = $core
$env:HIP_PLATFORM = "amd"
$env:HIP_COMPILER = "clang"
$env:HCC_AMDGPU_TARGET = "gfx1101"
$env:PATH = "$core\bin;$core\lib\llvm\bin;C:\Program\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;" + $env:PATH
$devlib = "$core/lib/llvm/amdgcn/bitcode"
$rc = "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/rc.exe"
Write-Host "ROCm ready: $core"
