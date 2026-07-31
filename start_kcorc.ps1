$ErrorActionPreference = "Stop"

# Always work in the directory containing this script
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "=========================================="
Write-Host " KCORC Summer School - environment setup"
Write-Host "=========================================="
Write-Host ""

# Install uv locally inside the course directory.
# No administrator rights and no PATH modification are required.
$uvDirectory = Join-Path $PSScriptRoot ".tools\uv"
$uvExecutable = Join-Path $uvDirectory "uv.exe"

if (-not (Test-Path $uvExecutable)) {
    Write-Host "uv was not found. Installing uv..."
    Write-Host ""

    New-Item -ItemType Directory -Force -Path $uvDirectory | Out-Null

    $env:UV_UNMANAGED_INSTALL = $uvDirectory

    # Pin the uv version used for the course.
    Invoke-RestMethod "https://astral.sh/uv/0.12.1/install.ps1" |
        Invoke-Expression
}

if (-not (Test-Path $uvExecutable)) {
    throw "uv installation failed. Expected file: $uvExecutable"
}

Write-Host ""
Write-Host "Using:"
& $uvExecutable --version

Write-Host ""
Write-Host "Creating and synchronizing the Python environment..."
Write-Host "The first run downloads Python and required packages."
Write-Host ""

& $uvExecutable sync --locked

if ($LASTEXITCODE -ne 0) {
    throw "Project environment synchronization failed."
}

Write-Host ""
Write-Host "Starting JupyterLab..."
Write-Host "Do not close this window while using the notebooks."
Write-Host ""

& $uvExecutable run --locked jupyter lab

if ($LASTEXITCODE -ne 0) {
    throw "JupyterLab failed to start."
}