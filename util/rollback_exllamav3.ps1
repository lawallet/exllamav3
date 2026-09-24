$ErrorActionPreference = 'Stop'
# Original wheel URL and SHA256 were read from the installed direct_url.json.
$wheelUrl = 'https://github.com/turboderp-org/exllamav3/releases/download/v1.5.1/exllamav3-1.5.1+cu128.torch2.9.0-cp313-cp313-win_amd64.whl'
$expectedHash = 'ac338afd706e071109c24d845f60c438fc36235420927ba0dc6e0985d2ca8ada'
$wheelName = 'exllamav3-1.5.1+cu128.torch2.9.0-cp313-cp313-win_amd64.whl'
$wheelPath = Join-Path (Split-Path $PSScriptRoot -Parent) "native-results\rollback\$wheelName"
if (-not (Test-Path -LiteralPath $wheelPath)) {
    $wheelPath = Join-Path $env:TEMP $wheelName
    Invoke-WebRequest -Uri $wheelUrl -OutFile $wheelPath
}
if ((Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash.ToLower() -ne $expectedHash) {
    throw 'Original wheel hash mismatch. Environment has not been modified.'
}
& D:\tabbyAPI\venv\Scripts\python.exe -m pip install --no-deps --force-reinstall $wheelPath
if ($LASTEXITCODE -ne 0) { throw 'Rollback installation failed.' }
# Avoid importing the experimental checkout when checking the restored package.
Push-Location $env:TEMP
try {
    & D:\tabbyAPI\venv\Scripts\python.exe -c 'import exllamav3, torch; print(exllamav3.__file__); print(torch.__version__, torch.version.cuda)'
    if ($LASTEXITCODE -ne 0) { throw 'Rollback import verification failed.' }
} finally { Pop-Location }
