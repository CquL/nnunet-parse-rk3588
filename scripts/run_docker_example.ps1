$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$ImageName = "nnunet-parse"

Copy-Item -Force -LiteralPath (Join-Path $Root "Dockerfile.example") -Destination (Join-Path $Root "Dockerfile")

docker build -t $ImageName $Root

docker run --rm --gpus all `
    -v "${Root}:/workspace" `
    $ImageName
