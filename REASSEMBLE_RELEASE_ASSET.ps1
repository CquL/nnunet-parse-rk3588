param(
    [string]$ChunkDirectory = ".",
    [string]$Output = "nnunet_parse_rk3588_full_package.zip"
)

$ErrorActionPreference = "Stop"

$chunks = Get-ChildItem -LiteralPath $ChunkDirectory -Filter "nnunet_parse_rk3588_full_package.zip.part*" |
    Sort-Object Name

if ($chunks.Count -eq 0) {
    throw "No chunk files found in $ChunkDirectory"
}

$outputPath = Join-Path $ChunkDirectory $Output
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}

$target = [System.IO.File]::Open($outputPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try {
    foreach ($chunk in $chunks) {
        Write-Host "Adding $($chunk.Name)"
        $source = [System.IO.File]::OpenRead($chunk.FullName)
        try {
            $source.CopyTo($target)
        }
        finally {
            $source.Dispose()
        }
    }
}
finally {
    $target.Dispose()
}

Write-Host "Reassembled: $outputPath"
Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath
