param(
    [string]$RepoName = "nnunet-parse-rk3588",
    [string]$Owner = "",
    [string]$Tag = "v0.1.0"
)

$ErrorActionPreference = "Stop"

$StageRoot = $PSScriptRoot
$RepoRoot = Join-Path $StageRoot "repo"
$Artifact = Join-Path $StageRoot "artifacts\nnunet_parse_rk3588_full_package.zip"
$ReleaseNotes = Join-Path $RepoRoot "RELEASE_NOTES.md"

if (-not (Test-Path -LiteralPath $RepoRoot)) {
    throw "Local repo folder not found: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $Artifact)) {
    throw "Release artifact not found: $Artifact"
}

gh auth status | Out-Host

if ([string]::IsNullOrWhiteSpace($Owner)) {
    $Owner = (gh api user -q ".login").Trim()
}

$RepoSpec = $RepoName
if (-not [string]::IsNullOrWhiteSpace($Owner)) {
    $RepoSpec = "$Owner/$RepoName"
}

Push-Location $RepoRoot
try {
    $existingRemote = ""
    try {
        $existingRemote = git remote get-url origin 2>$null
    }
    catch {
        $existingRemote = ""
    }
    if ([string]::IsNullOrWhiteSpace($existingRemote)) {
        gh repo create $RepoSpec --private --source . --remote origin --push
    } else {
        git push -u origin main
    }

    gh release create $Tag $Artifact `
        --repo $RepoSpec `
        --title "nnUNet PARSE RK3588 package $Tag" `
        --notes-file $ReleaseNotes

    Write-Host "Done. Private repository: https://github.com/$RepoSpec"
    Write-Host "Release asset: $Artifact"
}
finally {
    Pop-Location
}
