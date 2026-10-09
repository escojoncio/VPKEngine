# Translates a PS4 game dump (eboot.bin and sce_module\*.prx/*.sprx) for VPEngine's AOT guest CPU.
#
#   .\translate_game.ps1 -Game "A:\Games\CUSA03173" -Out "C:\vpengine-out" [-Vpaot .\vpaot.exe] [-Missing vpengine_missing.txt]
#
# Writes one set of C files per module (NAME_NNN.c, NAME.h, NAME_files.txt), a coverage report per
# module (NAME.json), and vpengine_registry.c. The C is derived from the game: it stays on this PC
# (or a private build), never in a public repository. -Missing: the file the engine writes on the
# headset (Documents\vpengine_missing.txt) with entries the previous translation lacked.
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Vpaot = (Join-Path $PSScriptRoot "vpaot.exe"),
    [string]$Missing = "",
    [int]$Split = 4000
)
$ErrorActionPreference = "Stop"
# -LiteralPath everywhere: dump folders are often named like "CUSA03173 [Astro Bot]", and
# brackets are wildcards for -Path.
if (-not (Test-Path -LiteralPath $Vpaot)) { $Vpaot = "vpaot.exe" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$files = @()
$eboot = Join-Path $Game "eboot.bin"
if (Test-Path -LiteralPath $eboot) { $files += Get-Item -LiteralPath $eboot }
$mods = Join-Path $Game "sce_module"
if (Test-Path -LiteralPath $mods) { $files += Get-ChildItem -LiteralPath $mods -File | Where-Object { $_.Extension -in ".prx", ".sprx" } }
if ($files.Count -eq 0) { throw "no eboot.bin or sce_module\*.prx in $Game" }
$names = @()
foreach ($f in $files) {
    $name = ($f.BaseName.ToLower() -replace '[^a-z0-9_]', '_')
    if ($name -match '^[0-9]') { $name = "m_$name" }
    if ($f.Extension -ne ".bin") { $name = "${name}_prx" }
    if ($names -contains $name) { throw "two modules map to the name $name (rename one of them)" }
    # The previous translation of this module goes first: a module that changes how many files it
    # is split into would otherwise leave stale files behind.
    Get-ChildItem -LiteralPath $Out -File | Where-Object { $_.Name -match "^$([regex]::Escape($name))(_\d+)?\.c$|^$([regex]::Escape($name))(_decl\.h|\.h|_files\.txt|\.json)$" } | Remove-Item
    $vpArgs = @("--elf", $f.FullName, "--pic", "--module", $name, "--split", $Split,
                "--out", (Join-Path $Out "$name.c"), "--stats", (Join-Path $Out "$name.json"))
    if ($Missing -and (Test-Path -LiteralPath $Missing)) { $vpArgs += @("--roots", $Missing) }
    Write-Host "== $($f.Name) -> $name"
    & $Vpaot @vpArgs
    if ($LASTEXITCODE -ne 0) { throw "vpaot failed on $($f.Name)" }
    $report = Get-Content -LiteralPath (Join-Path $Out "$name.json") -Raw | ConvertFrom-Json
    Write-Host ("   {0} functions, {1} instructions, {2:P2} supported" -f $report.functions, $report.instructions, $report.supported_fraction)
    $names += $name
}
& $Vpaot --registry (Join-Path $Out "vpengine_registry.c") @names
if ($LASTEXITCODE -ne 0) { throw "vpaot --registry failed" }
Write-Host "Done: $($names.Count) modules in $Out"
