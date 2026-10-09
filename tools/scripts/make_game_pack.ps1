# Builds a VPEngine game pack: the game's x86-64 code (eboot.bin + sce_module\*.prx/*.sprx)
# translated and compiled for the Apple Vision Pro, as one file that goes into the game's folder
# in VPS4 on the headset. The app signs it with your SideStore certificate and loads it: no JIT.
#
#   .\make_game_pack.ps1 -Game "D:\PS4\CUSA12392"            (writes CUSA12392\vpengine.vpgame)
#   .\make_game_pack.ps1 -Game ... -Missing vpengine_missing.txt   (entries the headset logged)
#
# Needs LLVM (clang and lld) once:   winget install LLVM.LLVM
# Disk: each piece of C is deleted as soon as it is compiled; only the compiled pieces are kept
# (in -Work, for a fast rebuild with -Missing) and the pack itself. -Clean deletes -Work at the end.
# Nothing made from the game leaves this PC except the pack you copy to the headset.
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Out = "",
    [string]$Work = "",
    [string]$Missing = "",
    [string]$Title = "",
    [int]$Jobs = 0,
    [int]$Split = 300,
    [switch]$Clean
)
$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
$Game = (Resolve-Path -LiteralPath $Game).Path
if (-not $Out) { $Out = Join-Path $Game "vpengine.vpgame" }
if (-not $Title) { $Title = Split-Path -Leaf $Game }
if (-not $Work) { $Work = Join-Path $env:LOCALAPPDATA ("VPEngine\" + ($Title -replace '[^A-Za-z0-9_.-]', '_')) }
if ($Jobs -le 0) { $Jobs = [Math]::Max(1, [Environment]::ProcessorCount) }

# The tools: vpaot next to this script; clang and lld from LLVM.
$Vpaot = Join-Path $Here "vpaot.exe"
if (-not (Test-Path -LiteralPath $Vpaot)) { $Vpaot = "vpaot.exe" }
function Find-Tool($name) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($d in @("$env:ProgramFiles\LLVM\bin", "${env:ProgramFiles(x86)}\LLVM\bin")) {
        $p = Join-Path $d "$name.exe"
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}
$Clang = Find-Tool "clang"
$Lld = Find-Tool "lld"
if (-not $Clang -or -not $Lld) {
    throw "LLVM is not installed. Install it once with:  winget install LLVM.LLVM   (then open a new PowerShell)"
}
# The runtime headers and link stubs, as the release zip lays them out (or the repository).
$Runtime = Join-Path $Here "runtime"
if (-not (Test-Path -LiteralPath (Join-Path $Runtime "vp_emit.h"))) { $Runtime = Join-Path $Here "..\..\runtime" }
$Sdk = Join-Path $Here "sdk"
if (-not (Test-Path -LiteralPath (Join-Path $Sdk "libVPRuntime.tbd"))) { $Sdk = Join-Path $Here "..\sdk" }
foreach ($p in @((Join-Path $Runtime "vp_emit.h"), (Join-Path $Runtime "freestanding\math.h"), (Join-Path $Sdk "libVPRuntime.tbd"), (Join-Path $Sdk "libSystem.tbd"))) {
    if (-not (Test-Path -LiteralPath $p)) { throw "missing $p (unpack the whole VPEngine-PC.zip)" }
}

New-Item -ItemType Directory -Force -Path $Work | Out-Null
$files = @()
$eboot = Join-Path $Game "eboot.bin"
if (Test-Path -LiteralPath $eboot) { $files += Get-Item -LiteralPath $eboot }
$mods = Join-Path $Game "sce_module"
if (Test-Path -LiteralPath $mods) { $files += Get-ChildItem -LiteralPath $mods -File | Where-Object { $_.Extension -in ".prx", ".sprx" } }
if ($files.Count -eq 0) { throw "no eboot.bin or sce_module\*.prx in $Game" }

$CFlags = @("--target=arm64-apple-xros2.0", "-O2", "-ffreestanding", "-fno-stack-protector", "-fno-math-errno",
            "-frounding-math", "-w", "-isystem", (Join-Path $Runtime "freestanding"), "-I", $Runtime, "-c")
$sha = [System.Security.Cryptography.SHA256]::Create()
$started = Get-Date

# 1. Translate every module (seconds) and list the pieces of C.
$names = @()
$pieces = @()
foreach ($f in $files) {
    $name = ($f.BaseName.ToLower() -replace '[^a-z0-9_]', '_')
    if ($name -match '^[0-9]') { $name = "m_$name" }
    if ($f.Extension -ne ".bin") { $name = "${name}_prx" }
    if ($names -contains $name) { throw "two modules map to the name $name (rename one of them)" }
    Get-ChildItem -LiteralPath $Work -File | Where-Object { $_.Name -match "^$([regex]::Escape($name))(_\d+)?\.c$|^$([regex]::Escape($name))(_decl\.h|\.h|_files\.txt)$" } | Remove-Item
    $vpArgs = @("--elf", $f.FullName, "--pic", "--module", $name, "--split", $Split,
                "--out", (Join-Path $Work "$name.c"), "--stats", (Join-Path $Work "$name.json"))
    if ($Missing -and (Test-Path -LiteralPath $Missing)) { $vpArgs += @("--roots", $Missing) }
    Write-Host "== $($f.Name) -> $name"
    & $Vpaot @vpArgs
    if ($LASTEXITCODE -ne 0) { throw "vpaot failed on $($f.Name)" }
    $report = Get-Content -LiteralPath (Join-Path $Work "$name.json") -Raw | ConvertFrom-Json
    Write-Host ("   {0} functions, {1} instructions, {2:P2} supported" -f $report.functions, $report.instructions, $report.supported_fraction)
    $list = Join-Path $Work "${name}_files.txt"
    if (Test-Path -LiteralPath $list) {
        foreach ($l in Get-Content -LiteralPath $list) { if ($l.Trim()) { $pieces += (Join-Path $Work (Split-Path -Leaf $l.Trim())) } }
    } else {
        $pieces += (Join-Path $Work "$name.c")
    }
    $names += $name
}
& $Vpaot --registry (Join-Path $Work "vpengine_registry.c") --pack $Title @names
if ($LASTEXITCODE -ne 0) { throw "vpaot --registry failed" }
$pieces += (Join-Path $Work "vpengine_registry.c")

# 2. Compile the pieces in parallel. A piece whose C is the same as last time keeps its object;
#    each piece's C is deleted once compiled.
$todo = @()
$objects = @()
foreach ($c in $pieces) {
    $o = [IO.Path]::ChangeExtension($c, ".o")
    $h = [BitConverter]::ToString($sha.ComputeHash([IO.File]::ReadAllBytes($c))).Replace("-", "")
    $stamp = "$o.sha256"
    $objects += $o
    if ((Test-Path -LiteralPath $o) -and (Test-Path -LiteralPath $stamp) -and ((Get-Content -LiteralPath $stamp -Raw).Trim() -eq $h)) {
        if ($c -notlike "*vpengine_registry.c") { Remove-Item -LiteralPath $c }
        continue
    }
    $todo += [pscustomobject]@{ C = $c; O = $o; Hash = $h; Stamp = $stamp }
}
# Stale objects of pieces that no longer exist (a translation with more or fewer pieces).
$keep = @{}; foreach ($o in $objects) { $keep[$o.ToLower()] = 1 }
Get-ChildItem -LiteralPath $Work -File -Filter *.o | Where-Object { -not $keep.ContainsKey($_.FullName.ToLower()) } | Remove-Item
Write-Host ("Compiling {0} of {1} pieces with {2} jobs (the rest are unchanged)..." -f $todo.Count, $pieces.Count, $Jobs)
$failed = $null
$done = 0
function Complete-Piece($r) {
    $r.Process.WaitForExit()
    if ($r.Process.ExitCode -ne 0) { $script:failed = $r.Item.C; return }
    Set-Content -LiteralPath $r.Item.Stamp -Value $r.Item.Hash
    Remove-Item -LiteralPath $r.Item.C -ErrorAction SilentlyContinue        # compiled: its C goes
    Remove-Item -LiteralPath ($r.Item.O + ".log") -ErrorAction SilentlyContinue
    $script:done++
    Write-Progress -Activity "Compiling the game pack" -Status "$script:done / $($todo.Count)" -PercentComplete (100 * $script:done / [Math]::Max(1, $todo.Count))
}
$queue = New-Object System.Collections.Queue
foreach ($t in $todo) { $queue.Enqueue($t) }
$running = New-Object System.Collections.ArrayList
while (($queue.Count -gt 0 -and -not $failed) -or $running.Count -gt 0) {
    for ($i = $running.Count - 1; $i -ge 0; $i--) {
        if ($running[$i].Process.HasExited) { Complete-Piece $running[$i]; $running.RemoveAt($i) }
    }
    while ($queue.Count -gt 0 -and $running.Count -lt $Jobs -and -not $failed) {
        $t = $queue.Dequeue()
        $argList = ($CFlags + @($t.C, "-o", $t.O) | ForEach-Object { '"' + $_ + '"' }) -join " "
        $p = Start-Process -FilePath $Clang -ArgumentList $argList -NoNewWindow -PassThru -RedirectStandardError ($t.O + ".log")
        $null = $p.Handle # keeps the exit code readable after the process ends
        [void]$running.Add([pscustomobject]@{ Process = $p; Item = $t })
    }
    Start-Sleep -Milliseconds 100
}
Write-Progress -Activity "Compiling the game pack" -Completed
if ($failed) { throw "clang failed on $failed (see $([IO.Path]::ChangeExtension($failed, '.o')).log)" }
Remove-Item -LiteralPath (Join-Path $Work "vpengine_registry.c") -ErrorAction SilentlyContinue

# 3. Link: one library for visionOS that only needs the app's runtime and the system.
$tmp = "$Out.tmp"
$linkArgs = @("-flavor", "darwin", "-arch", "arm64", "-platform_version", "xros", "2.0", "26.0", "-dylib",
              "-adhoc_codesign", "-install_name", "@rpath/vpengine.vpgame", "-rpath", "@executable_path/Frameworks",
              "-o", $tmp) + $objects + @((Join-Path $Sdk "libVPRuntime.tbd"), (Join-Path $Sdk "libSystem.tbd"))
& $Lld @linkArgs
if ($LASTEXITCODE -ne 0) { throw "linking the pack failed" }
Move-Item -Force -LiteralPath $tmp -Destination $Out
if ($Clean) { Remove-Item -Recurse -Force -LiteralPath $Work }
$mb = (Get-Item -LiteralPath $Out).Length / 1MB
Write-Host ("Done in {0:N1} min: {1} ({2:N0} MB, {3} modules)." -f ((Get-Date) - $started).TotalMinutes, $Out, $mb, $names.Count)
Write-Host "Copy it into this game's folder in VPS4 on the headset, next to eboot.bin."
