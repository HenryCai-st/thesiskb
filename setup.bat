<# : batch part - runs the PowerShell below; keep this file ASCII-only
@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:TKB_SETUP_SELF='%~f0'; & ([ScriptBlock]::Create([IO.File]::ReadAllText('%~f0'))) %*"
set "rc=%errorlevel%"
echo.
if not defined TKB_SETUP_NOPAUSE pause
exit /b %rc%
#>
# ==========================================================================================
#  ThesisKB setup for Windows 10/11 (NVIDIA GPU recommended).
#
#  Double-click, or from a terminal:
#      setup.bat                       install / repair everything (safe to re-run)
#      setup.bat -InstallDir D:\kb     where to clone the repo (when not run inside it)
#      setup.bat -SkipModels           skip the large model downloads (~20 GB)
#      setup.bat -SkipServices         don't install/start Docker and Ollama (testing)
#      setup.bat -HfToken hf_xxx       Hugging Face token without the question (automation)
#
#  What it does (each step is skipped when already done):
#    1. installs Git, uv, ffmpeg, Tesseract, Docker Desktop, Ollama with winget
#    2. clones github.com/HenryCai-st/thesiskb (unless run from inside the repo)
#    3. writes .env (Tesseract paths, optional Hugging Face token), German OCR language pack
#    4. creates the Python environment (uv sync --extra local, ~5 GB)
#    5. starts Docker Desktop, creates the database (PostgreSQL + pgvector) and its schema
#    6. starts Ollama and pulls the models; pre-downloads Whisper, reranker, Docling models
#    7. runs `tkb doctor`
#  A log is written to setup.log next to the repo.
# ==========================================================================================
param(
    [string]$InstallDir = "$env:USERPROFILE\thesiskb",
    [string]$RepoUrl = "https://github.com/HenryCai-st/thesiskb.git",
    [switch]$SkipModels,
    [switch]$SkipServices,
    [string]$HfToken = ""
)

# "Continue": native tools (uv, git, winget, docker) write progress to stderr, which Windows
# PowerShell 5.1 would turn into fatal errors under "Stop"; failures are checked by exit code
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$script:Warnings = New-Object System.Collections.Generic.List[string]

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "    ok   $text" -ForegroundColor Green }
function Info($text) { Write-Host "    ..   $text" }
function Warn($text) { Write-Host "    !!   $text" -ForegroundColor Yellow; $script:Warnings.Add($text) }
function Fail($text) { Write-Host ""; Write-Host "SETUP STOPPED: $text" -ForegroundColor Red; exit 1 }

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
}

function Has($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Run([string]$exe, [string[]]$arguments) {
    # native command; stop on a non-zero exit code
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { Fail "'$exe $($arguments -join ' ')' failed (exit $LASTEXITCODE)" }
}

function Winget-Install([string]$id, [string]$name, [scriptblock]$present) {
    if (& $present) { Ok "$name already installed"; return }
    Info "installing $name (winget $id) - confirm the Windows prompt if one appears"
    & winget install --id $id -e --silent --accept-source-agreements --accept-package-agreements
    Refresh-Path
    if (-not (& $present)) { Fail "$name did not install. Install it by hand, then re-run setup.bat." }
    Ok "$name installed"
}

function Set-EnvLine([string]$file, [string]$key, [string]$value) {
    $lines = @(Get-Content -LiteralPath $file -Encoding UTF8)
    $done = $false
    $out = foreach ($l in $lines) {
        if ($l -match "^\s*#?\s*$([regex]::Escape($key))=") {
            if (-not $done) { $done = $true; "$key=$value" }
        } else { $l }
    }
    if (-not $done) { $out = @($out) + "$key=$value" }
    [IO.File]::WriteAllLines($file, [string[]]$out)   # UTF-8 without BOM
}

function Get-EnvLine([string]$file, [string]$key) {
    $m = Select-String -LiteralPath $file -Pattern "^$([regex]::Escape($key))=(.*)$" | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value.Trim() } else { return "" }
}

function Wait-For([scriptblock]$test, [int]$seconds, [string]$what) {
    $t0 = Get-Date
    while (-not (& $test)) {
        if (((Get-Date) - $t0).TotalSeconds -gt $seconds) { return $false }
        Start-Sleep -Seconds 3
    }
    return $true
}

function Ollama-Up {
    try { $null = Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 2; return $true }
    catch { return $false }
}

function Docker-Up { & docker info --format "{{.ServerVersion}}" 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) }

# ------------------------------------------------------------------------------------------
Write-Host "ThesisKB setup" -ForegroundColor Cyan
Refresh-Path
if (-not (Has "winget")) {
    Fail "winget is missing. Install 'App Installer' from the Microsoft Store, then re-run."
}

# --- 1. tools -----------------------------------------------------------------------------
Step "1/7 Tools"
Winget-Install "Git.Git" "Git" { Has "git" }
Winget-Install "astral-sh.uv" "uv" { Has "uv" }
Winget-Install "Gyan.FFmpeg" "ffmpeg" { (Has "ffmpeg") -and (Has "ffprobe") }
$tessExe = "$env:ProgramFiles\Tesseract-OCR\tesseract.exe"
Winget-Install "UB-Mannheim.TesseractOCR" "Tesseract OCR" { Test-Path $tessExe }

$gpu = $false
if (Has "nvidia-smi") {
    $gpuName = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -eq 0 -and $gpuName) { $gpu = $true; Ok "NVIDIA GPU: $gpuName" }
}
if (-not $gpu) {
    Warn "No NVIDIA GPU found (nvidia-smi). Using the small CPU model set (config/examples/models.cpu.yaml). Install the NVIDIA driver and re-run to use the full models."
}

# --- 2. repository ------------------------------------------------------------------------
Step "2/7 Repository"
$self = $env:TKB_SETUP_SELF
$here = if ($self) { Split-Path -Parent $self } else { (Get-Location).Path }
if (Test-Path (Join-Path $here "src\thesiskb")) {
    $Repo = $here
    Ok "using the repo this script is in: $Repo"
} elseif (Test-Path (Join-Path $InstallDir "src\thesiskb")) {
    $Repo = $InstallDir
    Info "repo exists at $Repo - updating"
    & git -C $Repo pull --ff-only
    if ($LASTEXITCODE -ne 0) { Warn "git pull failed (local changes?) - continuing with the current version" }
} else {
    Info "cloning $RepoUrl to $InstallDir"
    Run "git" @("clone", $RepoUrl, $InstallDir)
    $Repo = $InstallDir
    Ok "cloned"
}
Set-Location $Repo
Start-Transcript -Path (Join-Path $Repo "setup.log") -Append | Out-Null

# --- 3. configuration ---------------------------------------------------------------------
Step "3/7 Configuration (.env, OCR languages)"
$envFile = Join-Path $Repo ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $Repo ".env.example") $envFile -ErrorAction Stop
    Ok "created .env from .env.example"
} else { Ok ".env exists (kept, only missing settings are added)" }

$tessdata = "$env:LOCALAPPDATA\thesiskb\tessdata"
New-Item -ItemType Directory -Force $tessdata | Out-Null
foreach ($lang in @("eng", "osd")) {
    $dst = Join-Path $tessdata "$lang.traineddata"
    $src = "$env:ProgramFiles\Tesseract-OCR\tessdata\$lang.traineddata"
    if (-not (Test-Path $dst) -and (Test-Path $src)) { Copy-Item $src $dst }
}
$deu = Join-Path $tessdata "deu.traineddata"
if (-not (Test-Path $deu)) {
    Info "downloading the German OCR language pack"
    try {
        Invoke-WebRequest "https://github.com/tesseract-ocr/tessdata_fast/raw/main/deu.traineddata" -OutFile $deu -UseBasicParsing -ErrorAction Stop
    } catch { Warn "German OCR pack download failed ($($_.Exception.Message)); German photos will be read less well" }
}
Set-EnvLine $envFile "TKB_TESSERACT_CMD" $tessExe
Set-EnvLine $envFile "TKB_TESSDATA_DIR" $tessdata
Ok "Tesseract configured (deu, eng) in $tessdata"
if (-not $gpu) { Set-EnvLine $envFile "TKB_MODELS_FILE" "examples/models.cpu.yaml" }

if ($HfToken) { Set-EnvLine $envFile "TKB_HF_TOKEN" $HfToken }
if (-not (Get-EnvLine $envFile "TKB_HF_TOKEN")) {
    Write-Host ""
    Write-Host "    Speaker separation in recordings needs a free Hugging Face READ token:" -ForegroundColor White
    Write-Host "      1. accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1"
    Write-Host "      2. create a token at  https://huggingface.co/settings/tokens  (type: Read)"
    $token = Read-Host "    Paste the token (starts with hf_), or press Enter to skip"
    if ($token -match "^hf_\w+$") { Set-EnvLine $envFile "TKB_HF_TOKEN" $token; Ok "token saved to .env" }
    elseif ($token) { Warn "That does not look like a token (hf_...); not saved. Add TKB_HF_TOKEN to .env later." }
    else { Warn "No Hugging Face token: recordings will fail at speaker separation until TKB_HF_TOKEN is set in .env (or set diarization backend: none in config/models.yaml)." }
} else { Ok "Hugging Face token present in .env" }

# --- 4. Python environment ----------------------------------------------------------------
Step "4/7 Python environment (uv sync --extra local, first time ~5 GB)"
Run "uv" @("sync", "--extra", "local")
Ok "environment ready (.venv)"

# --- 5. services: Docker + database -------------------------------------------------------
Step "5/7 Docker Desktop and database"
if ($SkipServices) {
    Info "skipped (-SkipServices)"
} else {
    $dockerExe = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $dockerExe)) {
        & wsl.exe --status 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Info "Docker Desktop needs WSL2 - enabling it (administrator prompt)"
            Start-Process wsl.exe -ArgumentList "--install", "--no-distribution" -Verb RunAs -Wait
            Write-Host ""
            Write-Host "    WSL2 was enabled. RESTART Windows, then run setup.bat again." -ForegroundColor Yellow
            exit 0
        }
    }
    Winget-Install "Docker.DockerDesktop" "Docker Desktop" { Test-Path $dockerExe }
    $env:Path += ";$env:ProgramFiles\Docker\Docker\resources\bin"
    if (-not (Docker-Up)) {
        Info "starting Docker Desktop (first start: accept the terms in its window)"
        Start-Process $dockerExe
        if (-not (Wait-For { Docker-Up } 300 "Docker")) {
            Fail "Docker did not start within 5 minutes. Finish Docker Desktop's first-run dialog (or restart Windows), then re-run setup.bat."
        }
    }
    Ok "Docker is running"
    Run "docker" @("compose", "up", "-d", "--wait")
    Run "uv" @("run", "tkb", "db", "upgrade")
    Ok "database ready (container thesiskb-postgres, starts with Docker Desktop)"
}

# --- 6. models ----------------------------------------------------------------------------
Step "6/7 Models"
if ($SkipServices -or $SkipModels) {
    Info "skipped"
} else {
    $ollamaApp = "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
    Winget-Install "Ollama.Ollama" "Ollama" { Test-Path $ollamaApp }
    [Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")  # one big model in VRAM
    if (-not (Ollama-Up)) {
        Info "starting Ollama"
        Start-Process $ollamaApp
        if (-not (Wait-For { Ollama-Up } 120 "Ollama")) { Fail "Ollama did not start. Start the Ollama app, then re-run setup.bat." }
    }
    Ok "Ollama is running"
    $ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    $models = if ($gpu) { @("qwen3:14b", "bge-m3", "qwen2.5vl:7b") } else { @("qwen3:4b", "bge-m3", "qwen2.5vl:3b") }
    foreach ($m in $models) {
        Info "ollama pull $m (skipped quickly if present)"
        Run $ollama @("pull", $m)
    }
    Info "pre-downloading Whisper, reranker and Docling models (Hugging Face cache)"
    $whisper = if ($gpu) { "mobiuslabsgmbh/faster-whisper-large-v3-turbo" } else { "Systran/faster-whisper-small" }
    Run "uv" @("run", "python", "-c",
        "from huggingface_hub import snapshot_download as d; d('$whisper'); d('BAAI/bge-reranker-v2-m3')")
    Run "uv" @("run", "docling-tools", "models", "download", "layout", "tableformer")
    Ok "models ready"
}

# --- 7. check -----------------------------------------------------------------------------
Step "7/7 Check (tkb doctor)"
& uv run tkb doctor
$doctorOk = ($LASTEXITCODE -eq 0)

Stop-Transcript | Out-Null
Write-Host ""
if ($script:Warnings.Count) {
    Write-Host "Notes:" -ForegroundColor Yellow
    $script:Warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
}
if ($doctorOk) {
    Write-Host ""
    Write-Host "ThesisKB is ready in $Repo" -ForegroundColor Green
    Write-Host "  put files in   $Repo\data\raw"
    Write-Host "  then run       cd `"$Repo`"   and   uv run tkb ingest data/raw"
    Write-Host "  web UI         uv run tkb serve   ->  http://127.0.0.1:8000"
    Write-Host "  after a reboot start Docker Desktop and Ollama first (or enable 'start at login' in both)"
    Write-Host "  guide          $Repo\docs\usage.md"
    exit 0
}
Write-Host "Setup finished, but tkb doctor reports a problem (see above). Fix it and re-run setup.bat." -ForegroundColor Yellow
exit 1
