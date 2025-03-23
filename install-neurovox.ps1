# NeuroVox Voice Assistant Windows Installer
# Run as Administrator for best results

# Text formatting
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Info($message) {
    Write-ColorOutput Blue "[INFO] $message"
}

function Write-Success($message) {
    Write-ColorOutput Green "[SUCCESS] $message"
}

function Write-Warning($message) {
    Write-ColorOutput Yellow "[WARNING] $message"
}

function Write-Error($message) {
    Write-ColorOutput Red "[ERROR] $message"
    return $false
}

# Print banner
Write-Host ""
Write-Host "================================================================" -ForegroundColor Blue
Write-Host "  _   _                      __     __           " -ForegroundColor Blue
Write-Host " | \ | | ___ _   _ _ __ ___ \ \   / /__  __  __ " -ForegroundColor Blue
Write-Host " |  \| |/ _ \ | | | '__/ _ \ \ \ / / _ \ \ \/ / " -ForegroundColor Blue
Write-Host " | |\  |  __/ |_| | | | (_) | \ V / (_) | >  <  " -ForegroundColor Blue
Write-Host " |_| \_|\___|\__,_|_|  \___/   \_/ \___/ /_/\_\ " -ForegroundColor Blue
Write-Host "                                                 " -ForegroundColor Blue
Write-Host "================================================================" -ForegroundColor Blue
Write-Host "     Advanced Local Voice Assistant Windows Installer           " -ForegroundColor Blue
Write-Host "================================================================" -ForegroundColor Blue
Write-Host ""

# Installation directory
$InstallDir = "$env:USERPROFILE\neurovox"
$TempDir = "$env:TEMP\neurovox_install"

# Confirm installation
$confirmation = Read-Host "This script will install NeuroVox to $InstallDir. Continue? (y/n)"
if ($confirmation -ne 'y') {
    exit
}

# Check for administrative privileges
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Warning "You are not running PowerShell as Administrator. Some operations may fail."
    $confirmation = Read-Host "Continue anyway? (y/n)"
    if ($confirmation -ne 'y') {
        exit
    }
}

# Create directories
Write-Info "Creating installation directories..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

# Check for NVIDIA GPU
Write-Info "Checking hardware..."
try {
    $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if ($gpuInfo) {
        Write-Success "NVIDIA GPU detected: $gpuInfo"
    }
} catch {
    Write-Warning "NVIDIA GPU not detected or drivers not installed. CPU-only mode will be very slow."
    $confirmation = Read-Host "Continue with CPU-only mode? (y/n)"
    if ($confirmation -ne 'y') {
        exit
    }
}

# Check Python
Write-Info "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python detected: $pythonVersion"
} catch {
    Write-Error "Python not found. Please install Python 3.9+ and try again."
    exit
}

# Install Ollama
Write-Info "Installing Ollama LLM backend..."
try {
    # Check if Ollama is already installed
    if (Test-Path "$env:LOCALAPPDATA\Ollama\ollama.exe") {
        Write-Success "Ollama already installed."
    } else {
        Write-Info "Downloading Ollama..."
        Invoke-WebRequest -Uri "https://ollama.com/download/ollama-windows-amd64.zip" -OutFile "$TempDir\ollama.zip"
        
        Write-Info "Extracting Ollama..."
        Expand-Archive -Path "$TempDir\ollama.zip" -DestinationPath "$InstallDir\ollama" -Force
        
        Write-Info "Setting up Ollama..."
        # Add to PATH for current session
        $env:Path += ";$InstallDir\ollama"
        
        # Add to PATH permanently for user
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$InstallDir\ollama*") {
            [Environment]::SetEnvironmentVariable("Path", $userPath + ";$InstallDir\ollama", "User")
        }
    }
    
    # Start Ollama server
    Write-Info "Starting Ollama server..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    
    # Wait for server to start
    Start-Sleep -Seconds 5
    
    # Download Llama model
    Write-Info "Downloading Llama 3 8B (4-bit quantized) model... This may take a while."
    & ollama pull llama3:8b-q4_0
    
    Write-Success "Ollama installed and model downloaded"
} catch {
    Write-Error "Failed to install Ollama: $_"
    Write-Warning "You may need to install Ollama manually from https://ollama.com/"
}

# Set up Python environment
Write-Info "Setting up Python environment..."
try {
    # Create virtual environment
    python -m venv "$InstallDir\venv"
    
    # Activate virtual environment
    & "$InstallDir\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install Python dependencies
    Write-Info "Installing Python dependencies..."
    python -m pip install faster-whisper numpy pydantic asyncio aiofiles sqlitedict pyaudio sounddevice soundfile requests
    
    Write-Success "Python environment set up successfully"
} catch {
    Write-Error "Failed to set up Python environment: $_"
    exit
}

# Create project files
Write-Info "Creating project files..."

# Create the directory structure
New-Item -ItemType Directory -Force -Path "$InstallDir\src\audio" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\src\brain" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\src\resources" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\src\services" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\src\utils" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\config" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\data\conversation_history" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\data\user_profiles" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\data\temp_audio" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\logs" | Out-Null

# Create startup scripts
Write-Info "Creating startup scripts..."
$content = @'
@echo off
REM NeuroVox Voice Assistant Startup Script
echo Starting NeuroVox Voice Assistant...
call venv\Scripts\activate.bat
cd src
python main.py
pause
'@
$content | Out-File -FilePath "$InstallDir\start.bat" -Encoding utf8

$content = @'
# Activate the virtual environment and start NeuroVox
Write-Host "Starting NeuroVox Voice Assistant..." -ForegroundColor Cyan
& "$PSScriptRoot\venv\Scripts\Activate.ps1"
Set-Location "$PSScriptRoot\src"
python main.py
'@
$content | Out-File -FilePath "$InstallDir\start.ps1" -Encoding utf8

# Create a shortcut on the desktop
Write-Info "Creating desktop shortcut..."
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\NeuroVox.lnk")
$Shortcut.TargetPath = "$InstallDir\start.bat"
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.IconLocation = "powershell.exe,0"
$Shortcut.Description = "NeuroVox Voice Assistant"
$Shortcut.Save()

# Cleanup temporary files
Write-Info "Cleaning up temporary files..."
Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue

# Complete
Write-Success "NeuroVox installation completed successfully!"
Write-Host ""
Write-Host "You can start NeuroVox by:" -ForegroundColor Yellow
Write-Host "1. Double-clicking the NeuroVox shortcut on your desktop" -ForegroundColor Yellow
Write-Host "2. Running $InstallDir\start.bat" -ForegroundColor Yellow
Write-Host "3. Running $InstallDir\start.ps1 in PowerShell" -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: When first run, the system will download required models which may take some time." -ForegroundColor Yellow
Write-Host "To configure your Kokoro TTS API, edit the file:" -ForegroundColor Yellow
Write-Host "$InstallDir\config\system_config.json" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
