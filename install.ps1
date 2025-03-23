# NeuroVox Installer
# This script installs the NeuroVox voice assistant

Write-Host "NeuroVox Voice Assistant Installer" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.9+ and try again." -ForegroundColor Red
    exit
}

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
python -m venv venv
& "./venv/Scripts/Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install Ollama
Write-Host "Installing Ollama..." -ForegroundColor Cyan
if (!(Test-Path "$env:LOCALAPPDATA\Ollama\ollama.exe")) {
    Invoke-WebRequest -Uri "https://ollama.com/download/ollama-windows-amd64.zip" -OutFile "$env:TEMP\ollama.zip"
    Expand-Archive -Path "$env:TEMP\ollama.zip" -DestinationPath ".\ollama" -Force

    # Add to PATH for current session
    $env:Path += ";$(Resolve-Path .\ollama)"

    # Add to PATH permanently for user
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $ollamaPath = (Resolve-Path .\ollama).Path
    if ($userPath -notlike "*$ollamaPath*") {
        [Environment]::SetEnvironmentVariable("Path", $userPath + ";$ollamaPath", "User")
    }

    Write-Host "Ollama installed to $(Resolve-Path .\ollama)" -ForegroundColor Green
} else {
    Write-Host "Ollama already installed" -ForegroundColor Green
}

# Download Llama 3 model
Write-Host "Downloading Llama 3 8B model (4-bit quantized)..." -ForegroundColor Cyan
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5
& ollama pull llama3:8b-q4_0

Write-Host "Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path ".\data\temp_audio" | Out-Null
New-Item -ItemType Directory -Force -Path ".\data\conversation_history" | Out-Null
New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null

# Create desktop shortcut
Write-Host "Creating desktop shortcut..." -ForegroundColor Cyan
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\NeuroVox.lnk")
$Shortcut.TargetPath = (Resolve-Path .\start.bat).Path
$Shortcut.WorkingDirectory = (Resolve-Path .).Path
$Shortcut.IconLocation = "powershell.exe,0"
$Shortcut.Description = "NeuroVox Voice Assistant"
$Shortcut.Save()

Write-Host ""
Write-Host "NeuroVox installation completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "You can start NeuroVox by:" -ForegroundColor Yellow
Write-Host "1. Double-clicking the NeuroVox shortcut on your desktop" -ForegroundColor Yellow
Write-Host "2. Running .\start.bat" -ForegroundColor Yellow
Write-Host "3. Running .\start.ps1 in PowerShell" -ForegroundColor Yellow
Write-Host ""
Write-Host "To configure your Kokoro TTS API, edit the file: .\config\system_config.json" -ForegroundColor Yellow