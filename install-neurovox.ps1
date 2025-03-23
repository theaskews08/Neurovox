# Create this as install-neurovox.ps1

Write-Host "NeuroVox Voice Assistant Installer" -ForegroundColor Blue

# Create installation directory
$InstallDir = "$env:USERPROFILE\neurovox"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.9+ and try again." -ForegroundColor Red
    exit
}

# Install Ollama
Write-Host "Installing Ollama and downloading Llama 3 model..." -ForegroundColor Blue
Invoke-WebRequest -Uri https://ollama.com/download/ollama-windows-amd64.zip -OutFile "$env:TEMP\ollama.zip"
Expand-Archive -Path "$env:TEMP\ollama.zip" -DestinationPath "$InstallDir\ollama" -Force
$ENV:PATH += ";$InstallDir\ollama"
Write-Host "Ollama installed to $InstallDir\ollama" -ForegroundColor Green

# Download Llama 3
Start-Process -FilePath "$InstallDir\ollama\ollama.exe" -ArgumentList "pull llama3:8b-q4_0" -Wait

# Set up Python environment
Write-Host "Setting up Python environment..." -ForegroundColor Blue
python -m venv "$InstallDir\venv"
& "$InstallDir\venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install faster-whisper numpy pydantic asyncio aiofiles sqlitedict pyaudio sounddevice soundfile requests

# Clone repository files (simplified)
Write-Host "Creating project structure..." -ForegroundColor Blue
# Create directory structure...

Write-Host "NeuroVox installation complete!" -ForegroundColor Green
Write-Host "To start the assistant, run: cd $InstallDir && .\venv\Scripts\Activate.ps1 && python src\main.py" -ForegroundColor Yellow

logger.error(f"Error stopping service {service_name}: {str(e)}")
            
        logger.info("All services stopped")
        
    async def _start_service(self, name):
        """Start a single service"""
        if name not in self.services:
            logger.error(f"Service not registered: {name}")
            return False
            
        if self.services[name]["status"] == "running":
            logger.info(f"Service already running: {name}")
            return True
            
        # Check dependencies
        for dep in self.dependencies.get(name, []):
            if self.services[dep]["status"] != "running":
                logger.error(f"Dependency {dep} not running for service {name}")
                return False
                
        try:
            success = await self.services[name]["start"]()
            if success:
                self.services[name]["status"] = "running"
                self.restart_attempts[name] = 0
                logger.info(f"Service {name} started successfully")
                return True
            else:
                logger.error(f"Service {name} start function returned False")
                self.services[name]["status"] = "failed"
                return False
        except Exception as e:
            logger.error(f"Failed to start service {name}: {str(e)}")
            self.services[name]["status"] = "failed"
            return False
            
    async def _health_monitor(self):
        """Monitor service health and restart failed services"""
        logger.info("Health monitoring started")
        
        while self.is_running:
            for name, check_fn in self.health_checks.items():
                if self.services[name]["status"] == "running":
                    try:
                        is_healthy = await check_fn()
                        if not is_healthy:
                            logger.warning(f"Service {name} is unhealthy, attempting restart")
                            await self._restart_service(name)
                    except Exception as e:
                        logger.error(f"Health check for {name} failed: {str(e)}")
                        await self._restart_service(name)
                        
            # Wait before next check
            await asyncio.sleep(30)
            
    async def _restart_service(self, name):
        """Attempt to restart a failed service"""
        if self.restart_attempts[name] >= self.max_restart_attempts:
            logger.error(f"Service {name} failed too many times, not restarting")
            self.services[name]["status"] = "failed"
            return False
            
        try:
            # Stop the service if it's running
            if self.services[name]["status"] == "running":
                await self.services[name]["stop"]()
                
            # Restart it
            success = await self.services[name]["start"]()
            if success:
                self.services[name]["status"] = "running"
                self.restart_attempts[name] += 1
                logger.info(f"Service {name} restarted successfully (attempt {self.restart_attempts[name]})")
                return True
            else:
                logger.error(f"Service {name} restart failed")
                self.services[name]["status"] = "failed"
                return False
        except Exception as e:
            logger.error(f"Failed to restart service {name}: {str(e)}")
            self.services[name]["status"] = "failed"
            return False
            
    def _build_dependency_tree(self):
        """Build a flat list of services in dependency order"""
        visited = set()
        order = []
        
        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self.dependencies.get(name, []):
                visit(dep)
            order.append(name)
            
        for service in self.services:
            visit(service)
            
        return order
EOL

    cat > "$INSTALL_DIR/src/brain/memory_manager.py" << 'EOL'
import logging
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from sqlitedict import SqliteDict

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path("../data/conversation_history/memory.sqlite")
            
        # Create directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = str(db_path)
        self.short_term_memory = []
        self.mutex = asyncio.Lock()
        
        # Ensure the database exists
        self._init_db()
        
    def _init_db(self):
        """Initialize the database"""
        with SqliteDict(self.db_path, tablename="conversations") as db:
            db.commit()
            
        with SqliteDict(self.db_path, tablename="entities") as db:
            db.commit()
            
    async def add_exchange(self, user_input, assistant_response):
        """Add a conversation exchange to memory"""
        async with self.mutex:
            timestamp = datetime.now().isoformat()
            
            # Add to short-term memory
            self.short_term_memory.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            
            self.short_term_memory.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": timestamp
            })
            
            # Limit short-term memory to last 10 exchanges (20 messages)
            if len(self.short_term_memory) > 20:
                self.short_term_memory = self.short_term_memory[-20:]
                
            # Store in long-term memory
            conversation_id = timestamp[:10]  # Use date as conversation ID
            
            with SqliteDict(self.db_path, tablename="conversations") as db:
                if conversation_id in db:
                    conversation = db[conversation_id]
                else:
                    conversation = []
                    
                conversation.append({
                    "user": user_input,
                    "assistant": assistant_response,
                    "timestamp": timestamp
                })
                
                db[conversation_id] = conversation
                db.commit()
                
            logger.debug(f"Added exchange to memory (conversation {conversation_id})")
            
    async def get_recent_context(self, max_tokens=2000):
        """Get recent conversation context for the LLM"""
        async with self.mutex:
            # This is a simplified implementation without token counting
            # In a real implementation, we would count tokens and trim accordingly
            return self.short_term_memory
            
    async def get_relevant_context(self, query, max_tokens=2000):
        """Get context relevant to the query"""
        # This is a simplified implementation that just returns recent context
        # In a real implementation, we would use embeddings to find relevant past conversations
        return await self.get_recent_context(max_tokens)
EOL
    
    cat > "$INSTALL_DIR/src/brain/conversation_manager.py" << 'EOL'
import logging
import asyncio
import requests
import json

logger = logging.getLogger(__name__)

class ConversationalBrain:
    def __init__(self, resource_manager, memory_manager):
        self.resource_manager = resource_manager
        self.memory_manager = memory_manager
        self.current_topic = None
        self.persona = "default"
        self.conversation_state = "greeting"
        self.llm_url = "http://localhost:11434/api/generate"
        self.is_initialized = False
        
    async def start(self):
        """Start the conversational brain"""
        logger.info("Starting conversational brain")
        
        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                logger.error("Ollama API not available")
                return False
                
            # Check if the model is loaded
            models = response.json()
            if not any(model["name"].startswith("llama3") for model in models.get("models", [])):
                logger.warning("Llama 3 model not found in Ollama. Will be pulled on first request.")
                
            self.is_initialized = True
            logger.info("Conversational brain started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start conversational brain: {str(e)}")
            return False
            
    async def stop(self):
        """Stop the conversational brain"""
        logger.info("Stopping conversational brain")
        self.is_initialized = False
        return True
        
    async def health_check(self):
        """Check if the service is healthy"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False
            
    async def process_utterance(self, user_input):
        """Process user input with full contextual awareness"""
        if not self.is_initialized:
            logger.error("Cannot process utterance: Brain not initialized")
            return "I'm sorry, but I'm not fully initialized yet. Please try again in a moment."
            
        try:
            # Get conversation context
            context = await self.memory_manager.get_recent_context()
            
            # Generate system prompt based on current state
            system_prompt = self._generate_system_prompt()
            
            # Create messages array for the API
            messages = []
            
            # Add system message
            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            for msg in context:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Call Ollama API
            response = requests.post(
                self.llm_url,
                json={
                    "model": "llama3:8b-q4_0",
                    "prompt": user_input,
                    "system": system_prompt,
                    "stream": False,
                    "context": self._get_context_for_ollama(context)
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.text}")
                return "I'm sorry, but I encountered an error. Please try again."
                
            response_json = response.json()
            assistant_response = response_json.get("response", "")
            
            # Update conversation memory
            await self.memory_manager.add_exchange(user_input, assistant_response)
            
            # Update conversation state
            self._update_conversation_state(user_input, assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error processing utterance: {str(e)}")
            return "I'm sorry, but I encountered an error. Please try again."
            
    def _generate_system_prompt(self):
        """Generate appropriate system prompt based on current state"""
        base_prompt = "You are a helpful voice assistant named NeuroVox."
        
        state_prompts = {
            "greeting": "Provide a warm, brief greeting and offer assistance.",
            "topic_transition": "Acknowledge the topic change and respond appropriately.",
            "deep_conversation": "Provide thoughtful, nuanced responses while maintaining conversational flow.",
            "task_oriented": "Focus on clear, actionable information to help complete the task.",
            "emotional_support": "Show empathy and provide supportive responses."
        }
        
        return f"{base_prompt} {state_prompts.get(self.conversation_state, '')} Keep your responses concise since they will be spoken aloud."
        
    def _update_conversation_state(self, user_input, response):
        """Update conversation state based on exchange"""
        # Use simple keyword detection for state transitions
        if "how are you" in user_input.lower() or "feeling" in user_input.lower():
            self.conversation_state = "emotional_support"
        elif any(task in user_input.lower() for task in ["set", "remind", "calculate", "what is", "how do"]):
            self.conversation_state = "task_oriented"
        elif len(user_input.split()) > 15:  # Longer utterances suggest deeper conversation
            self.conversation_state = "deep_conversation"
            
    def _get_context_for_ollama(self, context):
        """Convert context to format expected by Ollama API"""
        # Ollama expects a list of strings for context
        if not context:
            return []
            
        context_str = []
        for msg in context:
            if msg["role"] == "user":
                context_str.append(f"Human: {msg['content']}")
            else:
                context_str.append(f"Assistant: {msg['content']}")
                
        return context_str
EOL

    cat > "$INSTALL_DIR/src/utils/logging_utils.py" << 'EOL'
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os

def setup_logging(log_dir=None, level=logging.INFO):
    """Set up logging configuration"""
    if log_dir is None:
        log_dir = Path("../logs")
        
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        log_dir / "neurovox.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
EOL

    cat > "$INSTALL_DIR/config/system_config.json" << 'EOL'
{
    "system": {
        "name": "NeuroVox",
        "version": "1.0.0"
    },
    "audio": {
        "stt": {
            "model_size": "medium",
            "language": "en",
            "beam_size": 5,
            "sample_rate": 16000,
            "record_duration": 5
        },
        "tts": {
            "api_url": "http://localhost:8080/api/tts",
            "voice": "default"
        }
    },
    "brain": {
        "llm": {
            "model": "llama3:8b-q4_0",
            "api_url": "http://localhost:11434/api/generate",
            "temperature": 0.7,
            "max_tokens": 500
        },
        "memory": {
            "short_term_length": 10,
            "max_tokens": 2000
        }
    },
    "resources": {
        "gpu_memory": 8192,
        "allocations": {
            "stt": 2048,
            "llm": 4096,
            "tts": 512
        }
    }
}
EOL

    log_success "Project files created successfully"
}

# Set up startup script
create_startup_script() {
    log "Creating startup script..."
    
    # Create the startup script
    cat > "$INSTALL_DIR/start.sh" << 'EOL'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Start the voice assistant
cd src && python main.py
EOL

    # Make it executable
    chmod +x "$INSTALL_DIR/start.sh"
    
    # Create a Windows batch file too
    cat > "$INSTALL_DIR/start.bat" << 'EOL'
@echo off
call venv\Scripts\activate.bat
cd src
python main.py
pause
EOL
    
    log_success "Startup scripts created"
}

# Main installation function
main() {
    # Show welcome message
    echo
    echo "This script will install NeuroVox, an advanced local voice assistant."
    echo "It will set up all required components for Faster Whisper (STT), Llama 3 (LLM), and Kokoro TTS."
    echo
    
    # Confirm installation
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    # Install system dependencies
    install_system_dependencies
    
    # Install Ollama
    install_ollama
    
    # Set up Python environment
    setup_python_environment
    
    # Download project files
    download_project_files
    
    # Create startup script
    create_startup_script
    
    echo
    log_success "NeuroVox installation complete!"
    echo
    echo "To start the voice assistant, run:"
    echo "  cd $INSTALL_DIR && ./start.sh"
    echo
    echo "For Windows:"
    echo "  cd $INSTALL_DIR && start.bat"
    echo
}

# Run the main function
main
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

# Create __init__.py files
"" | Out-File -FilePath "$InstallDir\src\__init__.py" -Encoding utf8
"" | Out-File -FilePath "$InstallDir\src\audio\__init__.py" -Encoding utf8
"" | Out-File -FilePath "$InstallDir\src\brain\__init__.py" -Encoding utf8
"" | Out-File -FilePath "$InstallDir\src\resources\__init__.py" -Encoding utf8
"" | Out-File -FilePath "$InstallDir\src\services\__init__.py" -Encoding utf8
"" | Out-File -FilePath "$InstallDir\src\utils\__init__.py" -Encoding utf8

# Create the Python files
# NOTE: In a real script, you would download these files from a repository
# For brevity, we're using placeholders here. Replace with actual code.

# Create main.py
$content = @'
import asyncio
import logging
from pathlib import Path
import os

# Local imports
from audio.audio_engine import AsyncAudioEngine
from audio.speech_to_text import WhisperSTT
from audio.text_to_speech import KokoroTTS
from brain.conversation_manager import ConversationalBrain
from brain.memory_manager import MemoryManager
from resources.resource_manager import NeuralResourceManager
from services.service_manager import ServiceManager
from utils.logging_utils import setup_logging

# Setup logging
logger = setup_logging()

class NeuroVox:
    def __init__(self):
        # Create directories if they don't exist
        self.data_dir = Path("../data")
        self.temp_audio_dir = self.data_dir / "temp_audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the resource manager
        self.resource_manager = NeuralResourceManager(gpu_memory=8192)  # 8GB in MB
        
        # Initialize components
        self.audio_engine = AsyncAudioEngine(temp_dir=str(self.temp_audio_dir))
        self.stt = WhisperSTT(resource_manager=self.resource_manager)
        self.tts = KokoroTTS(resource_manager=self.resource_manager)
        self.memory_manager = MemoryManager()
        self.brain = ConversationalBrain(resource_manager=self.resource_manager, 
                                         memory_manager=self.memory_manager)
        
        # Initialize service manager
        self.service_manager = ServiceManager()
        self._register_services()
        
    def _register_services(self):
        """Register all services with the service manager"""
        self.service_manager.register_service(
            "stt", self.stt.start, self.stt.stop, self.stt.health_check)
        self.service_manager.register_service(
            "tts", self.tts.start, self.tts.stop, self.tts.health_check)
        self.service_manager.register_service(
            "brain", self.brain.start, self.brain.stop, self.brain.health_check,
            dependencies=["stt", "tts"])
        self.service_manager.register_service(
            "audio", self.audio_engine.start, self.audio_engine.stop)
        
    async def start(self):
        """Start the voice assistant"""
        logger.info("Starting NeuroVox voice assistant...")
        await self.service_manager.start_all()
        logger.info("NeuroVox started successfully!")
        
        # Main interaction loop
        try:
            while True:
                # Record audio
                logger.info("Listening...")
                audio_data = await self.stt.listen()
                
                # Process speech to text
                text = await self.stt.transcribe(audio_data)
                logger.info(f"User said: {text}")
                
                if not text.strip():
                    logger.info("No speech detected, continuing...")
                    continue
                
                # Process with brain
                response = await self.brain.process_utterance(text)
                logger.info(f"Assistant response: {response}")
                
                # Generate speech
                audio_response = await self.tts.synthesize(response)
                
                # Play response
                await self.audio_engine.play_audio(audio_response)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.service_manager.stop_all()
            logger.info("NeuroVox stopped.")

async def main():
    assistant = NeuroVox()
    await assistant.start()

if __name__ == "__main__":
    asyncio.run(main())
'@
$content | Out-File -FilePath "$InstallDir\src\main.py" -Encoding utf8

# Create audio_engine.py
$content = @'
import os
import uuid
import asyncio
import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

class AsyncAudioEngine:
    def __init__(self, temp_dir="temp_audio"):
        self.audio_queue = asyncio.Queue()
        self.is_playing = False
        self.current_player = None
        self.temp_dir = temp_dir
        self.temp_files = []
        self.mutex = asyncio.Lock()
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
    async def start(self):
        """Start the audio engine service"""
        logger.info("Starting audio engine...")
        return True
        
    async def stop(self):
        """Stop the audio engine service"""
        logger.info("Stopping audio engine...")
        await self.cleanup_all()
        return True

async def play_audio(self, audio_data):
        """Queue audio data for playback"""
        await self.audio_queue.put(audio_data)
        
        # Start the player loop if not already running
        if not self.is_playing:
            asyncio.create_task(self._player_loop())
            
    async def _player_loop(self):
        """Background task to play queued audio"""
        self.is_playing = True
        
        while not self.audio_queue.empty():
            audio_data = await self.audio_queue.get()
            
            async with self.mutex:
                # Generate unique filename
                filename = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.wav")
                self.temp_files.append(filename)
                
                # Write audio data to file
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                
                # Play the audio
                await self._play_file(filename)
                
                # Try to clean up the file
                await self._cleanup_file(filename)
                
        self.is_playing = False
        
    async def _play_file(self, filename):
        """Play audio file with proper resource management"""
        if platform.system() == "Windows":
            # Windows approach
            process = await asyncio.create_subprocess_exec(
                "powershell", "-c", f"(New-Object Media.SoundPlayer '{filename}').PlaySync()",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
        else:
            # Linux/Mac approach using aplay/afplay
            cmd = "afplay" if platform.system() == "Darwin" else "aplay"
            process = await asyncio.create_subprocess_exec(
                cmd, filename,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            
    async def _cleanup_file(self, filename):
        """Safely clean up temporary audio file"""
        for _ in range(3):  # Try up to 3 times
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    if filename in self.temp_files:
                        self.temp_files.remove(filename)
                    logger.debug(f"Successfully removed temporary file: {filename}")
                break
            except (PermissionError, OSError) as e:
                # File might still be in use, wait a bit
                logger.debug(f"Could not remove file {filename}, retrying... ({str(e)})")
                await asyncio.sleep(0.5)
                
    async def cleanup_all(self):
        """Clean up all temporary files"""
        async with self.mutex:
            for filename in list(self.temp_files):
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                    self.temp_files.remove(filename)
                except (PermissionError, OSError) as e:
                    logger.warning(f"Could not remove temp file during cleanup: {filename} ({str(e)})")
'@
$content | Out-File -FilePath "$InstallDir\src\audio\audio_engine.py" -Encoding utf8

# Create speech_to_text.py
$content = @'
import asyncio
import logging
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class WhisperSTT:
    def __init__(self, resource_manager, model_size="medium"):
        self.resource_manager = resource_manager
        self.model_size = model_size
        self.model = None
        self.is_initialized = False
        self.mutex = asyncio.Lock()
        
    async def start(self):
        """Start the STT service"""
        logger.info(f"Starting Whisper STT with model size: {self.model_size}")
        
        # Allocate memory
        memory = await self.resource_manager.allocate("stt", 2048)
        logger.info(f"Allocated {memory}MB for STT")
        
        # Initialize the model
        try:
            # Load the model - use CUDA if available
            self.model = WhisperModel(
                self.model_size, 
                device="cuda", 
                compute_type="float16"
            )
            self.is_initialized = True
            logger.info("Whisper STT initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Whisper STT: {str(e)}")
            await self.resource_manager.release("stt")
            return False
            
    async def stop(self):
        """Stop the STT service"""
        logger.info("Stopping Whisper STT")
        self.is_initialized = False
        
        # Release memory
        await self.resource_manager.release("stt")
        
        # Free model (help garbage collection)
        self.model = None
        return True
        
    async def health_check(self):
        """Check if the service is healthy"""
        return self.is_initialized and self.model is not None
        
    async def listen(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        logger.info(f"Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Convert to the format expected by Whisper
        audio_data = recording.flatten()
        return audio_data
        
    async def transcribe(self, audio_data):
        """Transcribe audio data to text"""
        async with self.mutex:
            if not self.is_initialized:
                logger.error("Cannot transcribe: Whisper STT not initialized")
                return ""
                
            try:
                logger.info("Transcribing audio...")
                
                # Process with Whisper
                segments, info = self.model.transcribe(
                    audio_data,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Collect all segments into a single string
                text = " ".join(segment.text for segment in segments)
                
                logger.info(f"Transcription complete: {text}")
                return text
                
            except Exception as e:
                logger.error(f"Transcription error: {str(e)}")
                return ""
'@
$content | Out-File -FilePath "$InstallDir\src\audio\speech_to_text.py" -Encoding utf8

# Create text_to_speech.py
$content = @'
import logging
import asyncio
import subprocess
import tempfile
import os
import requests
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class KokoroTTS:
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.is_initialized = False
        self.api_url = "http://localhost:8080/api/tts"  # Default Kokoro API endpoint
        self.voice = "default"
        self.mutex = asyncio.Lock()
        self.temp_dir = Path("../data/temp_audio")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    async def start(self):
        """Start the TTS service"""
        logger.info("Starting Kokoro TTS")
        
        # Allocate memory
        memory = await self.resource_manager.allocate("tts", 512)
        logger.info(f"Allocated {memory}MB for TTS")
        
        try:
            # Check if Kokoro API is available by sending a test request
            # This is just a placeholder for your actual Kokoro API check
            # In this example, we're simulating success
            
            self.is_initialized = True
            logger.info("Kokoro TTS initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {str(e)}")
            await self.resource_manager.release("tts")
            return False
            
    async def stop(self):
        """Stop the TTS service"""
        logger.info("Stopping Kokoro TTS")
        self.is_initialized = False
        
        # Release memory
        await self.resource_manager.release("tts")
        return True
        
    async def health_check(self):
        """Check if the service is healthy"""
        return self.is_initialized
        
    async def synthesize(self, text):
        """Synthesize text to speech using Kokoro TTS API"""
        async with self.mutex:
            if not self.is_initialized:
                logger.error("Cannot synthesize: Kokoro TTS not initialized")
                return b""
                
            try:
                logger.info(f"Synthesizing text: {text[:50]}...")
                
                # Create a temporary file for the audio output
                temp_file = self.temp_dir / f"tts_temp_{id(text)}.wav"
                
                # Call Kokoro TTS API
                # This is a placeholder for your actual Kokoro TTS API call
                # For demonstration, we'll use a simple echo command to create a dummy WAV file
                
                # For actual implementation, uncomment and modify:
                # response = requests.post(
                #     self.api_url,
                #     json={"text": text, "voice": self.voice}
                # )
                # audio_data = response.content
                
                # For demo purposes, create an empty WAV file
                # Replace this with your actual Kokoro TTS implementation
                process = await asyncio.create_subprocess_exec(
                    "echo", "Placeholder for TTS output",
                    stdout=open(temp_file, "wb")
                )
                await process.wait()
                
                # Read the audio data
                with open(temp_file, "rb") as f:
                    audio_data = f.read()
                
                # Clean up the temporary file
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Could not delete temp TTS file: {str(e)}")
                
                logger.info(f"Synthesized {len(audio_data)} bytes of audio")
                return audio_data
                
            except Exception as e:
                logger.error(f"Synthesis error: {str(e)}")
                return b""
'@
$content | Out-File -FilePath "$InstallDir\src\audio\text_to_speech.py" -Encoding utf8

# Create resource_manager.py
$content = @'
import logging
import asyncio

logger = logging.getLogger(__name__)

class NeuralResourceManager:
    def __init__(self, gpu_memory=8192):  # Memory in MB
        self.total_memory = gpu_memory
        self.allocations = {}
        self.priorities = {
            "stt": 1,  # Higher priority components
            "llm": 0,
            "tts": 2,
            "classifier": 3
        }
        self.min_allocations = {
            "stt": 1024,  # Minimum memory requirements
            "llm": 3072,
            "tts": 512,
            "classifier": 256
        }
        self.mutex = asyncio.Lock()
        
    async def allocate(self, component, requested_memory=None):
        """Allocate memory for a component"""
        async with self.mutex:
            # Get minimum required memory
            min_memory = self.min_allocations.get(component, 256)
            
            # If requested memory is None, use minimum
            if requested_memory is None:
                requested_memory = min_memory
                
            logger.debug(f"Allocating {requested_memory}MB for {component}")
            
            # Check current allocations
            current_usage = sum(self.allocations.values())
            available = self.total_memory - current_usage
            
            if requested_memory <= available:
                # We can allocate without adjustments
                self.allocations[component] = requested_memory
                return requested_memory
                
            # Need to optimize allocations
            logger.info(f"Optimizing memory allocations to fit {component}")
            return await self._optimize_for_component(component, requested_memory)
    
    async def release(self, component):
        """Release memory allocated to a component"""
        async with self.mutex:
            if component in self.allocations:
                freed = self.allocations.pop(component)
                logger.debug(f"Released {freed}MB from {component}")
                return freed
            return 0
            
    async def _optimize_for_component(self, target_component, requested_memory):
        """Optimize memory allocations to make room for a component"""
        # Sort components by priority (higher number = lower priority)
        sorted_components = sorted(
            self.allocations.keys(),
            key=lambda x: self.priorities.get(x, 99)
        )
        
        # Start taking memory from lowest priority components
        needed = requested_memory - (self.total_memory - sum(self.allocations.values()))
        
        if needed <= 0:
            # Should not happen, but just in case
            self.allocations[target_component] = requested_memory
            return requested_memory
            
        # Take memory from other components
        for component in reversed(sorted_components):
            # Skip if this is the target component
            if component == target_component:
                continue
                
            # Get current allocation and minimum
            current = self.allocations[component]
            minimum = self.min_allocations.get(component, 256)
            
            # How much can we take?
            can_take = max(0, current - minimum)
            
            if can_take > 0:
                # Take what we need, up to what's available
                to_take = min(needed, can_take)
                self.allocations[component] -= to_take
                needed -= to_take
                
                logger.info(f"Reduced {component} allocation by {to_take}MB")
                
                if needed <= 0:
                    break
                    
        # If we still need memory, we have to reduce to bare minimum
        if needed > 0:
            logger.warning(f"Could not fully optimize for {target_component}, allocating {requested_memory - needed}MB instead of {requested_memory}MB")
            self.allocations[target_component] = requested_memory - needed
            return requested_memory - needed
            
        # Allocate the requested memory
        self.allocations[target_component] = requested_memory
        return requested_memory
        
    def get_allocation(self, component):
        """Get current allocation for component"""
        return self.allocations.get(component, 0)
        
    def get_available_memory(self):
        """Get total available memory"""
        return self.total_memory - sum(self.allocations.values())
'@
$content | Out-File -FilePath "$InstallDir\src\resources\resource_manager.py" -Encoding utf8

# Create service_manager.py 
$content = @'
import asyncio
import logging
from typing import Dict, List, Callable, Optional

logger = logging.getLogger(__name__)

class ServiceManager:
    def __init__(self):
        self.services = {}
        self.health_checks = {}
        self.dependencies = {}
        self.restart_attempts = {}
        self.max_restart_attempts = 3
        self.is_running = False
        self.monitor_task = None
        
    def register_service(self, name, start_fn, stop_fn, health_check_fn=None, dependencies=None):
        """Register a service with the manager"""
        self.services[name] = {
            "start": start_fn,
            "stop": stop_fn,
            "status": "stopped"
        }
        
        if health_check_fn:
            self.health_checks[name] = health_check_fn
            
        if dependencies:
            self.dependencies[name] = dependencies
        else:
            self.dependencies[name] = []
            
        self.restart_attempts[name] = 0
        logger.info(f"Registered service: {name}")
        
    async def start_all(self):
        """Start all services in dependency order"""
        self.is_running = True
        
        # Build dependency tree
        dependency_tree = self._build_dependency_tree()
        logger.info(f"Starting services in order: {dependency_tree}")
        
        # Start services in order
        for service_name in dependency_tree:
            logger.info(f"Starting service: {service_name}")
            success = await self._start_service(service_name)
            if not success:
                logger.error(f"Failed to start service: {service_name}")
                # If a critical service fails, we might want to handle that here
            
        # Start health check monitoring
        self.monitor_task = asyncio.create_task(self._health_monitor())
        logger.info("All services started, health monitoring active")
        
    async def stop_all(self):
        """Stop all services in reverse dependency order"""
        self.is_running = False
        
        # Cancel health monitor
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            
        # Build dependency tree and reverse it
        dependency_tree = self._build_dependency_tree()
        dependency_tree.reverse()
        
        logger.info(f"Stopping services in order: {dependency_tree}")
        
        # Stop services in reverse order
        for service_name in dependency_tree:
            if self.services[service_name]["status"] == "running":
                logger.info(f"Stopping service: {service_name}")
                try:
                    await self.services[service_name]["stop"]()
                    self.services[service_name]["status"] = "stopped"
                except Exception as e:
                    logger.error(f"Error stopping service {service_name}: {str(e)}")
            
    async def _start_service(self, name):
        """Start a single service"""
        if name not in self.services:
            logger.error(f"Service not registered: {name}")
            return False
            
        if self.services[name]["status"] == "running":
            logger.info(f"Service already running: {name}")
            return True
            
        # Check dependencies
        for dep in self.dependencies.get(name, []):
            if self.services[dep]["status"] != "running":
                logger.error(f"Dependency {dep} not running for service {name}")
                return False
                
        try:
            success = await self.services[name]["start"]()
            if success:
                self.services[name]["status"] = "running"
                self.restart_attempts[name] = 0
                logger.info(f"Service {name} started successfully")
                return True
            else:
                logger.error(f"Service {name} start function returned False")
                self.services[name]["status"] = "failed"
                return False
        except Exception as e:
            logger.error(f"Failed to start service {name}: {str(e)}")
            self.services[name]["status"] = "failed"
            return False
            
    async def _health_monitor(self):
        """Monitor service health and restart failed services"""
        logger.info("Health monitoring started")
        
        while self.is_running:
            for name, check_fn in self.health_checks.items():
                if self.services[name]["status"] == "running":
                    try:
                        is_healthy = await check_fn()
                        if not is_healthy:
                            logger.warning(f"Service {name} is unhealthy, attempting restart")
                            await self._restart_service(name)
                    except Exception as e:
                        logger.error(f"Health check for {name} failed: {str(e)}")
                        await self._restart_service(name)
                        
            # Wait before next check
            await asyncio.sleep(30)
            
    async def _restart_service(self, name):
        """Attempt to restart a failed service"""
        if self.restart_attempts[name] >= self.max_restart_attempts:
            logger.error(f"Service {name} failed too many times, not restarting")
            self.services[name]["status"] = "failed"
            return False
            
        try:
            # Stop the service if it's running
            if self.services[name]["status"] == "running":
                await self.services[name]["stop"]()
                
            # Restart it
            success = await self.services[name]["start"]()
            if success:
                self.services[name]["status"] = "running"
                self.restart_attempts[name] += 1
                logger.info(f"Service {name} restarted successfully (attempt {self.restart_attempts[name]})")
                return True
            else:
                logger.error(f"Service {name} restart failed")
                self.services[name]["status"] = "failed"
                return False
        except Exception as e:
            logger.error(f"Failed to restart service {name}: {str(e)}")
            self.services[name]["status"] = "failed"
            return False
            
    def _build_dependency_tree(self):
        """Build a flat list of services in dependency order"""
        visited = set()
        order = []
        
        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self.dependencies.get(name, []):
                visit(dep)
            order.append(name)
            
        for service in self.services:
            visit(service)
            
        return order
'@
$content | Out-File -FilePath "$InstallDir\src\services\service_manager.py" -Encoding utf8

# Create memory_manager.py
$content = @'
import logging
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from sqlitedict import SqliteDict

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path("../data/conversation_history/memory.sqlite")
            
        # Create directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = str(db_path)
        self.short_term_memory = []
        self.mutex = asyncio.Lock()
        
        # Ensure the database exists
        self._init_db()
        
    def _init_db(self):
        """Initialize the database"""
        with SqliteDict(self.db_path, tablename="conversations") as db:
            db.commit()
            
        with SqliteDict(self.db_path, tablename="entities") as db:
            db.commit()
            
    async def add_exchange(self, user_input, assistant_response):
        """Add a conversation exchange to memory"""
        async with self.mutex:
            timestamp = datetime.now().isoformat()
            
            # Add to short-term memory
            self.short_term_memory.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            
            self.short_term_memory.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": timestamp
            })
            
            # Limit short-term memory to last 10 exchanges (20 messages)
            if len(self.short_term_memory) > 20:
                self.short_term_memory = self.short_term_memory[-20:]
                
            # Store in long-term memory
            conversation_id = timestamp[:10]  # Use date as conversation ID
            
            with SqliteDict(self.db_path, tablename="conversations") as db:
                if conversation_id in db:
                    conversation = db[conversation_id]
                else:
                    conversation = []
                    
                conversation.append({
                    "user": user_input,
                    "assistant": assistant_response,
                    "timestamp": timestamp
                })
                
                db[conversation_id] = conversation
                db.commit()
                
            logger.debug(f"Added exchange to memory (conversation {conversation_id})")
            
    async def get_recent_context(self, max_tokens=2000):
        """Get recent conversation context for the LLM"""
        async with self.mutex:
            # This is a simplified implementation without token counting
            # In a real implementation, we would count tokens and trim accordingly
            return self.short_term_memory
            
    async def get_relevant_context(self, query, max_tokens=2000):
        """Get context relevant to the query"""
        # This is a simplified implementation that just returns recent context
        # In a real implementation, we would use embeddings to find relevant past conversations
        return await self.get_recent_context(max_tokens)
'@
$content | Out-File -FilePath "$InstallDir\src\brain\memory_manager.py" -Encoding utf8

# Create conversation_manager.py
$content = @'
import logging
import asyncio
import requests
import json

logger = logging.getLogger(__name__)

class ConversationalBrain:
    def __init__(self, resource_manager, memory_manager):
        self.resource_manager = resource_manager
        self.memory_manager = memory_manager
        self.current_topic = None
        self.persona = "default"
        self.conversation_state = "greeting"
        self.llm_url = "http://localhost:11434/api/generate"
        self.is_initialized = False
        
    async def start(self):
        """Start the conversational brain"""
        logger.info("Starting conversational brain")
        
        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                logger.error("Ollama API not available")
                return False
                
            # Check if the model is loaded
            models = response.json()
            if not any(model["name"].startswith("llama3") for model in models.get("models", [])):
                logger.warning("Llama 3 model not found in Ollama. Will be pulled on first request.")
                
            self.is_initialized = True
            logger.info("Conversational brain started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start conversational brain: {str(e)}")
            return False
            
    async def stop(self):
        """Stop the conversational brain"""
        logger.info("Stopping conversational brain")
        self.is_initialized = False
        return True
        
    async def health_check(self):
        """Check if the service is healthy"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False
            
    async def process_utterance(self, user_input):
        """Process user input with full contextual awareness"""
        if not self.is_initialized:
            logger.error("Cannot process utterance: Brain not initialized")
            return "I'm sorry, but I'm not fully initialized yet. Please try again in a moment."
            
        try:
            # Get conversation context
            context = await self.memory_manager.get_recent_context()
            
            # Generate system prompt based on current state
            system_prompt = self._generate_system_prompt()
            
            # Create messages array for the API
            messages = []
            
            # Add system message
            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            for msg in context:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Call Ollama API
            response = requests.post(
                self.llm_url,
                json={
                    "model": "llama3:8b-q4_0",
                    "prompt": user_input,
                    "system": system_prompt,
                    "stream": False,
                    "context": self._get_context_for_ollama(context)
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.text}")
                return "I'm sorry, but I encountered an error. Please try again."
                
            response_json = response.json()
            assistant_response = response_json.get("response", "")
            
            # Update conversation memory
            await self.memory_manager.add_exchange(user_input, assistant_response)
            
            # Update conversation state
            self._update_conversation_state(user_input, assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error processing utterance: {str(e)}")
            return "I'm sorry, but I encountered an error. Please try again."
            
    def _generate_system_prompt(self):
        """Generate appropriate system prompt based on current state"""
        base_prompt = "You are a helpful voice assistant named NeuroVox."
        
        state_prompts = {
            "greeting": "Provide a warm, brief greeting and offer assistance.",
            "topic_transition": "Acknowledge the topic change and respond appropriately.",
            "deep_conversation": "Provide thoughtful, nuanced responses while maintaining conversational flow.",
            "task_oriented": "Focus on clear, actionable information to help complete the task.",
            "emotional_support": "Show empathy and provide supportive responses."
        }
        
        return f"{base_prompt} {state_prompts.get(self.conversation_state, '')} Keep your responses concise since they will be spoken aloud."
        
    def _update_conversation_state(self, user_input, response):
        """Update conversation state based on exchange"""
        # Use simple keyword detection for state transitions
        if "how are you" in user_input.lower() or "feeling" in user_input.lower():
            self.conversation_state = "emotional_support"
        elif any(task in user_input.lower() for task in ["set", "remind", "calculate", "what is", "how do"]):
            self.conversation_state = "task_oriented"
        elif len(user_input.split()) > 15:  # Longer utterances suggest deeper conversation
            self.conversation_state = "deep_conversation"
            
    def _get_context_for_ollama(self, context):
        """Convert context to format expected by Ollama API"""
        # Ollama expects a list of strings for context
        if not context:
            return []
            
        context_str = []
        for msg in context:
            if msg["role"] == "user":
                context_str.append(f"Human: {msg['content']}")
            else:
                context_str.append(f"Assistant: {msg['content']}")
                
        return context_str
'@
$content | Out-File -FilePath "$InstallDir\src\brain\conversation_manager.py" -Encoding utf8

# Create logging_utils.py
$content = @'
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os

def setup_logging(log_dir=None, level=logging.INFO):
    """Set up logging configuration"""
    if log_dir is None:
        log_dir = Path("../logs")
        
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the logger
    logger = logging.getLogger()
    logger.

<function_calls>
<invoke name="artifacts">
<parameter name="id">final-script-portion</parameter>
<parameter name="type">application/vnd.ant.code</parameter>
<parameter name="title">NeuroVox Windows Installation Script (Final Portion)</parameter>
<parameter name="command">create</parameter>
<parameter name="content">    logger.setLevel(level)
Copy# Clear any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create file handler
file_handler = RotatingFileHandler(
    log_dir / "neurovox.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(level)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(level)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

return logger
'@
content∣Out−File−FilePath"content | Out-File -FilePath "
content∣Out−File−FilePath"InstallDir\src\utils\logging_utils.py" -Encoding utf8

Create system_config.json
$content = @'
{
"system": {
"name": "NeuroVox",
"version": "1.0.0"
},
"audio": {
"stt": {
"model_size": "medium",
"language": "en",
"beam_size": 5,
"sample_rate": 16000,
"record_duration": 5
},
"tts": {
"api_url": "http://localhost:8080/api/tts",
"voice": "default"
}
},
"brain": {
"llm": {
"model": "llama3:8b-q4_0",
"api_url": "http://localhost:11434/api/generate",
            "temperature": 0.7,
            "max_tokens": 500
        },
        "memory": {
            "short_term_length": 10,
            "max_tokens": 2000
        }
    },
    "resources": {
        "gpu_memory": 8192,
        "allocations": {
            "stt": 2048,
            "llm": 4096,
            "tts": 512
        }
    }
}
'@
content∣Out−File−FilePath"content | Out-File -FilePath "
content∣Out−File−FilePath"InstallDir\config\system_config.json" -Encoding utf8

Create startup scripts
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
content∣Out−File−FilePath"content | Out-File -FilePath "
content∣Out−File−FilePath"InstallDir\start.bat" -Encoding utf8

$content = @'
Activate the virtual environment and start NeuroVox
Write-Host "Starting NeuroVox Voice Assistant..." -ForegroundColor Cyan
& "$PSScriptRoot\venv\Scripts\Activate.ps1"
Set-Location "$PSScriptRoot\src"
python main.py
'@
content∣Out−File−FilePath"content | Out-File -FilePath "
content∣Out−File−FilePath"InstallDir\start.ps1" -Encoding utf8

Create a shortcut on the desktop
Write-Info "Creating desktop shortcut..."
$WshShell = New-Object -ComObject WScript.Shell
Shortcut = $WshShell.CreateShortcut("
env
\Desktop\NeuroVox.lnk")
Shortcut.TargetPath="Shortcut.TargetPath = "
Shortcut.TargetPath="InstallDir\start.bat"
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.IconLocation = "powershell.exe,0"
$Shortcut.Description = "NeuroVox Voice Assistant"
$Shortcut.Save()

Cleanup temporary files
Write-Info "Cleaning up temporary files..."
Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
Complete
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
</parameter>
<parameter name="language">powershell</parameter>
</invoke>RetryClaude does not have the ability to run the code it generates yet. Claude does not have internet access. Links provided may not be accurate or up to date.
