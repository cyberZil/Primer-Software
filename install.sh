#!/usr/bin/env bash
# install.sh — XRP Primer AI Kit Demo setup script
# Tested on Ubuntu Server 22.04+ (Rubik Pi 3)
# Run with: bash install.sh

set -e # Exit on error
set -u # Treat unset vars as errors

# --- COLOR OUTPUT HELPERS ---
info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# --- STEP 1: System Update ---
info "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# --- STEP 2: Install System Dependencies ---
info "Installing required packages (git, curl, wget, python3.12, build tools, audio dev libs)..."
sudo apt install -y git curl wget python3.12 python3.12-venv build-essential python3-dev libportaudio2 portaudio19-dev

# --- STEP 3: Create projects folder (if missing) ---
PROJECTS_DIR="$HOME/projects"
if [ ! -d "$PROJECTS_DIR" ]; then
    info "Creating projects directory at $PROJECTS_DIR"
    mkdir -p "$PROJECTS_DIR"
fi
cd "$PROJECTS_DIR"

# --- STEP 4: Clone repo if missing ---
REPO_URL="https://github.com/XRP-AI-Kit/Primer-Software"
REPO_NAME="Primer-Software"
if [ ! -d "$REPO_NAME" ]; then
    info "Cloning repository..."
    git clone "$REPO_URL"
else
    warn "Repository already exists, skipping clone."
fi
cd "$REPO_NAME"

# --- STEP 5: Set up Python venv ---
if [ ! -d "venv" ]; then
    info "Creating Python 3.12 virtual environment..."
    python3.12 -m venv venv
else
    warn "Virtual environment already exists, skipping creation."
fi

info "Activating virtual environment..."
source venv/bin/activate

# --- STEP 6: Install Python dependencies ---
if [ -f "requirements.txt" ]; then
    info "Installing Python dependencies..."
    pip install -r requirements.txt
else
    error "No requirements.txt found — skipping Python package installation."
    exit 1
fi

# --- STEP 7: Install and Configure Ollama ---
if ! command -v ollama >/dev/null 2>&1; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    warn "Ollama already installed, skipping."
fi
info "Verifying Ollama service status..."
if ! systemctl is-active --quiet ollama; then
    error "Ollama service is not running. Please check the installation."
    exit 1
fi
success "Ollama service is active."

info "Downloading TinyLlama model via Ollama..."
# This command will pull the model if it doesn't exist.
ollama pull tinyllama
success "TinyLlama model is ready."

# --- STEP 8: Download Whisper ONNX models ---
info "Setting up Whisper ONNX models..."

WHISPER_DIR="lib/whisper"
DECODER_URL="https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main/onnx/decoder_model.onnx"
ENCODER_URL="https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main/onnx/encoder_model.onnx"

# Ensure the target directory exists
mkdir -p "$WHISPER_DIR"

download_model() {
    local url="$1"
    local output="$2"
    local filename
    filename=$(basename "$output")

    if [ -f "$output" ]; then
        warn "$filename already exists, skipping download."
    else
        info "Downloading $filename..."
        wget -q --show-progress -O "$output" "$url"
        if [ -s "$output" ]; then
            success "$filename downloaded successfully."
        else
            error "Failed to download $filename. Please check your connection and the URL."
            exit 1
        fi
    fi
}

download_model "$DECODER_URL" "$WHISPER_DIR/decoder_model.onnx"
download_model "$ENCODER_URL" "$WHISPER_DIR/encoder_model.onnx"

# --- STEP 9: Copy and Enable Systemd Service ---
info "Configuring and installing systemd service to run Primer on boot..."
 
PROJECT_PATH="/home/ubuntu/projects/$REPO_NAME"
SERVICE_SRC="primer.service"
SERVICE_DEST="/etc/systemd/system/primer.service"
 
if [ ! -f "$SERVICE_SRC" ]; then
    error "Service file '$SERVICE_SRC' not found in repository. Cannot proceed."
    exit 1
fi
 
info "Copying service file and setting correct project path..."
# Use sed to replace the placeholder __PROJECT_PATH__ with the actual path
# The use of a different separator `|` for sed avoids issues with slashes in the path
sudo sed "s|__PROJECT_PATH__|$PROJECT_PATH|g" "$SERVICE_SRC" > "$SERVICE_DEST"
 
info "Reloading systemd daemon and enabling the Primer service..."
sudo systemctl daemon-reload
sudo systemctl enable primer.service
 
# --- STEP 10: Final instructions ---
success "Installation complete!"
echo
echo "You can start the service manually now with:"
echo "  sudo systemctl start primer.service"
echo
echo "Or reboot the device to have it start automatically:"
echo "  sudo reboot"
echo
echo "To view logs, run:"
echo "  journalctl -u primer.service -f"
