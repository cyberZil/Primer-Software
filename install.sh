#!/usr/bin/env bash
# install.sh — XRP Primer AI Kit Demo setup script
# Tested on Ubuntu 22.04+ (Rubik Pi 3)
# Run with: bash install.sh

# FIXED: Added command-line screen blanking adjustment, auto-login configuration, model download, and ensured LF line endings.
set -e # Exit on error
set -u # Treat unset vars as errors

# --- COLOR OUTPUT HELPERS ---
info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# --- STEP 1: Disable Screen Blanking via CLI ---
info "Disabling screen blanking and auto-suspend for installation longevity (set to 'Never')..."
# Set idle-delay to 0 (Never) for the current GNOME session.
gsettings set org.gnome.desktop.session idle-delay 0
# Also disable screen locking to prevent interruptions
gsettings set org.gnome.desktop.screensaver lock-enabled false
success "Screen blanking disabled."


# --- STEP 2: Configure Auto-Login on Boot ---
info "Configuring GDM3 for automatic login for user '$(whoami)'..."
USER_TO_AUTOLOGIN=$(whoami)
# Use sed to enable AutomaticLoginEnable and set the target user in the GDM custom.conf
sudo sed -i '/^#AutomaticLoginEnable=/c\AutomaticLoginEnable=true' /etc/gdm3/custom.conf
sudo sed -i "/^#AutomaticLogin=/c\AutomaticLogin=$USER_TO_AUTOLOGIN" /etc/gdm3/custom.conf
success "Auto-login set for next reboot. (User: $USER_TO_AUTOLOGIN)"


# --- STEP 3: System Update (COMMENTED OUT AS REQUESTED) ---
# info "Updating system packages..."
# sudo apt update && sudo apt upgrade -y

# --- STEP 4: Install dependencies ---
info "Installing required packages (git, curl, build tools, python3.12, audio dev libs, venv)..."
sudo apt install -y git curl wget python3.12 python3.12-venv build-essential python3-dev libportaudio2 portaudio19-dev

# --- STEP 5: Create projects folder (if missing) ---
PROJECTS_DIR="$HOME/projects"
if [ ! -d "$PROJECTS_DIR" ]; then
    info "Creating projects directory at $PROJECTS_DIR"
    mkdir -p "$PROJECTS_DIR"
fi
cd "$PROJECTS_DIR"

# --- STEP 6: Clone repo if missing ---
REPO_URL="https://github.com/SaintSampo/Primer-Software"
REPO_NAME="Primer-Software"
if [ ! -d "$REPO_NAME" ]; then
    info "Cloning repository..."
    git clone "$REPO_URL"
else
    warn "Repository already exists, skipping clone."
fi
cd "$REPO_NAME"

# --- STEP 7: Set up Python venv ---
if [ ! -d "venv" ]; then
    info "Creating Python 3.12 virtual environment..."
    python3.12 -m venv venv
else
    warn "Virtual environment already exists, skipping creation."
fi

info "Activating virtual environment..."
source venv/bin/activate

# --- STEP 8: Install Python dependencies ---
if [ -f "requirements.txt" ]; then
    info "Installing Python dependencies..."
    pip install -r requirements.txt
else
    warn "No requirements.txt found — skipping."
fi

# --- STEP 9: Install Ollama if missing ---
if ! command -v ollama >/dev/null 2>&1; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    warn "Ollama already installed, skipping."
fi

# --- STEP 10: Download TinyLlama Model ---
info "Downloading TinyLlama model via Ollama..."
# This command automatically runs in the background thanks to the Ollama install script
ollama pull tinyllama
success "TinyLlama model download initiated."


# --- STEP 11: Download Whisper models automatically ---
info "Setting up Whisper ONNX models..."

WHISPER_DIR="lib/whisper"
DECODER_URL="https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main/onnx/decoder_model.onnx"
ENCODER_URL="https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main/onnx/encoder_model.onnx"

mkdir -p "$WHISPER_DIR"

download_model() {
    local url="$1"
    local output="$2"

    if [ -f "$output" ]; then
        warn "$(basename "$output") already exists, skipping download."
    else
        info "Downloading $(basename "$output")..."
        wget -q --show-progress -O "$output" "$url"
    fi

    if [ -s "$output" ]; then
        success "$(basename "$output") downloaded successfully."
    else
        error "Failed to download $(basename "$output"). Please check your connection."
        exit 1
    fi
}

download_model "$DECODER_URL" "$WHISPER_DIR/decoder_model.onnx"
download_model "$ENCODER_URL" "$WHISPER_DIR/encoder_model.onnx"

# --- STEP 12: Final instructions ---
success "Installation complete!"
echo
echo "Activate your Python environment with:"
echo "  source venv/bin/activate"
echo
echo "Then run the demo with:"
echo "  sudo ./venv/bin/python src/primer.py"
echo
warn "You may need to log out and back in for GPIO group changes to take effect. The system is also configured for automatic login on next boot."
