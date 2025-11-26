#!/usr/bin/env bash
# install.sh — XRP Primer AI Kit Demo setup script
# Tested on Ubuntu Server 22.04+ (Rubik Pi 3)
# Run with: bash install.sh

set -e # Exit on error
set -u # Treat unset vars as errors

# --- COLOR OUTPUT HELPERS ---
function info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
function success() { echo -e "\033[1;32m[SUCCESS]\033[0m $*"; }
function warn()    { echo -e "\033[1;33m[WARN]\033[0m $*"; }
function error()   { echo -e "\033[1;31m[ERROR]\033[0m $*" 1>&2; }

# --- STEP 1: System Update ---
info "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# --- STEP 2: Install System Dependencies ---
info "Installing required packages (git, curl, wget, python3.12, build tools, audio dev libs)..."
sudo apt install -y git curl wget python3.12 python3.12-venv build-essential python3-dev libportaudio2 portaudio19-dev cmake ninja-build libcurl4-openssl-dev

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

# --- STEP 9: Build llama.cpp with OpenCL ---
info "Setting up llama.cpp with OpenCL backend..."
LLM_DEV_DIR="$HOME/dev/llm"
mkdir -p "$LLM_DEV_DIR"

info "Symlinking OpenCL shared library..."
sudo rm -f /usr/lib/libOpenCL.so
sudo ln -s /lib/aarch64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/libOpenCL.so

info "Building OpenCL Headers..."
cd "$LLM_DEV_DIR"
if [ ! -d "OpenCL-Headers" ]; then
    git clone https://github.com/KhronosGroup/OpenCL-Headers
fi
cd OpenCL-Headers
git checkout 5d52989617e7ca7b8bb83d7306525dc9f58cdd46
mkdir -p build && cd build
cmake .. -G Ninja \
    -DBUILD_TESTING=OFF \
    -DOPENCL_HEADERS_BUILD_TESTING=OFF \
    -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="$LLM_DEV_DIR/opencl"
cmake --build . --target install

info "Building OpenCL ICD Loader..."
cd "$LLM_DEV_DIR"
if [ ! -d "OpenCL-ICD-Loader" ]; then
    git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
fi
cd OpenCL-ICD-Loader
git checkout 02134b05bdff750217bf0c4c11a9b13b63957b04
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LLM_DEV_DIR/opencl" \
    -DCMAKE_INSTALL_PREFIX="$LLM_DEV_DIR/opencl"
cmake --build . --target install

info "Symlinking OpenCL headers to /usr/include..."
sudo rm -f /usr/include/CL
sudo ln -s "$LLM_DEV_DIR/opencl/include/CL/" /usr/include/CL

info "Building llama.cpp..."
cd "$LLM_DEV_DIR"
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp
fi
cd llama.cpp
git checkout f6da8cb86a28f0319b40d9d2a957a26a7d875f8c
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_OPENCL=ON
ninja -j"$(nproc)"

info "Adding llama.cpp to PATH in .bash_profile..."
LLAMA_BIN_PATH="$LLM_DEV_DIR/llama.cpp/build/bin"

# Check if the entry already exists to avoid duplicates
if ! grep -q "# Begin llama.cpp" ~/.bash_profile; then
    echo "" >> ~/.bash_profile
    echo "# Begin llama.cpp" >> ~/.bash_profile
    echo "export PATH=\$PATH:$LLAMA_BIN_PATH" >> ~/.bash_profile
    echo "# End llama.cpp" >> ~/.bash_profile
    echo "" >> ~/.bash_profile
    success "llama.cpp path added to .bash_profile."
else
    warn "llama.cpp path already exists in .bash_profile, skipping."
fi

# --- STEP 10: Download and Quantize Qwen2 GGUF Model ---
info "Downloading and quantizing Qwen2-1.5B-Instruct GGUF model..."
LLAMA_CPP_DIR="$LLM_DEV_DIR/llama.cpp"
MODELS_DIR="$LLAMA_CPP_DIR/models"
mkdir -p "$MODELS_DIR"

FP16_MODEL_URL="https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-fp16.gguf"
FP16_MODEL_NAME="qwen2-1_5b-instruct-fp16.gguf"
Q4_MODEL_NAME="qwen2-1_5b-instruct-q4_0-pure.gguf"

cd "$MODELS_DIR"

# Download fp16 model if it doesn't exist
if [ ! -f "$FP16_MODEL_NAME" ]; then
    info "Downloading $FP16_MODEL_NAME..."
    wget -q --show-progress -O "$FP16_MODEL_NAME" "$FP16_MODEL_URL"
    success "$FP16_MODEL_NAME downloaded."
else
    warn "$FP16_MODEL_NAME already exists, skipping download."
fi

# Quantize the model if the quantized version doesn't exist
if [ ! -f "$Q4_MODEL_NAME" ]; then
    info "Quantizing model to Q4_0..."
    # Use the full path to the executable since the PATH is not yet sourced in this session
    "$LLAMA_BIN_PATH/llama-quantize" --pure "$FP16_MODEL_NAME" "$Q4_MODEL_NAME" Q4_0
    success "Model quantized to $Q4_MODEL_NAME."
else
    warn "$Q4_MODEL_NAME already exists, skipping quantization."
fi

# --- STEP 11: Copy and Enable Systemd Service ---
info "Configuring and installing systemd service to run Primer on boot..."
 
PROJECT_PATH="$HOME/projects/$REPO_NAME" # Use dynamic home directory
SERVICE_SRC="primer.service"
SERVICE_DEST="/etc/systemd/system/primer.service"
 
info "Copying service file and setting correct project path..."
sudo sed "s|__PROJECT_PATH__|$PROJECT_PATH|g" "$SERVICE_SRC" | sudo tee "$SERVICE_DEST" > /dev/null
 
# --- STEP 12: Final instructions ---
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
