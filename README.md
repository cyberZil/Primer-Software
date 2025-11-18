# XRP Primer AI Kit Demo

## Project Description

This repository provides a guide for setting up a local Large Language Model (LLM) serving environment on a Rubik Pi 3 (RPi3), utilizing Ubuntu Server, Python 3.12, and the Ollama framework.

## TODO:

Soon:
- LLM running on the NPU.
- solve the user permissions problem.
- look into S3 Bucket for redistributing large models
- launch the demo as a service during boot.
- look into how photonvision does[ software distribution]([url](https://github.com/PhotonVision/photon-image-modifier/blob/main/.github/workflows/main.yml))
- look into python [uv]([url](https://docs.astral.sh/uv/))

Future Plans:
- Serialized display writes should be possible for instant llm output
- [improving the speech to text]([url](https://aihub.qualcomm.com/iot/models/whisper_small_quantized?domain=Audio&useCase=Speech+Recognition&chipsets=qualcomm-qcs6490-proxy)) (or with Voxel?).
- RAG retrieval.

## 1. Initial Hardware Setup: Ubuntu OS on the Rubik Pi 3

The first step is to follow [this guide](https://www.thundercomm.com/rubik-pi-3/en/docs/rubik-pi-3-user-manual/1.0.0-u/Update-Software/3.2.Flash-using-Qualcomm-Launcher) to install Ubuntu Server on the Rubik Pi 3. Then follow [these instructions](https://www.thundercomm.com/rubik-pi-3/en/docs/rubik-pi-3-user-manual/1.0.0-u/Troubleshooting/11.3.ubuntu-desktop-vs-server/#switch-from-ubuntu-desktop-version-to-server-version) to switch to ubuntu desktop.

**Abridged OS install instructions**

Please read the official documentation linked above very closely. That said, here is summary of the steps required
- Put the Rubik Pi in EDL mode
- Flash Ubuntu with Qualcomm-Launcher
	- default username: ubuntu
	- default password: ubuntu
	- you will be asked to change the default password
	- reccomended new password: xrpaikit
	- you will be asked to setup a connection to a local wifi network (important)
 	- In your terminal, run the command Qualcomm-Launcher gives you to SSH into the Rubik Pi
- Once you Are SShed into the Pi, Log in to the ubuntu user
- Go to the next steps, run those commands on the PI

## 2. Setting Up the Development Environment

After logging in, execute the following commands to install necessary dependencies and clone the project.

**System Update:**
```bash
sudo apt update
sudo apt upgrade -y
```

**Install Git:**
```bash
sudo apt install git -y
```
**(Best practice) create a projects folder in your **
```
mkdir -p projects
cd projects
```

**Clone the Project:**
```bash
git clone https://github.com/XRP-AI-Kit/Primer-Software
cd Primer-Software
```

### 2.2. Installing Python 3.12

**Install Python 3.12 and Venv Module:**
```bash
sudo apt install python3.12 python3.12-venv -y
```

**Verify Installation:**
```bash
python3.12 --version
```

**Essential Installs:**
```bash
sudo apt install -y build-essential python3-dev libportaudio2 portaudio19-dev
```

### 2.3. Installing Ollama

Ollama is the core LLM serving application. Its installer script automatically handles architecture detection for the ARM platform.

**Execute the Ollama Installer:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify Service Status:**
Check that the Ollama service is running in the background.
```bash
systemctl status ollama
```

**Run a Test Model:**
Download a lightweight model to confirm functionality. (Due to RPi3's 1GB RAM, always choose small, highly quantized models like `tinyllama`).
```bash
ollama run tinyllama
```
Type `/bye` to exit the interactive session.

## 2.4 Whisper Model Setup

**1. Download the Models:**
Download the whisper models by running the following commands:
```bash
wget -P lib/whisper https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main/onnx/decoder_model.onnx
wget -P lib/whisper https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main/onnx/encoder_model.onnx
```

## 3. Python Development Environment

### 3.0 install system requirement library

```bash
sudo usermod -a -G gpio ubuntu
```

### 3.1. Creating a Virtual Environment (venv)

Isolating dependencies in a virtual environment is essential for project stability.

**Create Venv using Python 3.12:**
```bash
python3.12 -m venv venv
```

**Activate the Virtual Environment:**
```bash
source venv/bin/activate
```
> **Note**: Your command prompt will change (e.g., `(venv)user@host:~...`) indicating the environment is active. You must run this command every time you open a new terminal session.

### 3.2. Installing Dependencies

Install all necessary libraries defined in the `requirements.txt` file.

**Install Packages:**
```bash
pip install -r requirements.txt
```

The Rubik Pi 3 is now fully configured.

### 3.3. Run the Demo

Now that the virtual enviornment is active with all the needed dependancies, you can test speech to text by running:
```bash
./venv/bin/python src/whisper_prompt.py
```
or the language model with:
```bash
./venv/bin/python src/ollama_chat.py
```
if you want to test the display, you can with:
```bash
sudo ./venv/bin/python src/render_avatar.py
```
