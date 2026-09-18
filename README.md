<div align="center">

# 🪰 Drosophila Connectome Chess AI

**A true biological neural network, digitized and trained to play chess.**

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS_Accelerated-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*The line between biology and computer science is gone. We took the literal open-source mapping of a fruit fly brain (164,000 neurons, 25 million synapses) and hijacked its optic and motor nerves to play chess via Dopamine Reinforcement Learning.*

[**Watch the Viral Breakdown**](#) • [**Explore the Connectome**](https://flywire.ai) • [**Installation**](#-quick-start)

---

</div>

## 📖 Table of Contents
- [The Mad Science](#-the-mad-science)
- [How it Actually Works](#-how-it-actually-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Credits & Origin](#-credits--origin)

---

## 🧬 The Mad Science

A few years ago, researchers at Google and Janelia Research Campus took the physical brain of a fruit fly, stained it with heavy metals, and sliced it with a precision diamond knife into nanometer-thin sheets. Using advanced computer vision, they mapped every single neuron and synapse into a 3D digital schematic. 

**This project downloads that exact biological schematic and turns it into an active PyTorch network.** 

By wiring a digital chessboard into the fly's simulated optic nerves and reading its descending motor neurons, we use Dopamine Reinforcement Learning to physically rewire the biological pathways into a chess engine.

---

## ⚙️ How it Actually Works

This project utilizes a standard **Three-Step Biological Bridge**:

### 1. Sensory Mapping (The Eyes)
We map the 768 possible piece-on-square combinations (64 squares × 12 piece types) to a specific cluster of the fly's 8,865 sensory neurons. When a White Knight lands on F3, we inject a `1.0` voltage spike into that exact neuron. The fly literally *feels* the board as an electrical pattern.

### 2. Simulated Propagation (The Brain & Dopamine)
The raw connectome provides the wiring diagram, but not the synaptic strength. We initialize the pathways with baseline weights. As the sensory electricity cascades chaotically through the 164,000 neurons, we simulate games. When the fly makes a good move, a virtual **Dopamine Reward** triggers, permanently carving exact pathways into the tensor matrix.

### 3. Motor Readout (The Muscles)
The electricity reaches the descending motor neurons (the nerves that normally flap wings). We map these to the 4,096 possible physical chess moves (Start Square ➔ End Square). A chess engine acts as a survival filter, ignoring illegal "twitches" and executing the strongest legal muscle contraction.

---

## 🛠 Tech Stack

| Domain | Technology | Purpose |
|---|---|---|
| **Deep Learning** | `PyTorch` | Simulates the 164k neuron matrix (MPS Apple Silicon Accelerated). |
| **Backend API** | `Flask` & `Python 3` | Bridges the browser visualization to the biological tensor calculations. |
| **Frontend UI** | `Three.js` & `WebGL` | Renders the hyper-realistic fly, HDRI board, and live brain holograms. |
| **Logic Filter** | `chess.js` | Filters the chaotic motor neuron output for legal chess moves. |

---

## 📁 Project Structure

```text
📦 fly-connectome-chess
 ┣ 📂 __pycache__/           # Compiled Python files
 ┣ 📜 index.html             # Main 3D WebGL Frontend (Three.js)
 ┣ 📜 server.py              # Flask API Server (The Bridge)
 ┣ 📜 train_fly.py           # Core PyTorch Biological Network
 ┣ 📜 weights.pt             # The Trained Synaptic Memory (2k+ Games)
 ┣ 📜 metadata.json          # Training Milestone Tracker
 ┣ 📜 requirements.txt       # Python Dependencies
 ┗ 📜 README.md              # You are here
```

---

## 🚀 Quick Start

Want to challenge a biological connectome to a game of chess on your own machine? 

### Prerequisites
You will need Python 3.9+ and pip installed. If you are on an M-Series Mac, PyTorch will automatically use the `mps` hardware acceleration.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fly-connectome-chess.git
cd fly-connectome-chess
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requires: `torch`, `flask`, `numpy`, `chess`)*

### 3. Boot the Brain
Start the Flask server to load the `weights.pt` synaptic memory into RAM.
```bash
python server.py
```

### 4. Play the Game
Open your web browser and navigate to:
```text
http://localhost:5050
```
> **Note:** Hit the `+1000 Games` button in the UI if you want to push the fly back into the hyper-bolic time chamber and carve more dopamine pathways!

---

## 🔬 Credits & Origin

The core structural data for the neuron mapping is derived from the incredible, open-source work of the **Janelia Research Campus** and **Google Research** via the [FlyWire](https://flywire.ai) project. 

This project transforms their static connectome schematic into an active, reward-based Reinforcement Learning environment.

<div align="center">
  <br>
  <i>If you found this mind-blowing, consider dropping a ⭐ on the repo!</i>
</div>
