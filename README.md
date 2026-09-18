# 🪰 Fruit Fly Connectome Chess AI

**A biological neural network, digitized and trained to play chess.**

This project uses the open-source fruit fly (*Drosophila melanogaster*) connectome—a mathematical matrix mapping 164,000 biological neurons and 25 million synapses—to play chess. By injecting the state of a chessboard into the fly's simulated optic nerves and reading its descending motor neurons, we use Dopamine Reinforcement Learning to physically rewire the biological pathways into a chess engine.

## 🧠 How it Works

1. **Sensory Injection (The Eyes):** We map the 768 possible piece-on-square combinations to the fly's 8,865 sensory neurons.
2. **Simulated Propagation (The Brain):** The electrical spikes cascade through the 164k neurons in PyTorch.
3. **Motor Readout (The Muscles):** The output is read from descending motor neurons, which correspond to the 4,096 possible physical chess moves. A chess engine filters out illegal "twitches."
4. **Dopamine Rewiring:** Over thousands of simulated games, virtual dopamine rewards the network for successful checks and checkmates, carving new pathways through the biological connectome.

## 🚀 Features
* **PyTorch Biological Simulation:** A neural network architected *exactly* to the fly connectome matrix.
* **Apple Silicon (MPS) Acceleration:** Fast tensor math and tensor cloning optimized for M-series Macs.
* **Web GL Visualizer:** A stunning 3D interface showing the anatomical regions of the brain (Hemispheres, Cerebellum, Brainstem) flashing in real-time as electricity propagates through the network.
* **Kinematic Fly Animation:** A hyper-realistic 3D fly physically swoops down and grabs chess pieces.

## 💻 Running it Locally

1. Install requirements:
   ```bash
   pip3 install flask torch numpy chess
   ```
2. Run the server:
   ```bash
   python3 server.py
   ```
3. Open your browser to `http://localhost:5050` and challenge the fly!

## 🔬 Credits & Data
The core biological data is derived from the connectome mapping research conducted by the Janelia Research Campus and Google Research (FlyWire).
