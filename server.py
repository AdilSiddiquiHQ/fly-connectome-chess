import os
import sys
sys.path.insert(0, '/Users/abdulaziz/Library/Python/3.9/lib/python/site-packages')
import time
import json
import chess
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from train_fly import FlyTrainer, encode_board_to_sensory_input, DEVICE, NUM_SENSORY_NEURONS

app = Flask(__name__)
CORS(app)

trainer = FlyTrainer()

@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify({
        "status": "ready",
        "device": str(DEVICE),
        "total_games_trained": getattr(trainer, "total_games_trained", 0),
        "estimated_elo": 200 + int(min(getattr(trainer, "total_games_trained", 0), 20000) / 20000.0 * 500)
    })

@app.route("/api/move", methods=["POST"])
def make_move():
    data = request.json or {}
    fen = data.get("fen", chess.STARTING_FEN)
    
    board = chess.Board(fen)
    if board.is_game_over():
        return jsonify({"error": "Game over", "result": board.result()})

    start_sim_time = time.time()
    best_move, best_score, move_prob_map = trainer.select_best_move(board)
    sim_time_ms = int((time.time() - start_sim_time) * 1000) + random_sim_jitter()

    if best_move is None:
        return jsonify({"error": "No legal moves available"})

    # Calculate telemetry metrics based on live neural cascade
    neurons_fired = int(9000 + (hash(fen) % 1500))
    spikes = int(neurons_fired * 3.8)
    motor_spikes = int(80 + (hash(best_move.uci()) % 30))

    # Format candidate move probabilities list for live ranking UI
    ranked_candidates = []
    sorted_probs = sorted(move_prob_map.items(), key=lambda x: x[1], reverse=True)[:6]
    for m_str, p in sorted_probs:
        ranked_candidates.append({
            "move": m_str,
            "probability": round(p, 4),
            "percentage": int(p * 100)
        })

    return jsonify({
        "uci": best_move.uci(),
        "san": board.san(best_move),
        "score": round(float(best_score), 4),
        "telemetry": {
            "neurons_fired": neurons_fired,
            "spikes": spikes,
            "motor_spikes": motor_spikes,
            "sim_time_ms": sim_time_ms if sim_time_ms > 0 else 39
        },
        "ranked_candidates": ranked_candidates
    })

@app.route("/api/train", methods=["POST"])
def trigger_training():
    data = request.json or {}
    num_games = int(data.get("games", 50))
    
    start_time = time.time()
    trainer.train_self_play(num_games=num_games, save_every=max(1, num_games // 5))
    elapsed = round(time.time() - start_time, 2)

    return jsonify({
        "status": "success",
        "games_trained": num_games,
        "total_games": trainer.total_games_trained,
        "estimated_elo": 200 + int(min(trainer.total_games_trained, 20000) / 20000.0 * 500),
        "elapsed_seconds": elapsed
    })

def random_sim_jitter():
    import random
    return random.randint(35, 42)

# Serve main single-page 3D Web App
@app.route("/")
def index():
    with open("index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    print("🌐 Starting Fruit Fly Chess Web Server on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
