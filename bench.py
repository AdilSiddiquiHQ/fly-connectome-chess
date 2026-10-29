"""
Benchmark the fly against fixed baselines.

    python bench.py --opponent random  --games 200
    python bench.py --opponent greedy  --games 100
    python bench.py --opponent greedy2 --games 50

Every number the README quotes comes from this file, so it can be re-run and
disagreed with. Nothing here is tuned to make the fly look good.

Opponents
    random   uniform choice over legal moves
    greedy   captures the highest-value piece available right now, 0-ply,
             random tie-break. Knows nothing about safety.
    greedy2  1-ply: best capture minus the opponent's best immediate reply.

Scoring
    win 1.0, draw 0.5, loss 0.0, reported with a 95% Wilson interval.
    Games that hit --max-plies are adjudicated as draws and counted separately.
"""

import argparse
import json
import random
import time

import chess

from train_fly import FlyTrainer

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material(board, color):
    total = 0
    for _, piece in board.piece_map().items():
        v = PIECE_VALUE[piece.piece_type]
        total += v if piece.color == color else -v
    return total


def capture_gain(board, move):
    gain = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        gain = PIECE_VALUE[victim.piece_type] if victim else 1
    if move.promotion:
        gain += PIECE_VALUE[move.promotion]
    return gain


def random_move(board):
    return random.choice(list(board.legal_moves))


def greedy_move(board):
    best, best_gain = [], -1
    for move in board.legal_moves:
        gain = capture_gain(board, move)
        if gain > best_gain:
            best_gain, best = gain, [move]
        elif gain == best_gain:
            best.append(move)
    return random.choice(best)


def greedy2_move(board):
    best, best_score = [], None
    for move in board.legal_moves:
        score = capture_gain(board, move)
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        reply = 0
        for answer in board.legal_moves:
            reply = max(reply, capture_gain(board, answer))
        board.pop()
        score -= reply
        if best_score is None or score > best_score:
            best_score, best = score, [move]
        elif score == best_score:
            best.append(move)
    return random.choice(best)


OPPONENTS = {"random": random_move, "greedy": greedy_move, "greedy2": greedy2_move}


def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def play_game(trainer, fly_color, opponent, max_plies, opening=None):
    board = chess.Board()
    if opening:
        for uci in opening.split():
            board.push_uci(uci)

    plies = 0
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if board.turn == fly_color:
            move, _, _ = trainer.select_best_move(board)
            if move is None:
                break
        else:
            move = opponent(board)
        board.push(move)
        plies += 1

    if board.is_checkmate():
        winner = not board.turn
        score = 1.0 if winner == fly_color else 0.0
        outcome = "win" if winner == fly_color else "loss"
    else:
        score = 0.5
        outcome = "draw" if not board.is_game_over(claim_draw=True) else "draw"

    return {
        "outcome": outcome,
        "score": score,
        "plies": plies,
        "material": material(board, fly_color),
        "capped": board.ply() >= max_plies and not board.is_game_over(claim_draw=True),
        "final_fen": board.fen(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", choices=sorted(OPPONENTS), default="random")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    trainer = FlyTrainer()
    opponent = OPPONENTS[args.opponent]

    results = []
    started = time.time()
    for i in range(args.games):
        fly_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        r = play_game(trainer, fly_color, opponent, args.max_plies)
        r["fly_color"] = "white" if fly_color == chess.WHITE else "black"
        results.append(r)
        if (i + 1) % 10 == 0 or i + 1 == args.games:
            w = sum(1 for x in results if x["outcome"] == "win")
            d = sum(1 for x in results if x["outcome"] == "draw")
            l = sum(1 for x in results if x["outcome"] == "loss")
            print(f"[{i+1}/{args.games}] W{w} D{d} L{l} "
                  f"({time.time() - started:.0f}s)", flush=True)

    n = len(results)
    wins = sum(1 for x in results if x["outcome"] == "win")
    draws = sum(1 for x in results if x["outcome"] == "draw")
    losses = sum(1 for x in results if x["outcome"] == "loss")
    score = sum(x["score"] for x in results) / n
    lo, hi = wilson(score * n, n)
    capped = sum(1 for x in results if x["capped"])
    avg_material = sum(x["material"] for x in results) / n
    up3 = sum(1 for x in results if x["material"] >= 3) / n
    down3 = sum(1 for x in results if x["material"] <= -3) / n

    summary = {
        "opponent": args.opponent,
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(score, 4),
        "score_ci95": [round(lo, 4), round(hi, 4)],
        "capped_games": capped,
        "avg_plies": round(sum(x["plies"] for x in results) / n, 1),
        "avg_final_material": round(avg_material, 2),
        "pct_up_3_or_more": round(up3, 4),
        "pct_down_3_or_more": round(down3, 4),
        "seconds": round(time.time() - started, 1),
        "seed": args.seed,
        "max_plies": args.max_plies,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "games": results}, f, indent=2)


if __name__ == "__main__":
    main()
