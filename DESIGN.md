# Design log

A short record of what this project claimed, what it actually was, and what changed. Written so that a future reader — including me — can tell the difference without reading the whole repository.

## v1 claimed a connectome

The first README described a 164,000-neuron simulation of the *Drosophila* MaleCNS wired to a chessboard through simulated optic nerves, with dopamine reinforcement carving pathways into a PyTorch tensor matrix. The UI said `MALE CNS CONNECTOME · LIVE NEURAL CASCADE` and reported neuron counts, spike counts and simulation time after every move.

## v1 was four linear layers

What was actually in the repo:

```python
self.sensory_to_central = nn.Linear(8865, 512)
self.central_cascade_1  = nn.Linear(512, 512)
self.central_cascade_2  = nn.Linear(512, 512)
self.motor_decoder      = nn.Linear(512, 1)
```

5,065,217 parameters. A board is encoded into an 8,865-dimension vector, every legal move is scored, the highest score wins. No adjacency matrix, no spiking dynamics, no FlyWire download. `NUM_CENTRAL_NEURONS = 153593` was declared in `train_fly.py` and referenced by nothing else in the codebase. The neuron, spike and sim-time numbers in the sidebar were generated from `hash(fen)`. The estimated ELO was `200 + games / 20000 × 500`.

The part that was real: the 3D board, the procedurally modelled fly that picks up a piece and carries it to the destination square, the move encoder, the candidate scoring, the mate-in-one assists, and 9,353 games of actual self-play.

## How the gap survived

The code, the docstrings and the README were produced in one pass with an AI assistant, and I published all three without checking the second against the third. The comments described the intent — `# Simulates the spiking neural dynamics of the Drosophila MaleCNS v1.0 connectome` — and the README was written from the comments rather than from the implementation. The class docstring promised 2,129 motor outputs while the constructor accepted `motor_dim=2129` and silently discarded it. That single unused parameter is the whole bug: a number arrived from a paper, got attached to a signature, and never touched the computation.

Nothing in the pipeline ever compared the claim to the code, because the person who should have compared them was the person who wrote both.

## How it was caught

By opening `train_fly.py` and asking one question: *where does the connectome get loaded?* The answer is nowhere. `grep -r` for `flywire`, `adjacency`, `synapse`, `.h5`, `.npy` across the repo returns comments and UI strings, nothing else. The dataset that the name and the numbers came from — the FlyWire consortium's MaleCNS v1.0 release — is real, public, and not present in this repository in any form.

## What changed

- The README now leads with what is real and states the theatre explicitly.
- The repo description and topics no longer claim a biological network.
- The ELO is described as a counter, because that is what it is.
- `bench.py` exists, and every performance number quoted in the README comes from it and can be re-run.
- MIT licence added, because the old README claimed one that did not exist.

The fake telemetry is still in the UI. It is documented as fake in the README rather than quietly removed, because deleting it would have hidden the mistake instead of correcting it — and because the panel is the only place the project shows its own workings at all.

## What measuring it found

The README's claim that the fly "plays badly" was still an opinion, so `bench.py` was written to replace it with a number. Fifty games against a random mover, thirty against a material-grabber, seed 1, 150-ply cap:

| Opponent | W–D–L | Score | Avg. final material |
|:---|:---|:---|:---|
| Random mover | 1–48–1 | 0.50 | −1.1 |
| Material-greedy | 1–27–2 | 0.48 | −24.1 |

It scores 0.50 against a random mover. Not "roughly" — one win, one loss, forty-eight draws. And a bot that does nothing but grab material finishes twenty-four points up, which is a queen and a rook, while only managing checkmate twice in thirty games because the mate-in-one filter keeps blocking the finish.

The benchmark also found a bug that reading the code had not: the self-play target is the result from **White's** point of view, and `select_best_move` maximises that number. So in a Black position the network is optimising for the opponent. Same weights, same opponents:

| Fly colour | vs random | vs greedy |
|:---|---:|---:|
| White | +2.7 material | −22.7 |
| Black | −5.0 material | −25.5 |

That is the highest-value fix in the repository and it is two lines long. It was invisible until something was measured, which is the whole argument for measuring.

## What would make the claim true

Load the real MaleCNS or FlyWire adjacency matrix, run leaky integrate-and-fire dynamics over the frozen graph, map a small number of sensory pools and descending-neuron readouts, and train *only* the readout. Then publish the controls that make it a result rather than a demo: a degree-preserving rewired graph, a silenced-circuit run, and a plain MLP on the same inputs. Roughly a dozen public projects in this space have already done it; `cobanov/awesome-fly` indexes them.

The honest version of this project is not "the connectome plays chess." It is "a fly picks up your pieces and carries them away, and here is exactly how much of the brain behind it is real."

## The lesson I would keep

Write the claim after the code, not alongside it. A comment is not evidence, a docstring is not a test, and a number that appears in a constructor signature has not necessarily been used.
