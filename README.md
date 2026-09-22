# Adaptive Traffic AI

AI-based traffic flow prediction and adaptive signal control. Combines computer
vision, time-series forecasting, and traffic simulation to reduce intersection
wait times versus traditional fixed-timing signals.

## The problem

Most traffic lights run on a fixed timer, regardless of actual road conditions
— one arm of an intersection can be empty while another is jammed, and the
signal doesn't care. Reinforcement-learning approaches to adaptive signal
control exist (see Research below), but published work shows they are purely
**reactive**: they only adjust once congestion has already built up in the
current state. Under heavy traffic, this reactive-only approach can perform
*worse* than a static timer (see Vidali et al., 2019, below — their RL agent's
wait time increased by up to 145% over a static light in their high-traffic
scenario).

## Our approach

We add a **predictive layer** ahead of the control logic: instead of only
reacting to current vehicle counts, we forecast near-future traffic density
(1-5 minutes ahead) using an LSTM, and feed that forecast into the signal
controller. The goal is for the system to start adjusting *before* a queue
forms, not after.

Pipeline: **video → YOLOv8 detection + tracking → vehicle-count-per-second
time series → LSTM density forecast → adaptive signal controller → SUMO
simulation (evaluated against a fixed-timing baseline)**.

## Research this project builds on

- Vidali et al., 2019 (WOA) — *A Deep RL Approach to Adaptive Traffic Lights
  Management*. DQN agent on a single 4-way SUMO intersection; established the
  reward-function design (accumulated waiting time > raw waiting time for
  training stability) and the SUMO-based evaluation methodology we follow.
- Gao et al., 2017 (arXiv:1705.02755) — *Adaptive Traffic Signal Control: DRL
  with Experience Replay and Target Network*. Feeds raw vehicle position/speed
  data directly into a CNN rather than hand-crafted features; reported 47-86%
  delay reduction vs. baseline controllers. Basis for our training-stability
  approach (experience replay + target network).
- UA-DETRAC (Wen et al., 2015) — the vehicle detection/tracking benchmark
  dataset used to train our detector.

## Project phases

- [x] **Phase 0 — Setup.** Repo scaffold, environment, dependencies.
- [x] **Phase 1a — Vehicle detection.** YOLOv8, fine-tuned on UA-DETRAC
      (car/bus/van/others). Current result: mAP50 = 0.77 after 10 epochs
      (`runs/detect/train-2`), model saved at `models/yolov8n_uadetrac_v1.pt`.
- [ ] **Phase 1b — Tracking (in progress).** ByteTrack, via
      `src/detection/track_vehicles.py` — assigns persistent IDs across
      frames so vehicles are counted once, and produces a
      vehicle-count-per-second CSV per sequence.
- [ ] **Phase 2 — Time-series prediction.** Build a density dataset across
      many sequences; ARIMA baseline; LSTM predictor (with time-of-day /
      day-of-week features).
- [ ] **Phase 3 — Signal control + simulation.** Build a 4-way intersection
      in SUMO (synthetic, following Vidali et al.'s design — not from real
      footage); fixed-timing baseline controller; adaptive controller driven
      by the LSTM's predictions; wire the full pipeline together via TraCI.
- [ ] **Phase 4 — Evaluation.** Fixed vs. adaptive comparison across traffic
      scenarios (low/high/NS-heavy/EW-heavy, following Vidali et al.);
      metrics (wait time, queue length, throughput); charts; final report.

**Current status: Phase 1b (tracking), just started.**

## Scope for future work

- Extend beyond a single intersection to a small network of coordinated
  intersections (flagged as future work in Vidali et al. too).
- Try higher-information-density state representations (e.g. image/CNN-based
  state instead of occupancy-cell or count-based state).
- Live webcam/real-camera demo instead of only recorded benchmark footage.
- Explore whether a hybrid LSTM + RL controller (rather than LSTM feeding a
  rule-based controller) performs better once the basic pipeline is proven.

## Repo structure
cat > README.md << 'MDEOF'
# Adaptive Traffic AI

AI-based traffic flow prediction and adaptive signal control. Combines computer
vision, time-series forecasting, and traffic simulation to reduce intersection
wait times versus traditional fixed-timing signals.

## The problem

Most traffic lights run on a fixed timer, regardless of actual road conditions
— one arm of an intersection can be empty while another is jammed, and the
signal doesn't care. Reinforcement-learning approaches to adaptive signal
control exist (see Research below), but published work shows they are purely
**reactive**: they only adjust once congestion has already built up in the
current state. Under heavy traffic, this reactive-only approach can perform
*worse* than a static timer (see Vidali et al., 2019, below — their RL agent's
wait time increased by up to 145% over a static light in their high-traffic
scenario).

## Our approach

We add a **predictive layer** ahead of the control logic: instead of only
reacting to current vehicle counts, we forecast near-future traffic density
(1-5 minutes ahead) using an LSTM, and feed that forecast into the signal
controller. The goal is for the system to start adjusting *before* a queue
forms, not after.

Pipeline: **video → YOLOv8 detection + tracking → vehicle-count-per-second
time series → LSTM density forecast → adaptive signal controller → SUMO
simulation (evaluated against a fixed-timing baseline)**.

## Research this project builds on

- Vidali et al., 2019 (WOA) — *A Deep RL Approach to Adaptive Traffic Lights
  Management*. DQN agent on a single 4-way SUMO intersection; established the
  reward-function design (accumulated waiting time > raw waiting time for
  training stability) and the SUMO-based evaluation methodology we follow.
- Gao et al., 2017 (arXiv:1705.02755) — *Adaptive Traffic Signal Control: DRL
  with Experience Replay and Target Network*. Feeds raw vehicle position/speed
  data directly into a CNN rather than hand-crafted features; reported 47-86%
  delay reduction vs. baseline controllers. Basis for our training-stability
  approach (experience replay + target network).
- UA-DETRAC (Wen et al., 2015) — the vehicle detection/tracking benchmark
  dataset used to train our detector.

## Project phases

- [x] **Phase 0 — Setup.** Repo scaffold, environment, dependencies.
- [x] **Phase 1a — Vehicle detection.** YOLOv8, fine-tuned on UA-DETRAC
      (car/bus/van/others). Current result: mAP50 = 0.77 after 10 epochs
      (`runs/detect/train-2`), model saved at `models/yolov8n_uadetrac_v1.pt`.
- [ ] **Phase 1b — Tracking (in progress).** ByteTrack, via
      `src/detection/track_vehicles.py` — assigns persistent IDs across
      frames so vehicles are counted once, and produces a
      vehicle-count-per-second CSV per sequence.
- [ ] **Phase 2 — Time-series prediction.** Build a density dataset across
      many sequences; ARIMA baseline; LSTM predictor (with time-of-day /
      day-of-week features).
- [ ] **Phase 3 — Signal control + simulation.** Build a 4-way intersection
      in SUMO (synthetic, following Vidali et al.'s design — not from real
      footage); fixed-timing baseline controller; adaptive controller driven
      by the LSTM's predictions; wire the full pipeline together via TraCI.
- [ ] **Phase 4 — Evaluation.** Fixed vs. adaptive comparison across traffic
      scenarios (low/high/NS-heavy/EW-heavy, following Vidali et al.);
      metrics (wait time, queue length, throughput); charts; final report.

**Current status: Phase 1b (tracking), just started.**

## Scope for future work

- Extend beyond a single intersection to a small network of coordinated
  intersections (flagged as future work in Vidali et al. too).
- Try higher-information-density state representations (e.g. image/CNN-based
  state instead of occupancy-cell or count-based state).
- Live webcam/real-camera demo instead of only recorded benchmark footage.
- Explore whether a hybrid LSTM + RL controller (rather than LSTM feeding a
  rule-based controller) performs better once the basic pipeline is proven.

## Repo structure

adaptive-traffic-ai/
├── data/
│ ├── raw/ # UA-DETRAC (gitignored — see Dataset setup below)
│ └── processed/ # YOLO-format dataset, per-second vehicle counts
├── src/
│ ├── detection/ # YOLOv8 training, detection, tracking
│ ├── prediction/ # LSTM / ARIMA density forecasting (Phase 2)
│ ├── control/ # adaptive signal control logic (Phase 3)
│ └── simulation/ # SUMO + TraCI integration (Phase 3)
├── models/ # trained model weights (gitignored)
├── outputs/ # result charts, demo videos, tracking output
└── tests/


## Setup

Requires Python 3.10+. Conda recommended over `venv` on macOS (Apple's system
Python doesn't ship a working pip for `venv`).

```bash
conda create -n traffic-ai python=3.10 -y
conda activate traffic-ai
pip install -r requirements.txt
```

## Dataset setup (UA-DETRAC)

The official UA-DETRAC host (detrac-db.rit.albany.edu) is no longer serving
downloads. Use this Kaggle mirror instead:

1. https://www.kaggle.com/datasets/bratjay/ua-detrac-orig (free Kaggle
   account required)
2. Download and unzip. You should get these top-level folders:
   `DETRAC-Images/`, `DETRAC-Train-Annotations-XML/`,
   `DETRAC-Test-Annotations-XML/`, `DETRAC-MOT-toolkit/`
3. Place (or unzip directly into) `data/raw/ua-detrac/` so the path
   `data/raw/ua-detrac/DETRAC-Images/DETRAC-Images/<SEQUENCE>/img#####.jpg`
   resolves correctly.
4. Build the YOLO-format training set (subsamples every 10th frame,
   converts XML annotations to YOLO label format):
```bash
   python src/detection/prepare_dataset.py
```
   This writes `data/processed/dataset/` (images + labels, train/val split)
   and `data/processed/ua_detrac.yaml` (the config YOLOv8 trains from).

## Running things

**Train the detector** (fine-tune YOLOv8 on the prepared dataset):
```bash
yolo train model=yolov8n.pt data=data/processed/ua_detrac.yaml epochs=10 imgsz=640 batch=16 device=mps
```
(`device=mps` for Apple Silicon GPU; use `device=cpu` if unavailable, but
expect it to be much slower — roughly 3-5x — than `mps`.)

**Run detection on an image:**
```bash
python src/detection/detect_vehicles.py
```

**Track vehicles across a sequence and build a count-over-time CSV:**
```bash
python src/detection/track_vehicles.py MVI_20011
```
(swap `MVI_20011` for any sequence folder name under
`data/raw/ua-detrac/DETRAC-Images/DETRAC-Images/`)

## Current results

- Detector: mAP50 = 0.77, mAP50-95 = 0.57 (10 epochs, YOLOv8n, 7,164 train /
  1,023 val images). Per-class: car 0.85, bus 0.88, van 0.75, others 0.60
  (weakest class, only 32 training examples — expected).
- Loss was still decreasing at epoch 10 without plateauing, suggesting a
  longer run (50+ epochs) would improve on this further.
