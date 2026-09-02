# Pica — Agent Handoff

Gesture-controlled PC using Python, MediaPipe hand tracking, and a trained
PyTorch classifier that maps hand landmarks to gestures, which drive
`pyautogui` actions (cursor, clicks, media keys).

## Status
Scaffolding complete: all modules exist and each has a runnable `demo()`
self-check (`python -m src.components.<module>`), but **no gesture data has
been recorded and no model has been trained yet**. `models/gesture_classifier.pth`
does not exist — `main.py` will fail to start until it does.

## Setup
```
pip install -r requirements.txt
```

## Next steps (in order)
1. Open `notebooks/01_collect_data.ipynb`, run `collect('<label>')` once per
   gesture (labels must match the keys under `gestures:` in
   `config/config.yaml`, plus `point` for the cursor-drive gesture). Aim for
   150-300 samples per label, varying hand angle/distance slightly.
2. Open `notebooks/02_train_model.ipynb`, run all cells. It reads
   `data/annotations/*.npy`, normalizes landmarks, trains a small MLP, and
   writes `models/gesture_classifier.pth`.
3. Run `python main.py`. A debug window shows the hand skeleton and the
   live predicted gesture + confidence. Confirm each gesture in
   `config/config.yaml` triggers the right action; `q` quits.

## Layout
- `config/config.yaml` — gesture→action map, camera id, confidence threshold, cooldown.
- `src/components/` — `camera_stream.py` (threaded capture), `hand_tracker.py`
  (MediaPipe wrapper), `classifier.py` (`GestureNet` model + `GestureClassifier`
  loader/predictor), `system_control.py` (gesture → pyautogui action, with cooldown
  and cursor smoothing).
- `src/utils/` — `landmarks.py` (normalize/flatten, shared by training + inference),
  `config.py` (yaml loader), `visualizer.py` (debug overlay).
- `src/pipeline/run_app.py` — wires camera → tracker → classifier → system_control.
- `notebooks/` — data collection and training, see Next steps above.
- `data/`, `models/*.pth` are gitignored (user-specific recordings/weights).

## Known gaps / not yet built
- **Auto-start on boot is intentionally deferred** — do not add a Startup
  folder shortcut, service, or scheduled task unless the user asks for it in
  a future request. This was an explicit decision, not an oversight.
- No `swipe_left`/`swipe_right` motion-based classification — the current
  classifier is single-frame (one landmark snapshot → one gesture), so swipes
  would need to be trained as a static hand shape or reworked as a
  velocity/trajectory feature over a few frames if true "swipe" motion
  detection is wanted.
- No packaging/executable — runs as a plain Python script.

## Conventions
- No comments unless explaining a non-obvious WHY.
- Every non-trivial module ends with a `demo()` under `if __name__ ==
  "__main__":` — an assert-based smoke check, not a test framework. Keep
  this pattern for any new module.
- Config-driven, not hardcoded: gesture→action mapping, thresholds, and
  camera id all live in `config/config.yaml`, not in code.
