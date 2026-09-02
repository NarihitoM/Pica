<p align="center">
  <img src="https://raw.githubusercontent.com/NarihitoM/Pica/main/pica/assets/logo.jpg" alt="Pica logo" width="360">
</p>

# Pica

Pica is the project that can control and manage your laptop or pc with the gestures.
Python, MediaPipe hand tracking, and a small PyTorch classifier trained on your own
recorded gestures.

## Install

```
pip install narihito-pica
```

## Quick start

```
narihito-pica collect     # record every gesture, one at a time
narihito-pica train       # train the classifier on what you recorded
narihito-pica run         # start controlling your PC
```

`narihito-pica collect` walks through each gesture in your config and waits for you to press
Enter before recording it, so you can get your hand ready. Press `s` to skip one or
`q` to stop. To redo a single gesture:

```
narihito-pica collect open_palm --replace
```

Recording appends by default, so `--replace` is what you want when a gesture went badly
and you're re-recording it from scratch.

`narihito-pica train` prints how many samples each gesture has and how accurately the
trained model recognises each one, so you can see which gesture needs re-recording:

```
training on:
  close_palm      200 samples
  open_palm       200 samples
  total          1600 samples
accuracy:
  close_palm      98.5%  (197/200)
  open_palm      100.0%  (200/200)
```

## Gestures

| Gesture | Action |
|---|---|
| `open_palm` | move the cursor (tracks your wrist position) |
| `close_palm` | hold to drag -- press and hold moves windows, a quick close is a click |
| `one_finger_up` | volume up |
| `one_finger_down` | volume down |
| `two_finger_up` | scroll up (continuous while held) |
| `two_finger_down` | scroll down (continuous while held) |
| `three_finger_up` | screen brightness up |
| `three_finger_down` | screen brightness down |

Brightness steps by `brightness.step` percent per gesture (default 20) and uses Windows
WMI, so it works on laptop displays.

## Your files

Everything you create lives outside the package, so upgrading Pica never touches it:

```
narihito-pica where
```

```
home:        ~/.pica
config:      ~/.pica/config.yaml
recordings:  ~/.pica/annotations
model:       ~/.pica/gesture_classifier.pth
```

Set `PICA_HOME` to keep several separate setups. The config is copied from the packaged
default the first time you run anything, and is never overwritten after that.

## Commands

| Command | What it does |
|---|---|
| `narihito-pica run` | start gesture control (`q` in the window quits) |
| `narihito-pica collect [gesture]` | record samples; loops through every configured gesture if you don't name one |
| `narihito-pica train` | train on your recordings and save the model |
| `narihito-pica where` | print where your config, recordings and model live |

`narihito-pica collect` takes `--samples`, `--camera` and `--replace`. `narihito-pica train` takes
`--epochs`, `--batch-size` and `--lr`.

## Adding a new gesture

1. Add it to `~/.pica/config.yaml` under `gestures:` (or as `cursor.gesture` if it should
   drive the mouse), mapped to an action.
2. If the action doesn't exist yet, add it to `_run_action` in
   `pica/components/system_control/system_control.py`.
3. `narihito-pica collect <name>`
4. `narihito-pica train`

## Project layout

```
Pica/
├── main.py                       # entry point, same as the `narihito-pica` command
├── pyproject.toml
└── pica/
    ├── cli.py                    # `narihito-pica run` / `collect` / `train` / `where`
    ├── default_config.yaml       # seeded into ~/.pica/config.yaml on first run
    ├── assets/                   # logo + the bundled MediaPipe hand model
    ├── components/               # one folder per feature, each a package with a barrel export
    │   ├── brightness/           # steps display brightness up/down (WMI)
    │   ├── camera_stream/        # threaded webcam reader
    │   ├── classifier/           # PyTorch MLP: landmarks -> gesture name + confidence
    │   ├── hand_tracker/         # MediaPipe HandLandmarker wrapper
    │   └── system_control/       # gesture -> cursor / drag / volume / scroll / brightness
    ├── pipeline/
    │   ├── collect.py            # records landmark samples for one gesture
    │   ├── train.py              # recordings -> trained model
    │   └── run_app.py            # camera -> tracker -> classifier -> system_control
    └── utils/                    # landmark normalization, config loader, paths, overlay
```

Each component folder exports its public class from `__init__.py`, so imports stay flat:
`from pica.components.hand_tracker import HandTracker`.

## Developing

```
git clone https://github.com/NarihitoM/Pica.git
cd Pica
pip install -e .
```

Every module has a `demo()` self-check you can run directly:

```
py -m pica.pipeline.train
py -m pica.components.system_control.system_control
py -m pica.cli --demo
```

## Releasing

CI runs the self-checks on every push. Publishing is automatic:

1. Bump `version` in `pyproject.toml`.
2. Commit and push to `main`.

That's it. The publish workflow wakes up whenever `pyproject.toml` changes, skips if that
version is already on PyPI, and otherwise builds, uploads via Trusted Publishing (no API
token stored anywhere), and tags the commit `v<version>`.

## Troubleshooting

- **`narihito-pica run` says there's no trained model**: run `narihito-pica collect` then `narihito-pica train`.
- **A newly recorded gesture doesn't do anything**: recording only saves samples --
  it does not update the live model. Run `narihito-pica train` after recording.
- **Cursor lags behind your hand**: MediaPipe inference is ~55ms per frame on CPU, which
  caps the loop near 18fps. Lowering `cursor.smoothing` helps a little; the rest is the
  model.
- **Cursor jumps when clicking**: the cursor gesture tracks the wrist (stable), not a
  fingertip (moves as fingers curl) -- if you change `cursor.landmark_index`, keep this
  in mind.
- **A gesture works in testing but misfires live**: the model likely overfit to one
  distance/angle/lighting setup. When recording, move your hand around a bit while
  holding the pose instead of staying perfectly still.
- **Similar gestures get confused** (e.g. one finger vs two fingers): make sure the
  poses are clearly distinct when recording -- ambiguous hand shapes will bleed into
  each other no matter how much data you add.
