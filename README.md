<p align="center">
  <img src="https://raw.githubusercontent.com/NarihitoM/Pica/main/src/pica/assets/logo.jpg" alt="Pica logo" width="360">
</p>

# Pica

Pica lets you control your laptop with hand gestures. Wave to move the cursor, close your
palm to drag a window, hold two fingers up to scroll.

It watches your webcam with MediaPipe, turns your hand into 21 tracked points, and feeds
those to a small PyTorch model. The important part: the model is trained on *your* hands,
from recordings you make yourself. Nobody's hand looks quite like yours, and a model
trained on someone else's always feels slightly off.

Windows only for now, mostly because the brightness control talks to WMI.

## Install

```
pip install narihito-pica
```

## Quick start

Three commands, in this order:

```
narihito-pica collect     # record every gesture, one at a time
narihito-pica train       # train the model on what you recorded
narihito-pica run         # start controlling your PC
```

`collect` walks you through the gestures one by one. It waits for you to press Enter before
each one so you have a second to get your hand ready, then opens the camera and tells you
on screen which pose to hold and how many samples it still needs. Press `s` to skip a
gesture, `q` to stop.

Recording takes about a minute per gesture. Move your hand around a little while you hold
the pose. Closer, further, tilted, off to one side. If you record every sample from one
frozen position, the model learns that exact position and gets confused the moment you sit
differently.

Messed one up? Record just that one again:

```
narihito-pica collect open_palm --replace
```

`--replace` matters here. Recording appends by default, so without it you keep the bad
samples and add more on top.

Then train. It tells you what it learned from:

```
training on:
  close_palm      200 samples
  open_palm       200 samples
  total          1600 samples
accuracy:
  close_palm      98.5%  (197/200)
  open_palm      100.0%  (200/200)
```

Anything much below the others is the gesture to re-record.

## Gestures

| Gesture | Action |
|---|---|
| `open_palm` | move the cursor (tracks your wrist) |
| `close_palm` | hold to drag, quick close to click |
| `one_finger_up` | volume up |
| `one_finger_down` | volume down |
| `two_finger_up` | scroll up (keeps going while held) |
| `two_finger_down` | scroll down |
| `three_finger_up` | brightness up |
| `three_finger_down` | brightness down |

Brightness moves by `brightness.step` percent each time (20 by default) and goes through
Windows WMI, so it works on laptop screens.

Drag is worth calling out: hold `close_palm` and the mouse button stays down, so you can
actually pick a window up and move it. Let go and it releases. A quick open-close reads as
a plain click.

## Your files

Your recordings, config and trained model live in your home folder, not inside the
installed package. Upgrading Pica never touches them:

```
narihito-pica where
```

```
home:        ~/.pica
config:      ~/.pica/config.yaml
recordings:  ~/.pica/annotations
model:       ~/.pica/gesture_classifier.pth
```

The config is copied from the packaged default the first time you run anything, and never
overwritten after that, so your edits stick. Set `PICA_HOME` if you want to keep more than
one setup side by side.

## Commands

| Command | What it does |
|---|---|
| `narihito-pica run` | start gesture control (`q` in the window quits) |
| `narihito-pica collect [gesture]` | record samples; loops through every gesture if you don't name one |
| `narihito-pica train` | train on your recordings and save the model |
| `narihito-pica where` | print where your files live |

`collect` takes `--samples`, `--camera` and `--replace`. `train` takes `--epochs`,
`--batch-size` and `--lr`. The defaults are fine unless something's going wrong.

## Adding your own gesture

1. Add it to `~/.pica/config.yaml` under `gestures:`, mapped to an action. If it should
   drive the mouse instead, set it as `cursor.gesture`.
2. If the action doesn't exist yet, add it to `_run_action` in
   `src/pica/components/system_control/system_control.py`.
3. `narihito-pica collect <name>`
4. `narihito-pica train`

Step 4 is the one people forget. Recording only writes samples to disk, it doesn't touch
the model you're running.

## Project layout

```
Pica/
├── .github/workflows/            # CI (type check + tests + build) and PyPI publish
├── LICENSE
├── README.md
├── pyproject.toml
├── tests/                        # unittest suite, nothing to install
└── src/
    └── pica/
        ├── __main__.py           # `python -m pica`
        ├── cli.py                # run / collect / train / where
        ├── default_config.yaml   # seeded into ~/.pica/config.yaml on first run
        ├── assets/               # logo + the bundled MediaPipe hand model
        ├── components/           # one folder per feature, each with a barrel export
        │   ├── brightness/       # steps the display up and down via WMI
        │   ├── camera_stream/    # threaded webcam reader
        │   ├── classifier/       # PyTorch MLP: landmarks to gesture name + confidence
        │   ├── hand_tracker/     # MediaPipe HandLandmarker wrapper
        │   └── system_control/   # gesture to cursor / drag / volume / scroll / brightness
        ├── pipeline/
        │   ├── collect.py        # records landmark samples for one gesture
        │   ├── train.py          # recordings to trained model
        │   └── run_app.py        # camera to tracker to classifier to system_control
        └── utils/                # landmark normalization, config, paths, overlay
```

Each component folder exports its class from `__init__.py`, so imports stay short:
`from pica.components.hand_tracker import HandTracker`.

## Developing

```
git clone https://github.com/NarihitoM/Pica.git
cd Pica
pip install -e .
```

Run the tests:

```
python -m unittest discover -s tests -t .
```

Standard library only, no test framework to install. They point `PICA_HOME` at a temp
folder so your own recordings are never touched, and they mock `pyautogui`, so nothing
runs off with your real cursor mid-test.

Most modules also carry a `demo()` self-check you can run on its own:

```
python -m pica.pipeline.train
python -m pica.components.system_control.system_control
python -m pica.cli --demo
```

## What's new

### 0.4.2

Tests and docs. There's now a `unittest` suite covering the parts that break quietly:
landmark normalization, where your files land, append versus replace when recording,
label ordering in the dataset, and the cursor and drag logic with `pyautogui` mocked out.
The drag-release bug that used to leave the mouse button stuck when your hand left the
frame has a test pinning it now.

No behaviour changes. If it works for you today it works the same after upgrading.

### 0.4.1

Packaging fix. The logo and links on the PyPI page pointed at paths that stopped existing
when the package moved to a `src/` layout.

### 0.4.0 - guided recording

`collect` now puts the instructions on the camera feed itself: which pose to hold, whether
it's actually recording, how far along it is, and what's coming next. Before this the
window was a bare webcam feed and you had to keep glancing at the terminal to know what
Pica wanted from you.

It also draws the hand skeleton while recording, so you can see tracking drop out the
moment it happens instead of finding out at training time.

### 0.3.0 - know what you trained

`train` started reporting sample counts and accuracy per gesture. When a gesture misfires
live it's almost always one with fewer samples than the rest, or one the model keeps
confusing with its neighbour. Both are now obvious before you run anything.

Also moved to a `src/` layout, so what you import is what's installed rather than whatever
happens to be sitting in the working directory.

### 0.2.0 - stable command name

The console script became `narihito-pica`, matching the name on PyPI.

### 0.1.0 - first release

Installable from PyPI with `collect`, `train`, `run` and `where`.

Your recordings, config and model live in `~/.pica`, outside the package, so upgrading
never touches your data and reinstalling never loses it. `PICA_HOME` moves that if you
want more than one set.

## Troubleshooting

**It says there's no trained model.** You skipped a step. `narihito-pica collect`, then
`narihito-pica train`.

**A gesture I just recorded does nothing.** Recording only saves samples, it doesn't
update the running model. Train again.

**The cursor lags behind my hand.** MediaPipe takes roughly 55ms per frame on CPU, which
caps the loop somewhere near 18fps. Dropping `cursor.smoothing` helps a bit, but most of
the delay is the model and there's no setting that removes it.

**The cursor jumps when I click.** It tracks your wrist, which stays put, rather than a
fingertip, which swings around as your fingers curl. Worth remembering if you change
`cursor.landmark_index`.

**A gesture works fine while I'm testing but misfires in real use.** The model overfit to
one distance, angle or lighting setup. Re-record it and move around more while you hold
the pose.

**Two gestures keep getting confused.** Usually one finger versus two. If the poses look
similar to you they look similar to the model, and no amount of extra data fixes an
ambiguous pose. Pick shapes that are clearly different.
