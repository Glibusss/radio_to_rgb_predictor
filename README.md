# radio_to_rgb_predictor

This repository now focuses on a single practical baseline:

- input: optical image in `data/raw/optical`;
- optional reference radar screen in `data/raw/radar`;
- output: synthesized radar image in `data/processed/synthesis`;
- output: cleaned radar reference in `data/processed/reference`;
- method: physics-aware optical-to-radar baseline that encodes RCS, forest attenuation, radar geometry and aspect dependence.

## Layout

```text
data/
  raw/
    optical/
    radar/
  processed/
    debug/
    reference/
    synthesis/
docs/
  research.md
radar_synthesis/
  physics.py
  radar_display.py
scripts/
  extract_radar_reference.py
  prepare_training_chunks.py
  synthesize_example.py
```

## Run

```powershell
.\venv\Scripts\python.exe scripts\synthesize_example.py
.\venv\Scripts\python.exe scripts\extract_radar_reference.py
.\venv\Scripts\python.exe scripts\prepare_training_chunks.py
```

## Chunked training layout

```text
data/
  chunks/
    paired/
      train/
        optical/
        radar/
      val/
        optical/
        radar/
      metadata.json
    unpaired_radar/
```

`paired` is for supervised pretraining on synthetic targets.
`unpaired_radar` is a real radar patch bank that can later feed a discriminator, perceptual texture regularizer or domain adaptation stage.

Train chunks are augmented by default with `orig`, `rot90`, `rot180` and `flip_lr`.

Optional:

```powershell
.\venv\Scripts\python.exe scripts\synthesize_example.py --meters-per-pixel 1.5 --seed 42
```

## What the baseline models

- different effective RCS for built-up targets, roads, open ground and vegetation;
- lower forest response deeper inside dense vegetation;
- X-band response with 9.3 GHz wavelength and 50 W transmit power;
- 3 m antenna height, 1 deg beam width and 1.5 m range resolution;
- aspect sensitivity from local scene structure relative to radar look direction.

## What it does not solve yet

- end-to-end learned translation with CondGAN or diffusion;
- precise radar UI cleaning and registration for real paired supervision;
- calibrated radiometry against a specific Naida hardware revision.
