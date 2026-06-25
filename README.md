# shot-detection-51div

## Project Structure

### python_poc

`python_poc` contains the Python proof-of-concept pipeline.

`python_poc/main.py` receives an input image path, optionally applies target warping, optionally applies detection, and saves the final image to the given output path.

Example:

```bash
python python_poc/main.py \
  /path/to/input.jpg \
  /path/to/output.png \
  --warp true \
  --detection none
```

Main options:

```python
parser.add_argument("--warp", type=parse_bool, default=True, help="Whether to apply perspective warping.")
parser.add_argument("--detection", choices=["none", "yolo", "algorithm"], default="algorithm", help="Detection mode.")
```

- `--warp true`: apply perspective warping before detection/save.
- `--warp false`: skip warping and use the original input image for the next step.
- `--detection none`: do not run detection. The image is saved after the optional warping step.
- `--detection yolo`: run YOLO model inference.
- `--detection algorithm`: run algorithm-based detection.

### front_poc

`front_poc` should be implemented after `python_poc` is completed. It can use the completed Python POC features, including image warping and detection, as the backend or reference behavior for the frontend implementation.
