# Camera Calibration & Object Measurement

Camera calibration and real-world object measurement using OpenCV. The pipeline calibrates the camera from checkerboard images, then estimates object width in the scene using the calibrated focal length and known distance.

---

## Project Structure

```
cv_module2/
├── calibrate_and_measure.py   # Main script: calibration + measurement
├── calibration_images/        # Checkerboard images (*.jpeg) for calibration
├── object_images/             # Image of object to measure (object.jpeg)
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

**Generated at runtime (not in repo):**
- `calibration_data.npz` — saved camera matrix and distortion coefficients after Step 1

---

## Execution

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

*(If `requirements.txt` is empty, use: `pip3 install opencv-python numpy matplotlib`.)*

### 2. Run the pipeline

```bash
python3 calibrate_and_measure.py
```

**What the script does:**

1. **Step 1 — Camera calibration**
   - Reads checkerboard images from `calibration_images/*.jpeg`
   - Detects inner corners (7×9 grid, 18 mm square size)
   - Computes camera matrix and distortion coefficients (Zhang’s method)
   - Saves results to `calibration_data.npz`

2. **Step 2 — Object measurement**
   - Loads `calibration_data.npz` and uses focal length from the camera matrix
   - Loads `object_images/object.jpeg`, finds the largest contour, and gets bounding box width in pixels
   - Estimates real-world width (cm) using: `width_cm ≈ (pixel_width × distance_cm) / focal_length_px`
   - Prints estimated width, actual width, and percentage error
   - Shows a plot with the detected object bounding box

**Requirements:** At least 10 calibration images with detectable checkerboard corners; object image in `object_images/object.jpeg`. Update `distance_cm` and `actual_width` in the script to match your setup for meaningful error reporting.
