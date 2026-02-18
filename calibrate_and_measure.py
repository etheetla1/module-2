import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# =========================
# STEP 1: CAMERA CALIBRATION
# =========================

print("\n=== STEP 1: CAMERA CALIBRATION ===")

# Checkerboard configuration (MUST match your board)
CHECKERBOARD = (7, 9)     # inner corners (columns, rows)
SQUARE_SIZE = 18          # mm (2.5 cm)

# Path to calibration images
calib_path = "calibration_images/*.jpeg"

# Prepare real-world object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D real-world points
imgpoints = []  # 2D image points

images = glob.glob(calib_path)

print(f"Found {len(images)} calibration images")

if len(images) == 0:
    raise RuntimeError("No calibration images found. Check folder and file extensions.")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"[ERROR] Could not read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"[OK] Corners detected in {fname}")
    else:
        print(f"[FAIL] No corners detected in {fname}")

print(f"\nTotal valid calibration images used: {len(objpoints)}")

if len(objpoints) < 10:
    raise RuntimeError(
        "Not enough valid calibration images. "
        "At least 10 are recommended."
    )

# Calibrate the camera
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistortion Coefficients:\n", distCoeffs)

# Save calibration data
np.savez(
    "calibration_data.npz",
    cameraMatrix=cameraMatrix,
    distCoeffs=distCoeffs
)

print("\nCalibration data saved to calibration_data.npz")

# =========================
# STEP 2: OBJECT MEASUREMENT
# =========================

print("\n=== STEP 2: REAL-WORLD MEASUREMENT ===")

# Load calibration data
data = np.load("calibration_data.npz")
cameraMatrix = data["cameraMatrix"]

focal_length = cameraMatrix[0, 0]
print("Focal Length (pixels):", focal_length)

# Load object image
obj_img_path = "object_images/object.jpeg"
img = cv2.imread(obj_img_path)

if img is None:
    raise RuntimeError("Object image not found. Check object_images/object.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest)

# Known experimental values
distance_cm = 210    # 2.5 meters (measured)
actual_width = 21.0  # cm (measure your object)

# Perspective projection equation
estimated_width = (w * distance_cm) / focal_length
error = abs(estimated_width - actual_width) / actual_width * 100

print("\nEstimated Width (cm):", estimated_width)
print("Actual Width (cm):", actual_width)
print("Percentage Error:", error, "%")

# =========================
# VISUALIZATION
# =========================

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected Object for Measurement")
plt.axis("off")
plt.show()

print("\n=== SCRIPT COMPLETED SUCCESSFULLY ===")