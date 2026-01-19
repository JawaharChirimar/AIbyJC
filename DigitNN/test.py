import cv2
from pathlib import Path
import sys
sys.path.insert(0, str(Path.home() / "Development/AIbyJC/DigitNN"))
from DigitsExtractor import detect_digits_with_contours, detect_background_color_contours

img_path = Path(__file__).parent / "data/input/hw4.png"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

background_mean, foreground_mean = detect_background_color_contours(img)
print(f"Background: {background_mean}, Foreground: {foreground_mean}")

detections = detect_digits_with_contours(img, min_area=30)
print(f"Number of detections: {len(detections)}")
for i, det in enumerate(detections):
    print(f"{i}: {det}")