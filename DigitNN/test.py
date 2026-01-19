import cv2
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from DigitsExtractor import detect_digits_with_contours, detect_background_color_contours, extract_and_process_region, sort_detections_by_reading_order
from DigitClassifierALL import load_or_create_digit_classifier, classify_digit

img_path = Path(__file__).parent / "data/input/hw4.png"
img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

model_path = Path(__file__).parent / "data/modelForDE/run_sigmoid_aug2x_4conv_64dense_newData/digit_classifier_final.keras"
model = load_or_create_digit_classifier(train_model=False, classifier_model_path=str(model_path))

background_mean, foreground_mean = detect_background_color_contours(img)
detections = detect_digits_with_contours(img, min_area=30)
sorted_detections = sort_detections_by_reading_order(detections, img.shape[0])

for i, (line_num, digit_num, box) in enumerate(sorted_detections):
    processed = extract_and_process_region(img, box, background_mean=background_mean, foreground_mean=foreground_mean)
    digit, conf = classify_digit(model, processed)
    print(f"{i}: Digit={digit}, Conf={conf:.6f}")