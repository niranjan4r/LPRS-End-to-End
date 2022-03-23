import cv2

def apply_preprocessing(image):
    # gaussian = cv2.GaussianBlur(image, (7, 7), 0)
    # gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_img = clahe.apply(equalized)
    return preprocessed_img