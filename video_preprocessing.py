import cv2

def apply_preprocessing(image):
    # gaussian = cv2.GaussianBlur(image, (7, 7), 0)
    # gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_img = clahe.apply(equalized)
    return preprocessed_img

cap = cv2.VideoCapture("./test.avi")
while not cap.isOpened():
    cap = cv2.VideoCapture("./test.avi")
    cv2.waitKey(1000)
    print ("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
size = (int(cap.get(3)), int(cap.get(4)))

print(size)
print(cap.get(5))
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
result = cv2.VideoWriter('output.avi', fourcc, 30, size, 0)

while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        newframe = apply_preprocessing(frame)
        result.write(newframe)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # print (str(pos_frame) + " frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
        print ("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
cap.release()
result.release()