import cv2

data = 'haarcascade_frontalface_default.xml'
# data = 'haarcascade_profileface.xml'
# data = 'haarcascade_russian_plate_number.xml'

faces_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + data)


# img = cv2.imread('faces.jpeg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = faces_cascades.detectMultiScale(img_gray)
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

# cv2.imshow("Result", img)

# cv2.waitKey(0)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if success:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faces_cascades.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Result", frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'): break