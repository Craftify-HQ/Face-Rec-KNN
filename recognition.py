import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def knn(X, y, z, k=1):
  d = np.sum((X - z) ** 2, axis=1)
  idx = np.argsort(d)[:k]
  cls, vote = np.unique(y[idx], return_counts=True)
  return cls[np.argmax(vote)]


y = []
X = []
for f in os.listdir():
  if os.path.isdir(f) and not f.startswith('.'):
    for i in os.listdir(f):
      if i.endswith('.jpg'):
        img = cv2.imread(f + '/' + i, cv2.IMREAD_GRAYSCALE)
        if img is not None:
          img_resized = cv2.resize(img, (100, 100))
          X.append(img_resized.flatten())
          y.append(f)

X = np.array(X)
y = np.array(y)
print('Shape ของข้อมูล X:', X.shape)
print('รายชื่อใน Dataset:', np.unique(y))

cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  if not ret or frame is None:
    break

  x1, y1, x2, y2 = 800, 100, 1200, 650

  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

  face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

  face = cv2.equalizeHist(face)

  face_resized = cv2.resize(face, (100, 100))
  z = face_resized.flatten()

  if len(X) > 0:
    name = knn(X, y, z)
    cv2.putText(
        frame,
        str(name),
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

  cv2.imshow('Face Recognition', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()