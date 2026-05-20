from ultralytics import YOLO
import cv2

model = YOLO("runs/classify/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    results = model(frame)

    result = results[0]

    names = result.names
    probs = result.probs.data.tolist()

    index = probs.index(max(probs))

    prediction = names[index]
    confidence = probs[index]

    text = f"{prediction} {confidence:.2f}"

    cv2.putText(frame,text,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Blade Inspection",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()