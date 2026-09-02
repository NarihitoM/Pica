import cv2

from src.components.camera_stream import CameraStream
from src.components.classifier import GestureClassifier
from src.components.hand_tracker import HandTracker
from src.components.system_control import SystemControl
from src.utils.config import load_config
from src.utils.visualizer import draw_landmarks, draw_status


def _predict(classifier, landmarks, handedness):
    """Classifier was trained on one hand's shape -- mirror a left hand onto it for
    prediction only. Callers keep the raw landmarks so cursor direction stays real."""
    class_landmarks = landmarks.copy()
    if handedness == "Left":
        class_landmarks[:, 0] = 1.0 - class_landmarks[:, 0]
    return classifier.predict(class_landmarks)


def run():
    config = load_config()
    threshold = config.get("confidence_threshold", 0.75)
    cursor_threshold = config.get("cursor", {}).get("confidence_threshold", 0.6)
    cursor_gesture = config.get("cursor", {}).get("gesture")
    brightness_gesture = config.get("brightness", {}).get("gesture", "pinch")

    camera = CameraStream(camera_id=config.get("camera_id", 0)).start()
    tracker = HandTracker()
    classifier = GestureClassifier()
    control = SystemControl(config)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            hands = tracker.process_with_handedness(frame)

            gesture, confidence = None, None
            for landmarks, _ in hands:
                draw_landmarks(frame, landmarks)

            predictions = [_predict(classifier, lm, hd) for lm, hd in hands]

            both_pinching = len(hands) >= 2 and all(
                name == brightness_gesture and score >= threshold
                for name, score in predictions[:2]
            )

            if both_pinching:
                level = control.handle_two_hands(hands[0][0], hands[1][0])
                gesture = f"{brightness_gesture} -> brightness {level:.0f}%"
            elif hands:
                landmarks = hands[0][0]
                gesture, confidence = predictions[0]
                gate = cursor_threshold if gesture == cursor_gesture else threshold
                if confidence >= gate:
                    control.handle(gesture, landmarks)

            draw_status(frame, gesture, confidence)
            cv2.imshow("Pica", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        tracker.close()
        control.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
