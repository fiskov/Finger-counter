import logging
import os
import cv2 as cv
import mediapipe as mp
import time
import math

def get_fingers_count(hand):
    ps = [(lm.x, lm.y, lm.z) for lm in hand.landmark]

    # numbers from https://mediapipe.dev/images/mobile/hand_landmarks.png
    # thumb-finger
    tf = 0
    if math.dist(ps[17], ps[4]) > math.dist(ps[17], ps[3]):
        tf = 1

    # other fingers
    middle_points = [7, 11, 15, 19]
    end_points = [8, 12, 16, 20]
    cnt = sum(1 for i in range(len(middle_points))
              if math.dist(ps[0], ps[middle_points[i]]) < math.dist(ps[0], ps[end_points[i]])
              )

    return cnt + tf


# show FPS overlay
def draw_text_fps(img,
          font=cv.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0)
          ):

    if not hasattr(draw_text_fps, "frame_time_prev"):
        draw_text_fps.frame_time_prev = 0
    if not hasattr(draw_text_fps, "fps_filter"):
        draw_text_fps.fps_filter = 0

    frame_time_new = time.time()
    fps = 1/(frame_time_new - draw_text_fps.frame_time_prev)
    draw_text_fps.frame_time_prev = frame_time_new

    alpha = 0.95
    draw_text_fps.fps_filter = draw_text_fps.fps_filter * alpha + fps * (1-alpha)

    text = f"{draw_text_fps.fps_filter:.1f}"
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

# show fingers overlay
def draw_text_fingers_count(img, fingers_count_1, fingers_count_2,
          font=cv.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=2,
          font_thickness=3,
          text_color=(0, 255, 0)
          ):
    text = f"{fingers_count_1} + {fingers_count_2} = {fingers_count_1+fingers_count_2}"
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


if __name__ == "__main__":

    logging.basicConfig()
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.INFO)

    logger.info("Start")
    time_start = time.time()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            fingers_count_1 = 0
            fingers_count_2 = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                if len(results.multi_hand_landmarks) > 1:
                    fingers_count_1 = get_fingers_count(results.multi_hand_landmarks[0])
                    fingers_count_2 = get_fingers_count(results.multi_hand_landmarks[1])

            # Flip the image horizontally for a selfie-view display.
            image = cv.flip(image, 1)
            draw_text_fps(image)
            draw_text_fingers_count(image, fingers_count_1, fingers_count_2, pos=(0,30))
            cv.imshow('Calc fingers', image)

            # save image to disk
            # if (time.time() - time_start) > 1:
            #     cv.imwrite(time.strftime("%Y%m%d-%H%M%S")+".png", image)
            #     time_start = time.time()

            if cv.waitKey(5) & 0xFF == 27:
                break

    cap.release()
