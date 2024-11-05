import cv2
import numpy as np
import mediapipe as mp
import math

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Known coin diameter in mm (e.g., US quarter = 24.26 mm)
real_coin_diameter_mm = 24

def detect_coin(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles (coins) using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=10, maxRadius=100)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle for visualization
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            return 2 * r
    return None

def calculate_scale_factor(coin_diameter_pixels, real_coin_diameter_mm):
    return real_coin_diameter_mm / coin_diameter_pixels

def detect_hand(image):
    # Initialize MediaPipe Hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                return hand_landmarks
    return None

def get_distance_in_pixels(landmark1, landmark2, image_width, image_height):
    x1, y1 = int(landmark1.x * image_width), int(landmark1.y * image_height)
    x2, y2 = int(landmark2.x * image_width), int(landmark2.y * image_height)
    
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_hand_metrics(hand_landmarks, image_height, image_width):
    # Wrist to middle fingertip (hand length)
    hand_length_pixels = get_distance_in_pixels(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                                                image_width, image_height)

    # Wrist to index fingertip (trigger distance)
    trigger_distance_pixels = get_distance_in_pixels(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                                                     hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                                                     image_width, image_height)

    # Thumb tip to pinky tip (grip length)
    grip_length_pixels = get_distance_in_pixels(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                                                image_width, image_height)
    
    return hand_length_pixels, trigger_distance_pixels, grip_length_pixels

def main():
    # Load the image
    image = cv2.imread("18.jpg")
    image_height, image_width, _ = image.shape
    
    # Step 1: Detect the coin
    coin_diameter_pixels = detect_coin(image)
    
    if coin_diameter_pixels is None:
        print("No coin detected.")
        return
    
    # Step 2: Calculate the scale factor
    scale_factor = calculate_scale_factor(coin_diameter_pixels, real_coin_diameter_mm)
    
    # Step 3: Detect hand and get landmarks
    hand_landmarks = detect_hand(image)
    
    if hand_landmarks is None:
        print("No hand detected.")
        return
    
    # Step 4: Calculate hand metrics in pixels
    hand_length_pixels, trigger_distance_pixels, grip_length_pixels = get_hand_metrics(hand_landmarks, image_height, image_width)
    
    # Step 5: Convert metrics to mm
    hand_length_mm = hand_length_pixels * scale_factor
    trigger_distance_mm = trigger_distance_pixels * scale_factor
    grip_length_mm = grip_length_pixels * scale_factor
    
    print(f"Hand length: {hand_length_mm:.2f} mm")
    print(f"Trigger distance: {trigger_distance_mm:.2f} mm")
    print(f"Grip length: {grip_length_mm:.2f} mm")
    
    # Display the image with landmarks
    cv2.imshow("Hand and Coin Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
