import numpy as np
import cv2
import pyttsx3
from tensorflow import keras
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Camera settings
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.90  # Adjusted for better accuracy
font = cv2.FONT_HERSHEY_SIMPLEX

# Setup the video camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load the trained model
model = keras.models.load_model('model.h5')

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

def getClassName(classNo):
    classes = ['Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
               'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
               'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
               'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
               'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
               'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
               'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
               'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
               'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
               'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
               'End of no passing by vehicles over 3.5 metric tons']
    return classes[classNo]

def detect_sign(img):
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    
    predictions = model.predict(img)
    probabilityValue = np.max(predictions)
    
    if probabilityValue > threshold:
        classIndex = np.argmax(predictions)
        return classIndex, probabilityValue
    return None, None

consecutive_frames = 5
detections = []

display_start_time = 0
display_duration = 2  # Display duration in seconds

while True:
    # Read image from camera
    success, imgOriginal = cap.read()
    
    # Detect sign
    classIndex, probabilityValue = detect_sign(imgOriginal)
    
    if classIndex is not None:
        detections.append(classIndex)
        if len(detections) >= consecutive_frames:
            if detections.count(classIndex) == consecutive_frames:
                class_name = getClassName(classIndex)
                
                # Display the class and probability on the image
                cv2.putText(imgOriginal, f"CLASS: {classIndex} {class_name}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                if probabilityValue is not None:
                    cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Speak the class name
                engine.say(class_name)
                engine.runAndWait()
                
                # Record the start time to display the text for a specific duration
                display_start_time = time.time()
            detections = []
    else:
        detections = []

    # Display the class and probability for a set duration
    if time.time() - display_start_time < display_duration:
        if classIndex is not None:
            class_name = getClassName(classIndex)
            cv2.putText(imgOriginal, f"CLASS: {classIndex} {class_name}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            if probabilityValue is not None:
                cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Result", imgOriginal)

    # Break the loop if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
cap.release()
