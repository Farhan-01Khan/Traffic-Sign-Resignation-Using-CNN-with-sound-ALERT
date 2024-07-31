import numpy as np
import cv2
import pyttsx3
from tensorflow import keras
import tkinter as tk
from threading import Thread, Event

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
    gray_img = preprocessing(img)
    img = gray_img.reshape(1, 32, 32, 1)
    
    predictions = model.predict(img)
    probabilityValue = np.max(predictions)
    
    if probabilityValue > threshold:
        classIndex = np.argmax(predictions)
        return classIndex, probabilityValue, gray_img
    return None, None, gray_img

def webcam_feed():
    consecutive_frames = 5
    detections = []

    while not stop_event.is_set():
        # Read image from camera
        success, imgOriginal = cap.read()
        
        # Detect sign
        classIndex, probabilityValue, gray_img = detect_sign(imgOriginal)
        
        if classIndex is not None:
            detections.append(classIndex)
            if len(detections) >= consecutive_frames:
                if detections.count(classIndex) == consecutive_frames:
                    class_name = getClassName(classIndex)
                    
                    # Display the class and probability on the image
                    cv2.putText(imgOriginal, f"CLASS: { classIndex } { class_name }", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Speak the class name
                    engine.say(class_name)
                    engine.runAndWait()
                detections = []
        else:
            detections = []

        # Show the original and grayscale images
        cv2.imshow("Result", imgOriginal)
        cv2.imshow("Grayscale Image", gray_img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()
    cap.release()

def on_closing():
    stop_event.set()
    root.quit()

# Create a stop event
stop_event = Event()

# Create the GUI window
root = tk.Tk()
root.title("Traffic Sign Recognition")

# Create and place the exit button
exit_button = tk.Button(root, text="Exit", command=on_closing)
exit_button.pack()

# Start the webcam feed in a separate thread
thread = Thread(target=webcam_feed)
thread.start()

# Start the GUI event loop
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
