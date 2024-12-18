import cv2

# Load the pre-trained classifier for frontal face detection
f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture from the default camera
capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    _, img = capture.read()
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = f_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('img', img)
    
    # Break the loop on pressing the 'ESC' key
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the capture and close the window
capture.release()
cv2.destroyAllWindows()
