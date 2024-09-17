from collections import deque
import time
import cv2
from neural_network.cameraManager import CameraManager

from neural_network.detectedBlock import DetectedBlock

def camera_main(config, camera_queue: deque):
    # initialize the camera
    myCamHandler = CameraManager()
    myCamHandler.setup()

    freq = cv2.getTickFrequency()
    
    # Let camera warm up
    time.sleep(0.3)

    running = True
    while running:
        start=time.time()

        # Capture frame
        frame = myCamHandler._camera.capture_array()
        camera_queue.append(frame)

        time.sleep(abs(1/config.camera.fps-(time.time()-start)))

        # Exit on 'q' key
        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Quitting")
            break

def object_recognition(config, camera_queue: deque, recognized_object_queue: deque):
    
   def object_recognition(config, camera_queue: deque, recognized_object_queue: deque):
    while True:
        if len(camera_queue) > 0:
            # Get the latest frame from the queue
            frame = camera_queue.pop()

            # Process the frame (e.g., convert to grayscale)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Example processing: Edge detection
            edges = cv2.Canny(gray_frame, 100, 200)

            # Display the processed image
            cv2.imshow("Processed Image", edges)

            # Dummy detection logic (can be replaced with your own object recognition logic)
            if detect_something(edges):
                recognized_object = DetectedBlock(position=(100, 200), size=(50, 50))  # Dummy data
                recognized_object_queue.append(recognized_object)
                print(f"Detected object: {recognized_object}")

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

def detect_something(image):
    """
    Detects objects in an image using contour detection.
    This function finds contours in the image and detects objects based on the size of the contours.
    
    Args:
        image (numpy.ndarray): The input image (grayscale or binary).
    
    Returns:
        bool: True if an object is detected, False otherwise.
        list: A list of contours representing the detected objects.
    """
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store valid contours
    detected_contours = []

    # Loop over the contours to filter out small/big contours (noise)
    for contour in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours by size (e.g., only consider contours with a minimum size)
        if 100 < cv2.contourArea(contour) < 10000:  # Adjust these values based on your use case
            # Draw the bounding box on the image (for visualization purposes)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Add the contour to the detected list
            detected_contours.append(contour)

    # Check if we detected any valid contours
    if len(detected_contours) > 0:
        return True, detected_contours  # Return True if objects are detected along with the contours
    else:
        return False, []  # No objects detected
