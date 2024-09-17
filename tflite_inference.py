import numpy as np
import tensorflow as tf
from PIL import Image
from picamera2 import Picamera2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='my_mobilenet_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the camera
camera = Picamera2()

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize the image as per your model input size
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_image(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # The model output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)
    confidence_score = output_data[0][predicted_label]
    return predicted_label, confidence_score

# Function to capture an image from the camera and classify
def capture_and_classify():
    image_path = '/home/pi/captured_image.jpg'
    camera.capture(image_path)  # Capture the image
    preprocessed_image = preprocess_image(image_path)
    label, confidence = classify_image(preprocessed_image)
    print(f'Detected object label: {label}, Confidence: {confidence}')

capture_and_classify()
