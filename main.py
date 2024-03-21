import cv2
import time
import base64
import os
import requests
import threading
from dotenv import load_dotenv


load_dotenv()


# Important (.......Download the cv2 )
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera (change the argument to 0 for the default camera)
cap = cv2.VideoCapture(0)

# Set the frame width and height
cap.set(3, 640)
cap.set(4, 480)

# Variable to store the last capture time
last_capture_time = time.time()

# Folder to store captured face images
save_images_folder = 'images' # change the Folder Path
os.makedirs(save_images_folder, exist_ok=True)  # Create the folder if it doesn't exist

# API URL and Authentication Token
# api_image_url = "http://localhost:7200/image-url" # Change the Localhost into IP address(only for raspberry pi)
# api_save_url = "http://localhost:7200/images"
api_image_url =os.getenv("IMAGE_URL")
api_save_url =os.getenv("UPLOAD_IMAGE_URL")


# Function to process images and print API responses
def process_images_and_print_output(image_path, api_image_url, api_save_url):
    with open(image_path, "rb") as image_file:
        files = {'image': image_file}
        # headers = {'Authtoken': f'{auth_token}'}


        response = requests.post(api_image_url, files=files)
        file_url = response.json().get('data', {}).get('fileUrl')

        print(f"Image: {image_path}, API Response: {response.text}")

        if response.status_code == 201:
            image_url = file_url
            print("Image URL---------", image_url)

            # Make a POST request to save the image URL to the database
            save_response = requests.post(api_save_url, json={'imageUrl': image_url, "cameraId": 2})
            print("Save Response", save_response)

            # Check if the save request was successful
            if save_response.status_code == 201:
                print("Image URL saved to database successfully.")
            else:
                print("Failed to save image URL to database.")

def capture_images():
    global last_capture_time

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            expand_factor = 0.2
            expanded_x = max(0, int(x - w * expand_factor))
            expanded_y = max(0, int(y - h * expand_factor))
            expanded_w = min(frame.shape[1], int(w * (1 + 2 * expand_factor)))
            expanded_h = min(frame.shape[0], int(h * (1 + 2 * expand_factor)))

            face_img = frame[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w]

            current_time = time.time()
            if current_time - last_capture_time >= 1:
                img_filename = f"{save_images_folder}/face_{int(current_time)}.jpg"
                cv2.imwrite(img_filename, face_img)

                last_capture_time = current_time

                # Create a new thread to process the captured image using the function
                threading.Thread(target=process_images_and_print_output, args=(img_filename, api_image_url, api_save_url)).start()

        cv2.imshow('Face Detection', frame)

        # Add the function to delete old images
        delete_old_images(save_images_folder)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to delete images older than 2 minutes
def delete_old_images(folder_path):
    current_time = time.time()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            creation_time = os.path.getctime(file_path)
            if current_time - creation_time > 120:
                os.remove(file_path)
                print(f"Deleted old image: {file_path}")

# Create a thread for capturing images
capture_thread = threading.Thread(target=capture_images)
capture_thread.start()

# Wait for the capture thread to finish before releasing the camera and closing windows
capture_thread.join()
cap.release()
cv2.destroyAllWindows()
