from ultralytics import YOLO
import cv2
import math
import cvzone
import os
import random

# Load a model
email_sent = False


def sendmail(image):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email import encoders
    from PIL import Image
    import numpy as np
    import io

    sender_email = "pixashield@gmail.com"
    receiver_email = "ishannaik7@gmail.com"

    password = "bjxc lysq nizf qmle"

    # Create a message object
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "ANOMALY DETECTED!!!"

    # Email body
    body = "This is the body of the email."
    message.attach(MIMEText(body, "plain"))

    # Convert tensor to image
    tensor_data = image  # Replace with your tensor
    image = Image.fromarray(tensor_data)

    # Save image to a byte stream
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="PNG")
    image_byte_array.seek(0)

    # Attach image to the email
    img = MIMEImage(image_byte_array.read())
    img.add_header("Content-Disposition", "attachment", filename="image.png")
    message.attach(img)

    # Establish a connection with the SMTP server (for Gmail, use 'smtp.gmail.com' and port 587)
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        # Start TLS for security
        server.starttls()

        # Login to your email account
        server.login(sender_email, password)

        # Send the email
        server.send_message(message)

    print("Email with tensor image sent successfully!")


model = YOLO(r"C:\Users\ishan\Desktop\RAJASTHAN\best_infrared.pt")

images_folder = r"C:\Users\ishan\Desktop\RAJASTHAN\test\images"

classNames = [
    "person",
    "dog",
    "drone",
    "fire",
    "car",
]

# Get a list of image files in the folder
image_files = [
    f
    for f in os.listdir(images_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
]

# Use an iterator to go through images one by one
image_iterator = iter(image_files)

while True:
    try:
        # Get the next image file
        image_file = next(image_iterator)
        img_path = os.path.join(images_folder, image_file)

        # Load the image
        img = cv2.imread(img_path)

        # Perform object detection
        results = model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                currentClass = classNames[cls]
                print(currentClass)
                if conf > 0.2:
                    if currentClass in classNames:
                        # Uncomment the following line if you want to send an email
                        # sendmail(img)
                        email_sent = True

                    cvzone.putTextRect(
                        img,
                        f"{classNames[cls]} {conf}",
                        (max(0, x1), max(35, y1)),
                        scale=1,
                        thickness=1,
                        colorT=(255, 255, 255),
                        offset=5,
                    )
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        cv2.imshow("Image", img)
        key = cv2.waitKey(50)  # Display each image for 3 seconds

        if key == 27:  # Break the loop if 'Esc' key is pressed
            break

    except StopIteration:
        # End of images, break the loop
        break

cv2.destroyAllWindows()
