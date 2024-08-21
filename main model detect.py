from ultralytics import YOLO
import cv2
import math
import cvzone
import torch

# Ensure that the model uses the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    receiver_email = "kunal4103@gmail.com"
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
        server.starttls()
        server.login(sender_email, password)
        server.send_message(message)

    print("Email with tensor image sent successfully!")

model = YOLO(
    r"C:\Users\Kunal\Desktop\RJPOLICE_HACK_238_PIXASHIELD_3-main\gunkaro_Naik\best_infrared (1).pt"
).to(device)  # Load the model on the GPU

# Load the image
img = cv2.imread(
    r"C:\Users\Kunal\Desktop\RJPOLICE_HACK_238_PIXASHIELD_3-main\gunkaro_Naik\2.jpg"
)

# Resize image to be divisible by 32
height, width = img.shape[:2]
new_height = math.ceil(height / 32) * 32
new_width = math.ceil(width / 32) * 32
img_resized = cv2.resize(img, (new_width, new_height))

classNames = [
    "person",
    "dog",
    "drone",
    "fire",
    "car",
]

# Perform inference on the GPU
img_tensor = torch.from_numpy(img_resized).to(device).float()
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to BCHW format
results = model(img_tensor)  # No need for stream=True as it's a single image

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
                sendmail(img_resized)
                email_sent = True

            cvzone.putTextRect(
                img_resized,
                f"{classNames[cls]} {conf}",
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1,
                colorT=(255, 255, 255),
                offset=5,
            )
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 0, 255), 3)

cv2.imshow("Image", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
