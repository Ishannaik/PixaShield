from ultralytics import YOLO
import cv2
import math
import cvzone
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
    message["Subject"] = "GUN_DETECTED!!!"

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
    


model = YOLO("best.pt")  # load a pretrained model (recommended for training)
cap = cv2.VideoCapture(r"air_gun.mp4")  # For Video
classNames = ["gun", "gun"]
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            
            currentClass = classNames[cls]
            print(currentClass)
            if conf>0.2:
                if currentClass =='gun':
                    myColor = (0, 0,255)
                    if email_sent == False:
                        sendmail(img)
                        email_sent = True

                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    
