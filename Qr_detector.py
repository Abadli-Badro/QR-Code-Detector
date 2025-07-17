import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Set camera properties for better detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

detector = cv2.QRCodeDetector()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("QR Code detector started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Convert to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)
    
    data, bbox, _ = detector.detectAndDecode(frame)
    
    if bbox is None or data == "":
        print("No QR code detected in color frame, trying grayscale...")
        data, bbox, _ = detector.detectAndDecode(gray)
    
    if bbox is None or data == "":
        print("No QR code detected in grayscale, applying Gaussian blur...")
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        data, bbox, _ = detector.detectAndDecode(blurred)
    
    if bbox is not None and len(bbox) > 0:
        # Reshape bbox if needed
        if len(bbox.shape) == 3:
            bbox = bbox[0]
        
        bbox = bbox.astype(int)
        
        for i in range(len(bbox)):
            pt1 = tuple(bbox[i])
            pt2 = tuple(bbox[(i + 1) % len(bbox)])
            cv2.line(frame, pt1, pt2, color=(0, 255, 0), thickness=3)
        
        for point in bbox:
            cv2.circle(frame, tuple(point), 5, (255, 0, 0), -1)

        if data and data.strip():
            print(f"QR Code detected: {data}")
            
            text_x = int(bbox[0][0])
            text_y = int(bbox[0][1] - 10)
            
            if text_y < 30:
                text_y = int(bbox[0][1] + 30)
            
            text_size = cv2.getTextSize(data, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                         (text_x + text_size[0], text_y + 5), (0, 0, 0), -1)
            
            # Put text
            cv2.putText(frame, data, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            break
    
    cv2.putText(frame, "Point camera at QR code - Press 'q' to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('QR Code Detector', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("QR Code detector stopped.")