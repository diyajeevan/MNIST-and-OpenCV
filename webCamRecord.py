import cv2
from collections import deque
import numpy as np
from keras.models import load_model

# Load your pre-trained CNN model for digit recognition
model = load_model('cnn-opencv.h5')

def main():
    cap = cv2.VideoCapture(0)
    
    # HSV range for yellow (tuned for the Magic Masala packet yellow)
    Lower_yellow = np.array([20, 100, 100])
    Upper_yellow = np.array([30, 255, 255])
    
    # Define the bounding box (ROI) coordinates and size
    x, y, w, h = 220, 100, 300, 300
    
    # Blackboard for drawing within the ROI
    blackboard_roi = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Deque for tracking the yellow object's center (for drawing)
    pts = deque(maxlen=512)
    
    # This will store the final recognized digit (if any)
    finalDigit = None
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        # Mirror the image for a natural interaction
        img = cv2.flip(img, 1)
        
        # Extract the region of interest (ROI) where drawing takes place
        roi = img[y:y+h, x:x+w]
        
        # Convert ROI to HSV and create mask for yellow
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi_hsv, Lower_yellow, Upper_yellow)
        
        # Reduce noise with slight blurring
        blur = cv2.medianBlur(mask, 5)
        blur = cv2.GaussianBlur(blur, (3, 3), 0)
        
        # Apply binary thresholding (with OTSU)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
        # Find contours within the ROI
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        center = None
        
        if len(cnts) > 0:
            # Use the largest contour (assumed to be the yellow marker)
            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            
            # Accept contour only if area is large enough (tweak threshold if needed)
            if area > 100:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    center = (int(cx), int(cy))
                
                # Optionally scale down the radius for a tighter circle
                radius = radius * 0.5
                
                # Draw the detection circle and center on the ROI for visualization
                cv2.circle(roi, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
                cv2.circle(roi, center, 5, (0, 0, 255), -1)
                
                # Append the detected center to the deque for drawing
                pts.appendleft(center)
                
                # Draw connecting lines (the drawing) on the blackboard and on the ROI
                for i in range(1, len(pts)):
                    if pts[i-1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard_roi, pts[i-1], pts[i], (255, 255, 255), 8)
                    cv2.line(roi, pts[i-1], pts[i], (0, 0, 255), 5)
                    
        else:
            # When no yellow is detected (marker lifted), try to recognize the drawn digit.
            if len(pts) > 0:
                # Preprocess the drawing (blackboard) for recognition
                blackboard_gray = cv2.cvtColor(blackboard_roi, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                
                # Find contours in the blackboard drawing
                cnts2, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(cnts2) > 0:
                    cnt2 = max(cnts2, key=cv2.contourArea)
                    if cv2.contourArea(cnt2) > 2000:
                        xx, yy, ww, hh = cv2.boundingRect(cnt2)
                        digit = blackboard_gray[yy:yy+hh, xx:xx+ww]
                        try:
                            # Resize to 28x28 as required by the CNN model
                            newimg = cv2.resize(digit, (28, 28))
                        except Exception as e:
                            newimg = np.zeros((28,28), dtype=np.uint8)
                        newimg = newimg.reshape(1, 28, 28, 1)
                        newimg = newimg.astype('float32') / 255.0
                        
                        # Predict the digit using the loaded CNN model
                        pred = model.predict(newimg)
                        finalDigit = int(pred.argmax())
                        print("Recognized Digit:", finalDigit)
            
            # Reset the drawing variables
            pts = deque(maxlen=512)
            blackboard_roi = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Overlay the drawing from the blackboard onto the ROI
        roi_with_drawing = cv2.addWeighted(roi, 1, blackboard_roi, 1, 0)
        img[y:y+h, x:x+w] = roi_with_drawing
        
        # Draw the ROI bounding box on the main image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the recognized digit if available
        if finalDigit is not None:
            cv2.putText(img, str(finalDigit), (10, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
        
        cv2.imshow("Frame", img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
