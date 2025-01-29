import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Function to preprocess the frame (grayscale, blurring, and color filtering)
def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Adjust these values based on the currency color
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(cleaned_mask, 100, 200)
    return edges

# Function to detect watermark using template matching
def detect_watermark(note_roi):
    watermark_template = cv2.imread('watermark_template.png', 0)  # Load watermark template image
    if watermark_template is None:
        print("Error: Watermark template image not found!")
        return False

    note_gray = cv2.cvtColor(note_roi, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(note_gray, watermark_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)

    print(f"Watermark detection: {'Found' if len(loc[0]) > 0 else 'Not Found'}")
    return len(loc[0]) > 0

# Function to detect security threads using template matching
def detect_security_threads(note_roi):
    security_thread_template = cv2.imread('security_threads_template.png', 0)  # Load security thread template image
    if security_thread_template is None:
        print("Error: Security thread template image not found!")
        return False

    note_gray = cv2.cvtColor(note_roi, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(note_gray, security_thread_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)

    print(f"Security thread detection: {'Found' if len(loc[0]) > 0 else 'Not Found'}")
    return len(loc[0]) > 0

# Function to detect RBI seal using template matching
def detect_rbi_seal(note_roi):
    rbi_seal_template = cv2.imread('rbi_seal_template.png', 0)  # Load RBI seal template
    if rbi_seal_template is None:
        print("Error: RBI seal template image not found!")
        return False

    note_gray = cv2.cvtColor(note_roi, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(note_gray, rbi_seal_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)

    print(f"RBI Seal detection: {'Found' if len(loc[0]) > 0 else 'Not Found'}")
    return len(loc[0]) > 0

# Function to detect denomination (₹500) using template matching
def detect_denomination(frame, region_of_interest):
    roi = frame[region_of_interest[1]:region_of_interest[3], region_of_interest[0]:region_of_interest[2]]
    denomination_template = cv2.imread('denom_500.png', 0)  # Load ₹500 template image
    if denomination_template is None:
        print("Error: ₹500 denomination template image not found!")
        return None

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_roi, denomination_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)

    print(f"Denomination detection: {'Found' if len(loc[0]) > 0 else 'Not Found'}")
    return "500" if len(loc[0]) > 0 else None

def detect_fake_currency(frame, roi=None):
    # If a specific ROI is defined, crop the frame
    if roi:
        frame = frame[roi[1]:roi[3], roi[0]:roi[2]]  # Cropping the frame using ROI

    edges = preprocess_frame(frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_notes = 0  # To count the number of detected notes

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Adjusted area threshold
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        note_roi = frame[y:y + h, x:x + w]

        # Check for RBI seal
        rbi_seal_status = "RBI Seal Found" if detect_rbi_seal(note_roi) else "No RBI Seal"

        # Check for watermark
        watermark_status = "Watermark Found" if detect_watermark(note_roi) else "No Watermark"
        security_threads_status = "Security Thread Found" if detect_security_threads(note_roi) else "No Security Thread"

        # Check for denomination
        region_of_interest = (int(x + 0.1 * w), int(y + 0.1 * h), int(x + 0.9 * w), int(y + 0.4 * h))
        denomination = detect_denomination(frame, region_of_interest)

        denomination_status = f"Denomination: ₹{denomination}" if denomination else "Denomination Not Detected"

        # Combine all status messages
        note_status = f"{rbi_seal_status}, {watermark_status}, {denomination_status}"
        cv2.putText(frame, note_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        detected_notes += 1  # Increment detected notes count

    # Annotate the total number of detected notes
    cv2.putText(frame, f"Detected ₹500 Notes: {detected_notes}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame

# In the GUI class, we can set an ROI to focus on a specific region (e.g., a certain part of the camera frame).
class CurrencyDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Label to show video feed
        self.video_source = 0  # Use the first webcam
        self.cap = cv2.VideoCapture(self.video_source)

        # Check if the camera opened correctly
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.window.quit()

        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Define ROI for the currency detection (e.g., the central area of the frame)
        self.roi = (200, 100, 800, 600)  # Example ROI, adjust based on your requirements.

        self.update_video_frame()

        self.window.mainloop()

    def update_video_frame(self):
        ret, frame = self.cap.read()

        if ret:
            processed_frame = detect_fake_currency(frame, roi=self.roi)

            # Convert OpenCV frame to PIL Image and then to ImageTk
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

            # Update the canvas with the new image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to the image object

        self.window.after(10, self.update_video_frame)

# Initialize and run the GUI
root = tk.Tk()
app = CurrencyDetectorApp(root, "Fake Currency Detection with ₹500 Notes")
