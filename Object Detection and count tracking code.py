from ultralytics import YOLO
import cv2

# Load YOLOv5 'u' model
model = YOLO("yolov5su.pt")
print("Model loaded successfully.")

# Open webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize count variable
total_count = 0

# Specify the class you want to count (e.g., 'person')
target_class = 'person'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Get class names and detections
    detections = results[0].boxes
    names = model.names  # Dictionary of class index to name

    count = 0  # count for this frame

    # Loop through detections
    for box in detections:
        class_id = int(box.cls[0])
        class_name = names[class_id]

        if class_name == target_class:
            count += 1

    total_count += count  # update total count

    # Display count on frame
    cv2.putText(frame, f"{target_class.capitalize()} Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Plot the results
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save total count to file
with open("count_result.txt", "w") as f:
    f.write(f"Total {target_class}s detected: {total_count}")