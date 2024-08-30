import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from io import BytesIO

# Load YOLOv8 model
model = YOLO('best.pt')

def process_image(image):
    # Run inference with YOLOv8
    results = model(image)
    
    # Extract predictions
    predictions = results.pandas().xyxy[0]  # Get the results as a pandas DataFrame

    # Initialize the mask with zeros
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Assuming 'floor' class is labeled as 0; adjust if necessary
    floor_class_id = 0

    # Create a mask from predictions
    for _, row in predictions.iterrows():
        class_id = int(row['class'])
        if class_id == floor_class_id:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            mask[y1:y2, x1:x2] = 255

    # Convert mask to binary image
    _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Define the regions for left, center, and right
    left_region = thresh[:, :width // 3]
    center_region = thresh[:, width // 3: 2 * width // 3]
    right_region = thresh[:, 2 * width // 3:]

    # Calculate the percentage of the floor in each region
    left_floor = np.sum(left_region == 255) / left_region.size * 100
    center_floor = np.sum(center_region == 255) / center_region.size * 100
    right_floor = np.sum(right_region == 255) / right_region.size * 100

    # Determine the directions
    directions = []
    if left_floor > 10:  # Adjust threshold as needed
        directions.append("left")
    if center_floor > 10:
        directions.append("straight")
    if right_floor > 10:
        directions.append("right")

    return thresh, directions

# Streamlit App
st.title('YOLOv8 Object Detection with Streamlit')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Read and display image
    image_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    # Process image and display results
    thresh, directions = process_image(image)
    
    # Display directions
    if directions:
        st.write("The rover can move in the following directions:", ", ".join(directions))
    else:
        st.write("No clear path available.")
    
    # Display thresholded image
    st.write("Thresholded Image:")
    fig, ax = plt.subplots()
    ax.imshow(thresh, cmap='gray')
    ax.set_title('Thresholded Image')
    st.pyplot(fig)
