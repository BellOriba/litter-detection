import os
import cv2
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib as plt

# Load TACO dataset annotations (path to annotations.json)
taco_annotations_path = 'datasets/TACO/data/annotations.json'
with open(taco_annotations_path, 'r') as f:
    annotations = json.load(f)

# Path to the folder containing the 15 subdirectories
base_image_dir = 'datasets/TACO/data/'

# Extract images and bounding boxes
images = []
labels = []
bboxes = []

# Define class names (you can modify this according to your needs)
class_names = ['plastic', 'paper', 'metal', 'glass', 'organic']

# Preprocess images and annotations
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    image_info = annotations['images'][image_id]

    # Find the folder where the image is located
    image_filename = image_info['file_name']  # The image file name includes the folder name
    image_path = os.path.join(base_image_dir, image_filename)  # Combine folder and image file name

    # Debugging step: print the image path
    print(f"Loading image from: {image_path}")
    
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Failed to load image at {image_path}")
        continue  # Skip this image and continue with the next one
    
    # Convert segmentation masks to bounding boxes
    segmentation = np.array(annotation['segmentation'][0])
    x_min = np.min(segmentation[0::2])
    y_min = np.min(segmentation[1::2])
    x_max = np.max(segmentation[0::2])
    y_max = np.max(segmentation[1::2])
    
    bbox = [x_min, y_min, x_max, y_max]
    
    # Normalize bounding box
    h, w, _ = image.shape
    bbox_normalized = [x_min / w, y_min / h, x_max / w, y_max / h]
    
    # Append data for training
    images.append(cv2.resize(image, (128, 128)))
    labels.append(annotation['category_id'])  # The category ID maps to class_names
    bboxes.append(bbox_normalized)

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)
bboxes = np.array(bboxes)

# Split into train and test sets
X_train, X_test, y_train, y_test, bbox_train, bbox_test = train_test_split(
    images, labels, bboxes, test_size=0.2, random_state=42
)

# Define CNN model
def create_model():
    input_image = layers.Input(shape=(128, 128, 3))
    
    # CNN layers for feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_image)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten the output for Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    
    # Output 1: Object Class prediction
    output_class = layers.Dense(len(class_names), activation='softmax', name='class_output')(x)
    
    # Output 2: Bounding box regression (x_min, y_min, x_max, y_max)
    output_bbox = layers.Dense(4, activation='sigmoid', name='bbox_output')(x)
    
    # Model definition with two outputs
    model = models.Model(inputs=input_image, outputs=[output_class, output_bbox])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'bbox_output': 'mean_squared_error'
        },
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mean_squared_error'
        }
    )
    
    return model

# Create the model
model = create_model()

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, {'class_output': y_train, 'bbox_output': bbox_train},
    epochs=10,
    batch_size=32,
    validation_data=(X_test, {'class_output': y_test, 'bbox_output': bbox_test})
)

# Evaluate the model on test set
test_loss, class_loss, bbox_loss, class_acc, bbox_mse = model.evaluate(
    X_test, {'class_output': y_test, 'bbox_output': bbox_test}
)

print(f"Test Loss: {test_loss}")
print(f"Classification Accuracy: {class_acc}")
print(f"Bounding Box MSE: {bbox_mse}")

# Predict on new image
new_image = cv2.imread('new_image_path.jpg')
new_image_resized = cv2.resize(new_image, (128, 128))

# Predict class and bounding box
pred_class, pred_bbox = model.predict(np.expand_dims(new_image_resized, axis=0))

# Get predicted class label
predicted_label = class_names[np.argmax(pred_class)]

# Bounding box coordinates
x_min, y_min, x_max, y_max = pred_bbox[0]

# Visualize the result
cv2.rectangle(new_image_resized, (int(x_min * 128), int(y_min * 128)), (int(x_max * 128), int(y_max * 128)), (0, 255, 0), 2)
cv2.putText(new_image_resized, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display result
plt.imshow(cv2.cvtColor(new_image_resized, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()