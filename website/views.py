from flask import Blueprint, render_template, request, jsonify, send_from_directory, redirect
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import os
from google.cloud import storage

views = Blueprint('views', __name__)

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket_name = 'webproject12'
bucket = storage_client.bucket(bucket_name)

# Helper function to download files from GCS
def download_blob(bucket_name, source_blob_name, destination_file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    print(f"Downloading {source_blob_name} to {destination_file_name}...")
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")
    # Verify file size after download
    if os.path.exists(destination_file_name):
        print(f"File size of {destination_file_name}: {os.path.getsize(destination_file_name)} bytes")

# Download model and features files from GCS
model_path = '/tmp/resnet2_model.h5'
features_path = '/tmp/features_data.h5'

try:
    print("Downloading model and features files...")
    download_blob(bucket_name, 'resnet2_model.h5', model_path)
    download_blob(bucket_name, 'features_data.h5', features_path)
    print("Model and features files downloaded.")
except Exception as e:
    print(f"Error downloading files: {e}")

# Load the HDF5 model
try:
    print("Loading model...")
    loaded_model = load_model(model_path)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")

# Use the loaded model to create a feature extractor
try:
    feature_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-2].output)
    print("Feature extractor created.")
except Exception as e:
    print(f"Error creating feature extractor: {e}")

# Load the precomputed features and filenames from the HDF5 file
def load_features_and_labels(hdf5_file):
    try:
        with h5py.File(hdf5_file, 'r') as f:
            features = np.array(f['features'])
            filenames = [name.decode('utf-8') for name in np.array(f['filenames'])]

            # Check if 'labels' exists in the file
            if 'labels' in f:
                labels = [label.decode('utf-8') for label in np.array(f['labels'])]
            else:
                labels = ['Unknown'] * len(filenames)  # or handle appropriately

            print(f"Loaded features and labels from {hdf5_file}.")
            return features, filenames, labels
    except Exception as e:
        print(f"Error loading features and labels from {hdf5_file}: {e}")
        return None, None, None

features, filenames, labels = load_features_and_labels(features_path)

# Ensure the upload directory exists
upload_dir = '/tmp/uploads'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    print("Upload directory created.")

@views.route('/')
def home():
    return render_template('index.html')

@views.route('/upload', methods=['POST'])
def upload():
    print("Upload route called.")
    file = request.files['image']
    if file:
        print("File received: ", file.filename)
        # Save the uploaded file to a temporary location
        temp_path = os.path.join(upload_dir, file.filename)
        file.save(temp_path)
        print("File saved to: ", temp_path)

        # Find similar images
        predicted_category, similar_images = find_similar_images(temp_path, features, filenames, labels)
        print("Similar images found: ", similar_images)

        # Clean up the temporary file
        os.remove(temp_path)

        return jsonify({
            'predicted_category': predicted_category,
            'similar_images': similar_images
        })
    print("No image uploaded.")
    return jsonify({"error": "No image uploaded"}), 400

def extract_features(img_path):
    from tensorflow.keras.preprocessing import image  # Ensure this import is correct
    img = image.load_img(img_path, target_size=(224, 224))  # Ensure target size matches model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    features = feature_model.predict(img_array)
    return features.flatten()

def find_similar_images(uploaded_image_path, features, filenames, labels, top_n=5):
    # Extract features from the uploaded image
    uploaded_image_features = extract_features(uploaded_image_path)
    print("Uploaded image features shape:", uploaded_image_features.shape)  # Debugging line

    # Ensure that the uploaded_image_features array is not empty
    if uploaded_image_features.shape[0] == 0:
        return "Unknown", ["No similar images found. Please try again with a different image."]

    # Compute similarity
    similarities = cosine_similarity([uploaded_image_features], features)

    # Get the top N most similar images
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]
    top_similar_images = [filenames[i] for i in similar_indices]
    predicted_category = labels[similar_indices[0]]

    # Replace the dataset path with the static path
    static_base_path = 'https://storage.googleapis.com/webproject12/static/images'
    top_similar_images_static = []

    for image in top_similar_images:
        # Decode bytes to string if necessary
        if isinstance(image, bytes):
            image = image.decode('utf-8')

        # Ensure consistent use of forward slashes
        image = image.replace("\\", "/")

        # Remove the leading part of the path to leave only the category and filename
        if 'Dataset/' in image:
            image = image.split('Dataset/')[1]

        static_image_path = f'{static_base_path}/{image}'
        top_similar_images_static.append(static_image_path)
        print(f"Converted {image} to {static_image_path}")

    return predicted_category, top_similar_images_static

@views.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(upload_dir, filename)
