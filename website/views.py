from flask import Flask, Blueprint, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import os

app = Flask(__name__)

views = Blueprint('views', __name__)

# Load the HDF5 model
model_path = '/home/jmrosado007/resnet2_model.h5'
print("Loading model...")
loaded_model = load_model(model_path)
print("Model loaded.")

# Use the loaded model to create a feature extractor
feature_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-2].output)
print("Feature extractor created.")

# Load the precomputed features and filenames from the HDF5 file
def load_features_and_labels(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        features = np.array(f['features'])
        filenames = [name.decode('utf-8') for name in np.array(f['filenames']>
        labels = [label.decode('utf-8') for label in np.array(f['labels'])]  >
    return features, filenames, labels

features, filenames, labels = load_features_and_labels('/home/jmrosado007/Fea>
print("Features and filenames loaded.")

# Ensure the upload directory exists
upload_dir = '/home/jmrosado007/website/static/uploads'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    print("Upload directory created.")

# Serve the dataset images
@app.route('/dataset/<path:filename>')
def serve_dataset_images(filename):
    dataset_dir = '/home/jmrosado007/Dataset'
    return send_from_directory(dataset_dir, filename)

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
        predicted_category, similar_images = find_similar_images(temp_path, f>
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
    from tensorflow.keras.preprocessing import image  # Ensure this import is>
    img = image.load_img(img_path, target_size=(224, 224))  # Ensure target s>
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the im>
    features = feature_model.predict(img_array)
    return features.flatten()
def find_similar_images(uploaded_image_path, features, filenames, labels, top>
    # Extract features from the uploaded image
    uploaded_image_features = extract_features(uploaded_image_path)
    print("Uploaded image features shape:", uploaded_image_features.shape)  #>

    # Ensure that the uploaded_image_features array is not empty
    if uploaded_image_features.shape[0] == 0:
        return "Unknown", ["No similar images found. Please try again with a >

    # Compute similarity
    similarities = cosine_similarity([uploaded_image_features], features)

    # Get the most similar image to determine the label
    most_similar_index = np.argmax(similarities[0])
    predicted_label = labels[most_similar_index]

    # Filter features, filenames, and labels by the predicted label
    label_indices = [i for i, label in enumerate(labels) if label == predicte>
    filtered_features = features[label_indices]
    filtered_filenames = [filenames[i] for i in label_indices]

    # Compute similarity again within the filtered set
    filtered_similarities = cosine_similarity([uploaded_image_features], filt>

    # Get the top N most similar images within the same label
    similar_indices = np.argsort(filtered_similarities[0])[::-1][:top_n]
    top_similar_images = [filtered_filenames[i] for i in similar_indices]

    # Replace the dataset path with the static path
    static_base_path = '/static/images'
    top_similar_images_static = []
    
    for image in top_similar_images:
        # Decode bytes to string if necessary
        if isinstance(image, bytes):
            image = image.decode('utf-8')

        # Ensure consistent use of forward slashes
        image = image.replace("\\", "/")

        # Remove the leading part of the path to leave only the category and >
        if 'Dataset/' in image:
            image = image.split('Dataset/')[1]

        static_image_path = f'{static_base_path}/{image}'
        top_similar_images_static.append(static_image_path)
        print(f"Converted {image} to {static_image_path}")

    return predicted_label, top_similar_images_static
@views.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(upload_dir, filename)

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File is too large. The maximum file is 16MB"}),>

@views.route('/development')
def development():
        return render_template('development.html')


app.register_blueprint(views)

if __name__ == "__main__":
    app.run(debug=True)
