from flask import Flask, redirect, request, jsonify
from google.cloud import storage
import os
import tensorflow as tf

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'

    # Initialize Google Cloud Storage client
    storage_client = storage.Client()
    bucket_name = 'your-bucket-name'
    bucket = storage_client.bucket(bucket_name)

    models_loaded = False

    @app.before_request
    def load_models():
        nonlocal models_loaded
        if not models_loaded:
            # Model files to download
            model_files = [
                ('features_data.h5', '/tmp/features_data.h5'),
                ('resnet2_model.h5', '/tmp/res_model.h5')
            ]

            for source_blob_name, destination_file_name in model_files:
                if not os.path.exists(destination_file_name):
                    blob = bucket.blob(source_blob_name)
                    blob.download_to_filename(destination_file_name)

            # Load models
            global features_data, inception_model, res_model
            features_data = tf.keras.models.load_model('/tmp/features_data.h5')
            resnet2_model = tf.keras.models.load_model('/tmp/resnet2_model.h5')

            models_loaded = True

    @app.route('/static/<path:filename>')
    def static_files(filename):
        return redirect(f"https://storage.googleapis.com/{bucket_name}/static/{filename}")

    @app.route('/identify', methods=['POST'])
    def identify():
        data = request.get_json()
        image_path = data['image_path']
        # Add your image identification logic here
        similar_images = []  # Replace with actual logic
        return jsonify(similar_images=similar_images)

    from .views import views
    from .auth import auth
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/auth/')

    return app
