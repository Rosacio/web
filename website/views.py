
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
