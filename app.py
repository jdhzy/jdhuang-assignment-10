from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from model import process_image_query, process_text_query, process_hybrid_query

app = Flask(__name__)

# Add a route to serve files from coco_images_resized
@app.route('/coco_images_resized/<path:filename>')
def serve_image(filename):
    # Replace with the absolute or relative path to the folder
    image_folder = os.path.abspath("coco_images_resized")
    return send_from_directory(image_folder, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query_type = request.form.get('query_type')
    text_query = request.form.get('text_query', "")
    lam = float(request.form.get('lam', 0.8))  # Default weight for hybrid
    image_file = request.files.get('image_query')

    try:
        if query_type == "Image query" and image_file:
            results = process_image_query(image_file)
        elif query_type == "Text query":
            results = process_text_query(text_query)
        elif query_type == "Hybrid query" and image_file:
            results = process_hybrid_query(text_query, image_file, lam)
        else:
            return jsonify({"error": "Invalid query or missing input."}), 400

        # Convert file paths to relative URLs
        for result in results:
            result["file_path"] = f"/{result['file_path']}"

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)