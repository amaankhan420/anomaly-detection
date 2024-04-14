import cv2
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from predictor import plot_heat_map_for_image

app = Flask(__name__)
CORS(app)


@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the file name from the request
    file_name = request.json['file_name']

    # Define base directory for images
    base_directory = '/path/to/your/images/directory'

    # Check if file name contains "good" or "anomaly"
    if "good" in file_name:
        base_directory = 'planks/good_data'
    elif "anomaly" in file_name:
        base_directory = 'planks/anomaly_data'
    else:
        return jsonify({'message': 'Invalid file name'})

    # Construct the full path to the image file
    image_path = os.path.join(base_directory, file_name)

    img = cv2.imread(image_path)

    plot_heat_map_for_image(img)

    return '', 200


if __name__ == '__main__':
    app.run(debug=True)
