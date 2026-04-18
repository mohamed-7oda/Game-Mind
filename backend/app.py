from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid

from tracking_pipeline import run_tracking
from test_pipeline import run_event_detection

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/process", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    mode = request.form.get("mode", "")

    unique_name = str(uuid.uuid4()) + "_" + file.filename
    video_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(video_path)

    if mode == "tracking":
        # SAVE AS MP4 (VERY IMPORTANT)
        base_name = os.path.splitext(unique_name)[0]
        output_name = f"output_{base_name}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        final_path = run_tracking(video_path, output_path)

        return send_file(final_path, mimetype="video/mp4")

    elif mode == "event":
        result = run_event_detection(video_path)
        return jsonify(result)

    return jsonify({"error": "Invalid mode"}), 400


if __name__ == "__main__":
    app.run(debug=True)