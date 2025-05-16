from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from concurrent.futures import ProcessPoolExecutor
import subprocess
import uuid
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
basePath = "/workspaces/wave2lip/Wav2Lip-GFPGAN"
wav2lipPath = basePath + "/Wav2Lip-master"
inputPath = basePath + "/inputs"
outputPath = basePath + "/outputs"

os.makedirs(inputPath, exist_ok=True)
os.makedirs(outputPath, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'wav'}

app = Flask(__name__)
executor = ProcessPoolExecutor(max_workers=1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_inference(video_path, audio_path, output_path):
    cmd = [
        'python', 'inference.py',
        '--checkpoint_path', 'checkpoints/wav2lip.pth',
        '--face', video_path,
        '--audio', audio_path,
        '--outfile', output_path
    ]
    subprocess.run(cmd, cwd=wav2lipPath, check=True)

@app.route('/process', methods=['POST'])
def process_files():
    logging.info("Request received")

    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({"error": "Both video and audio files are required"}), 400

    video = request.files['video']
    audio = request.files['audio']

    if not allowed_file(video.filename) or not allowed_file(audio.filename):
        return jsonify({"error": "Invalid file type"}), 400

    req_id = str(uuid.uuid4())
    video_path = os.path.join(inputPath, f"video_{req_id}.mp4")
    audio_path = os.path.join(inputPath, f"audio_{req_id}.wav")
    result_path = os.path.join(outputPath, f"result_{req_id}.mp4")

    video.save(video_path)
    audio.save(audio_path)

    future = executor.submit(run_inference, video_path, audio_path, result_path)

    try:
        future.result()
        if not os.path.exists(result_path):
            return jsonify({"error": "Output not generated"}), 500
        return send_file(result_path, as_attachment=True)
    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
