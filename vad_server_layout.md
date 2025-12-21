#!/usr/bin/env python3
import os
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from flask import Flask, request, jsonify
import base64
import io
import librosa
import soundfile as sf

# Configuration
MODEL_PATH = os.path.expanduser("~/models/smart-turn-v3/smart-turn-v3.1-gpu.onnx")
PORT = 7863

app = Flask(__name__)

def build_session(onnx_path):
    """Build ONNX runtime session with optimized settings."""
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Use CUDA execution provider if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
    return audio_array

# Initialize model components
print(f"Loading model from: {MODEL_PATH}")
feature_extractor = WhisperFeatureExtractor(chunk_length=8)
session = build_session(MODEL_PATH)
print("Model loaded successfully")

def predict_endpoint(audio_array):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.
    
    Args:
        audio_array: Numpy array containing audio samples at 16kHz
        
    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """
    # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)
    
    # Process audio using Whisper's feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )
    
    # Extract features and ensure correct shape for ONNX
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension
    
    # Run ONNX inference
    outputs = session.run(None, {"input_features": input_features})
    
    # Extract probability (ONNX model returns sigmoid probabilities)
    probability = outputs[0][0].item()
    
    # Make prediction (1 for Complete, 0 for Incomplete)
    prediction = 1 if probability > 0.5 else 0
    
    return {
        "prediction": prediction,
        "probability": probability,
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": "smart-turn-v3.1-gpu"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that accepts audio data.
    
    Accepts JSON with either:
    - "audio_base64": Base64 encoded audio data
    - "audio_url": URL to audio file (not implemented in this basic version)
    
    Returns:
    - prediction: 1 for complete turn, 0 for incomplete
    - probability: Confidence score (0-1)
    - status: "complete" or "incomplete"
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Handle base64 encoded audio
        if "audio_base64" in data:
            audio_data = base64.b64decode(data["audio_base64"])
            
            # Load audio from bytes
            audio, sr = sf.read(io.BytesIO(audio_data))
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Convert to float32
            audio = audio.astype(np.float32)
            
            # Normalize to [-1, 1] range
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Make prediction
            result = predict_endpoint(audio)
            
            return jsonify({
                "prediction": result["prediction"],
                "probability": result["probability"],
                "status": "complete" if result["prediction"] == 1 else "incomplete"
            })
        
        else:
            return jsonify({"error": "No audio data provided. Use 'audio_base64' field."}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_raw', methods=['POST'])
def predict_raw():
    """
    Alternative endpoint that accepts raw audio bytes.
    
    Expects audio data as binary in the request body.
    Query parameters:
    - sample_rate: Sample rate of the audio (default: 16000)
    """
    try:
        # Get audio data from request body
        audio_data = request.data
        
        if not audio_data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Get sample rate from query params
        sample_rate = request.args.get('sample_rate', 16000, type=int)
        
        # Convert bytes to numpy array (assuming float32)
        audio = np.frombuffer(audio_data, dtype=np.float32)
        
        # Resample if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # Normalize
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Make prediction
        result = predict_endpoint(audio)
        
        return jsonify({
            "prediction": result["prediction"],
            "probability": result["probability"],
            "status": "complete" if result["prediction"] == 1 else "incomplete"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"Starting VAD server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)