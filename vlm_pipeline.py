#!/usr/bin/env python3
"""VLM server using HuggingFace pipeline API - handles config automatically."""

import os
import json
import base64
import torch
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from transformers import pipeline
import logging
import warnings
warnings.filterwarnings("ignore")

# Disable Flash Attention
os.environ["TRANSFORMERS_USE_FLASH_ATTENTION"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global pipeline
vlm_pipeline = None

def load_pipeline():
    """Load VLM using pipeline API."""
    global vlm_pipeline
    
    logger.info("Loading VLM using HuggingFace pipeline...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Determine device
        if torch.cuda.is_available():
            device = 0  # Use first GPU
            logger.info("Using GPU")
        else:
            device = -1  # Use CPU
            logger.info("Using CPU (will be slow)")
        
        # Try Phi-3.5 Vision first
        try:
            logger.info("Loading Phi-3.5 Vision...")
            vlm_pipeline = pipeline(
                "image-text-to-text",
                model="microsoft/Phi-3.5-vision-instruct",
                trust_remote_code=True,
                device=device,
                torch_dtype="auto"
            )
            logger.info("✓ Phi-3.5 Vision loaded successfully!")
            return True
            
        except Exception as e:
            logger.warning(f"Phi-3.5 Vision failed: {e}")
            logger.info("Falling back to smaller model...")
            
            # Fallback to BLIP for door detection
            vlm_pipeline = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=device
            )
            logger.info("✓ BLIP model loaded as fallback")
            return True
            
    except Exception as e:
        logger.error(f"Failed to load any model: {e}")
        return False

def process_with_pipeline(image, text_prompt):
    """Process image with the pipeline."""
    global vlm_pipeline
    
    if vlm_pipeline is None:
        return None
    
    try:
        # Check which type of pipeline we have
        if vlm_pipeline.task == "image-text-to-text":
            # Phi-3.5 Vision style
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ]
            }]
            result = vlm_pipeline(messages, images=[image])
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    return result[0].get('generated_text', str(result))
                return str(result[0])
            return str(result)
            
        else:
            # BLIP style - simple image captioning
            # For door detection, we'll analyze the caption
            caption = vlm_pipeline(image)
            if isinstance(caption, list) and len(caption) > 0:
                caption_text = caption[0].get('generated_text', str(caption))
                
                # Analyze caption for doors
                door_keywords = ['door', 'entrance', 'doorway', 'entry', 'exit', 'portal']
                has_door = any(keyword in caption_text.lower() for keyword in door_keywords)
                
                if has_door:
                    return f"Door detected in image. Caption: {caption_text}"
                else:
                    return f"No door detected. Scene: {caption_text}"
            return str(caption)
            
    except Exception as e:
        logger.error(f"Pipeline processing error: {e}")
        return f"Error: {str(e)}"

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if vlm_pipeline else "loading",
        "model": "pipeline",
        "cuda": torch.cuda.is_available()
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible endpoint using pipeline."""
    if vlm_pipeline is None:
        return jsonify({"error": "Pipeline not loaded"}), 503
    
    try:
        data = request.json
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Extract image and text
        last_message = messages[-1]
        content = last_message.get('content', [])
        
        image = None
        text_prompt = ""
        
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url.startswith('data:image'):
                        base64_str = image_url.split(',')[1]
                        image_data = base64.b64decode(base64_str)
                        image = Image.open(BytesIO(image_data))
                elif item.get('type') == 'text':
                    text_prompt = item.get('text', '')
        
        if image is None:
            return jsonify({"error": "No image provided"}), 400
        
        # Process with pipeline
        logger.info("Processing image with pipeline...")
        response_text = process_with_pipeline(image, text_prompt)
        
        if response_text is None:
            return jsonify({"error": "Processing failed"}), 500
        
        return jsonify({
            "id": "chatcmpl-pipeline",
            "object": "chat.completion",
            "model": "pipeline",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint with sample image."""
    try:
        # Test with a simple colored square
        import numpy as np
        test_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        
        result = process_with_pipeline(test_image, "What do you see?")
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if load_pipeline():
        logger.info("Starting VLM pipeline server on port 8080...")
        logger.info("Using HuggingFace pipeline API")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Failed to load pipeline")
        exit(1)