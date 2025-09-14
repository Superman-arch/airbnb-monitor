#!/usr/bin/env python3
"""Native Python VLM server using HuggingFace transformers for Jetson."""

import os
import json
import base64
import torch
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variables
model = None
processor = None

def load_model():
    """Load Phi-3.5 Vision model with compatibility fixes."""
    global model, processor
    
    logger.info("Loading Phi-3.5 Vision model...")
    logger.info(f"PyTorch version: {torch.__version__}")
    model_id = "microsoft/Phi-3.5-vision-instruct"
    
    try:
        # Fix float8 compatibility for older PyTorch versions
        if not hasattr(torch, 'float8_e4m3fn'):
            logger.info("Patching PyTorch for float8 compatibility...")
            # Create dummy float8 types that map to float16
            torch.float8_e4m3fn = torch.float16
            torch.float8_e5m2 = torch.float16
            torch.float8_e4m3fnuz = torch.float16
            torch.float8_e5m2fnuz = torch.float16
            
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4  # Optimized for multi-frame
        )
        
        # Load model with compatibility settings
        logger.info("Loading model weights (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for better compatibility
            low_cpu_mem_usage=True,
            _attn_implementation='eager',  # Don't require flash_attn
            ignore_mismatched_sizes=True  # Ignore size mismatches from float8
        )
        
        # Optional: Convert to half precision after loading for memory efficiency
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6:
            logger.info("Converting model to half precision for GPU efficiency...")
            model = model.half()
            
        logger.info(f"Model loaded successfully on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Trying fallback loading method...")
        
        # Fallback: Try loading without some optimizations
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
            logger.info("Model loaded with fallback method")
            return True
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model is not None else "loading",
        "model": "Phi-3.5-vision-instruct",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models (OpenAI compatible)."""
    return jsonify({
        "data": [{
            "id": "Phi-3.5-vision-instruct",
            "object": "model",
            "owned_by": "microsoft"
        }]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Extract image and text from the last message
        last_message = messages[-1]
        content = last_message.get('content', [])
        
        image = None
        text_prompt = ""
        
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'image_url':
                    # Decode base64 image
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url.startswith('data:image'):
                        # Extract base64 data
                        base64_str = image_url.split(',')[1]
                        image_data = base64.b64decode(base64_str)
                        image = Image.open(BytesIO(image_data))
                elif item.get('type') == 'text':
                    text_prompt = item.get('text', '')
            else:
                text_prompt = str(item)
        
        if image is None:
            return jsonify({"error": "No image provided"}), 400
        
        # Format prompt for Phi-3.5 Vision
        formatted_messages = [
            {"role": "user", "content": text_prompt}
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process with model
        inputs = processor(prompt, [image], return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # Remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response_text = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Return OpenAI-compatible response
        return jsonify({
            "id": "chatcmpl-vlm",
            "object": "chat.completion",
            "model": "Phi-3.5-vision-instruct",
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
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting VLM server on port 8080...")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)