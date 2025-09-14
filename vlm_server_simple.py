#!/usr/bin/env python3
"""Simple VLM server without bitsandbytes - works with PyTorch 2.0.1."""

import os
import gc
import json
import base64
import torch
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoProcessor
import logging
import warnings
warnings.filterwarnings("ignore")

# Memory settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variables
model = None
processor = None

def load_model_simple():
    """Load Phi-3.5 Vision without bitsandbytes."""
    global model, processor
    
    logger.info("Loading Phi-3.5 Vision (Simple mode for Jetson)...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    model_id = "microsoft/Phi-3.5-vision-instruct"
    
    try:
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Fix float8 compatibility for PyTorch 2.0.1
        if not hasattr(torch, 'float8_e4m3fn'):
            logger.info("Adding float8 compatibility patch...")
            torch.float8_e4m3fn = torch.float16
            torch.float8_e5m2 = torch.float16
            torch.float8_e4m3fnuz = torch.float16
            torch.float8_e5m2fnuz = torch.float16
        
        # Fix GradScaler if missing
        if not hasattr(torch.amp, 'GradScaler'):
            logger.info("Adding GradScaler compatibility...")
            if torch.cuda.is_available():
                from torch.cuda.amp import GradScaler
                torch.amp.GradScaler = GradScaler
            else:
                # Dummy GradScaler for CPU
                class DummyGradScaler:
                    def __init__(self, *args, **kwargs):
                        pass
                    def scale(self, loss):
                        return loss
                    def step(self, optimizer):
                        optimizer.step()
                    def update(self):
                        pass
                torch.amp.GradScaler = DummyGradScaler
        
        # Load processor
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=1  # Minimize memory usage
        )
        
        # Load model with simple settings
        logger.info("Loading model (this will take a few minutes)...")
        
        if torch.cuda.is_available():
            # GPU loading with fp16
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use fp16 for memory efficiency
                low_cpu_mem_usage=True,
                max_memory={0: "6.5GB", "cpu": "1.5GB"},
                offload_folder="offload",
                _attn_implementation='eager'
            )
            logger.info("✓ Model loaded on GPU with fp16")
        else:
            # CPU-only fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": "cpu"},
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            logger.info("✓ Model loaded on CPU")
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Model ready for inference!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model is not None else "loading",
        "model": "Phi-3.5-vision-instruct",
        "mode": "simple",
        "cuda": torch.cuda.is_available()
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                        
                        # Resize to save memory
                        max_size = (480, 480)
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        
                elif item.get('type') == 'text':
                    text_prompt = item.get('text', '')
        
        if image is None:
            return jsonify({"error": "No image provided"}), 400
        
        # Format prompt
        formatted_messages = [
            {"role": "user", "content": text_prompt}
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process with model
        with torch.no_grad():
            inputs = processor(prompt, [image], return_tensors="pt")
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=150,  # Reduced for memory
                temperature=0.1,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
            # Clean up
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Decode
        generate_ids = generate_ids[:, -150:]
        response_text = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Check memory before starting
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_mem:.1f}GB")
    
    # Load model
    if load_model_simple():
        logger.info("Starting simple VLM server on port 8080...")
        logger.info("This version works with PyTorch 2.0.1")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Failed to load model")
        logger.error("Try running: ./fix_pytorch_deps.sh")
        exit(1)