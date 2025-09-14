#!/usr/bin/env python3
"""CPU-only VLM server - works immediately without CUDA setup."""

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

# Force CPU mode and disable Flash Attention
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU
os.environ["TRANSFORMERS_USE_FLASH_ATTENTION"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None
processor = None

def load_model_cpu():
    """Load Phi-3.5 Vision in CPU-only mode."""
    global model, processor
    
    logger.info("Loading Phi-3.5 Vision (CPU mode - no CUDA required)...")
    logger.info("This will be slow but works immediately!")
    
    model_id = "microsoft/Phi-3.5-vision-instruct"
    
    try:
        # Fix compatibility
        if not hasattr(torch, 'float8_e4m3fn'):
            torch.float8_e4m3fn = torch.float16
            torch.float8_e5m2 = torch.float16
            torch.float8_e4m3fnuz = torch.float16
            torch.float8_e5m2fnuz = torch.float16
        
        # Load processor
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=1  # Minimal for CPU
        )
        
        # Load model - CPU only, no Flash Attention issues
        logger.info("Loading model on CPU (this will take 2-3 minutes)...")
        
        # Try with transformers config override
        from transformers import Phi3VConfig
        config = Phi3VConfig.from_pretrained(model_id, trust_remote_code=True)
        config._flash_attn_2_enabled = False
        config.use_flash_attention_2 = False
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        logger.info("✓ Model loaded on CPU successfully!")
        logger.info("Note: Inference will be slow (~30 seconds per image)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if model else "loading",
        "model": "Phi-3.5-vision-instruct",
        "mode": "cpu-only"
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        messages = data.get('messages', [])
        
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
                        # Resize for CPU processing
                        image.thumbnail((320, 320), Image.Resampling.LANCZOS)
                elif item.get('type') == 'text':
                    text_prompt = item.get('text', '')
        
        if image is None:
            return jsonify({"error": "No image provided"}), 400
        
        # Process
        logger.info("Processing image (this will take ~30 seconds on CPU)...")
        
        formatted_messages = [{"role": "user", "content": text_prompt}]
        prompt = processor.tokenizer.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True
        )
        
        with torch.no_grad():
            inputs = processor(prompt, [image], return_tensors="pt")
            
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=100,  # Minimal for CPU
                temperature=0.1,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        response_text = processor.batch_decode(
            generate_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        gc.collect()
        
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
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if load_model_cpu():
        logger.info("Starting CPU-only VLM server on port 8080...")
        logger.info("This works immediately but is slow!")
        logger.info("For GPU acceleration, run: ./install_pytorch_cuda.sh")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Failed to load model")
        exit(1)