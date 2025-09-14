#!/usr/bin/env python3
"""Optimized VLM server for Jetson Nano Super (8GB RAM)."""

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

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variables
model = None
processor = None

def print_memory_stats():
    """Print current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    import psutil
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    logger.info(f"RAM Usage: {ram_usage:.2f}GB")

def load_model_optimized():
    """Load Phi-3.5 Vision with memory optimizations for 8GB Jetson."""
    global model, processor
    
    logger.info("Loading Phi-3.5 Vision (Optimized for 8GB Jetson Nano Super)...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    model_id = "microsoft/Phi-3.5-vision-instruct"
    
    try:
        # Clear any existing cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Fix float8 compatibility
        if not hasattr(torch, 'float8_e4m3fn'):
            logger.info("Patching PyTorch for float8 compatibility...")
            torch.float8_e4m3fn = torch.float16
            torch.float8_e5m2 = torch.float16
            torch.float8_e4m3fnuz = torch.float16
            torch.float8_e5m2fnuz = torch.float16
        
        # Load processor with minimal settings
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=1  # Reduce from 4 to 1 for memory
        )
        
        # Try 8-bit quantization first (requires bitsandbytes)
        try:
            logger.info("Attempting 8-bit quantized loading...")
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                max_memory={0: "6GB", "cpu": "2GB"},
                offload_folder="offload"
            )
            logger.info("✓ Model loaded with 8-bit quantization")
            
        except ImportError:
            logger.warning("bitsandbytes not available, trying without quantization...")
            
            # Fallback: Load with memory mapping and offloading
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use fp16 to save memory
                low_cpu_mem_usage=True,
                max_memory={0: "5GB", "cpu": "3GB"},  # Split between GPU and CPU
                offload_folder="offload",
                offload_state_dict=True,
                _attn_implementation='eager'
            )
            logger.info("✓ Model loaded with CPU offloading")
        
        # Clear cache after loading
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        print_memory_stats()
        logger.info("Model ready for inference!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        
        # Last resort: minimal model loading
        try:
            logger.info("Trying minimal loading configuration...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": "cpu"},  # Load entirely on CPU
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move only essential layers to GPU if available
            if torch.cuda.is_available():
                model.vision_embed_tokens = model.vision_embed_tokens.cuda()
            
            logger.info("✓ Model loaded in CPU mode")
            return True
            
        except Exception as e2:
            logger.error(f"All loading methods failed: {e2}")
            return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model is not None else "loading",
        "model": "Phi-3.5-vision-instruct",
        "mode": "optimized_8gb"
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Clear cache before processing
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
                        
                        # Resize image to save memory
                        max_size = (640, 640)
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
        
        # Process with model (minimal settings)
        with torch.no_grad():
            inputs = processor(prompt, [image], return_tensors="pt")
            
            # Move to appropriate device
            if torch.cuda.is_available() and hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with minimal settings
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=200,  # Reduced from 500
                temperature=0.1,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
            # Clear intermediate tensors
            del inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Decode response
        generate_ids = generate_ids[:, -200:]  # Only decode generated tokens
        response_text = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clear cache after processing
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
        logger.error(f"Error processing request: {e}")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Install bitsandbytes if not present
    try:
        import bitsandbytes
    except ImportError:
        logger.info("Installing bitsandbytes for 8-bit quantization...")
        os.system("pip3 install bitsandbytes")
    
    # Load model
    if load_model_optimized():
        logger.info("Starting optimized VLM server on port 8080...")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Failed to load model. Please run jetson_optimize.sh first")
        exit(1)