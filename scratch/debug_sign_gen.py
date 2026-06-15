import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from app.services.sign_generator import SignGenerator
    from app.config import load_config
    
    config = load_config()
    generator = SignGenerator(
        endpoint=config.sign_model_endpoint,
        sign_dir=config.sign_dir,
        timeout_seconds=config.sign_model_timeout_seconds,
    )
    print("SignGenerator initialized successfully")
    
    result = generator.generate("hello")
    print(f"Result backend: {result.backend}")
    print(f"Result video_url: {result.video_url}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
