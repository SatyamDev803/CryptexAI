import os
import argparse
import uvicorn
from src.api import app

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bitcoin Price Prediction Backend")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", 8000)),
        help="Port to run the server on"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Ensure required directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )