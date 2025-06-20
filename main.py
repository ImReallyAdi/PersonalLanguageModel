"""
Main entry point for the LLM API server
This file is used by Replit for deployment
"""

if __name__ == "__main__":
    import uvicorn
    import os
    from api import app
    
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=port)