"""
Main entry point for the LLM API server
This file is used by Replit for deployment
"""

if __name__ == "__main__":
    import uvicorn
    from api import app
    
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)