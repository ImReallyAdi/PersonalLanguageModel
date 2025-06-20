# Deployment Guide

## GitHub Pages + Replit Deployment

This guide shows how to deploy your LLM system with the frontend on GitHub Pages and backend on Replit.

### Step 1: Deploy Backend to Replit

1. **Fork this repository** to your GitHub account

2. **Import to Replit**:
   - Go to [Replit](https://replit.com)
   - Click "Create Repl" > "Import from GitHub"
   - Enter your repository URL
   - Choose "Python" as the language

3. **Configure Replit**:
   - Replit will automatically detect the Python project
   - The main file should be set to `api.py`
   - Click "Run" - Replit will install dependencies and start the API server

4. **Get your API URL**:
   - After running, you'll see a URL like: `https://your-repl-name.your-username.repl.co`
   - This is your API endpoint

5. **Keep it running**:
   - Replit will keep your app running as long as it receives requests
   - For 24/7 uptime, consider upgrading to Replit Hacker plan

### Step 2: Deploy Frontend to GitHub Pages

1. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Click "Settings" tab
   - Scroll to "Pages" section
   - Under "Source", select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Click "Save"

2. **Wait for deployment**:
   - GitHub will build and deploy your site
   - This usually takes 1-2 minutes
   - You'll get a URL like: `https://yourusername.github.io/your-repo-name`

### Step 3: Connect Frontend to Backend

1. **Visit your GitHub Pages site**
2. **Configure API URL**:
   - In the API URL field at the bottom, enter your Replit URL
   - Example: `https://your-repl-name.your-username.repl.co`
   - Click "Test Connection"
   - You should see "Connected! Ready to chat."

### Step 4: Train Your Model

1. **Option A: Use the web interface**:
   - Visit your Replit URL directly
   - Go to `/docs` to see the API documentation
   - Use the training endpoint to train a model

2. **Option B: Train via API**:
   ```bash
   curl -X POST "https://your-repl-name.your-username.repl.co/model/train" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your training text here...",
       "sequence_length": 20,
       "batch_size": 8,
       "learning_rate": 0.01,
       "num_epochs": 3,
       "embed_dim": 64,
       "num_heads": 2,
       "num_layers": 2
     }'
   ```

3. **Check training status**:
   ```bash
   curl "https://your-repl-name.your-username.repl.co/model/training/status"
   ```

## Alternative Deployment Options

### Vercel + Railway

1. **Deploy API to Railway**:
   - Connect your GitHub repo to Railway
   - Railway will auto-deploy the Python backend
   - Get your Railway URL

2. **Deploy Frontend to Vercel**:
   - Connect your GitHub repo to Vercel
   - Vercel will auto-deploy the static files
   - Configure the API URL in the interface

### Netlify + Heroku

1. **Deploy API to Heroku**:
   - Create a `Procfile`: `web: uvicorn api:app --host=0.0.0.0 --port=$PORT`
   - Deploy to Heroku
   - Get your Heroku app URL

2. **Deploy Frontend to Netlify**:
   - Connect your GitHub repo to Netlify
   - Netlify will auto-deploy from main branch
   - Configure the API URL

## Environment Configuration

### For Production Deployment

Add these environment variables to your backend service:

```bash
# Database (if using external PostgreSQL)
DATABASE_URL=postgresql://user:pass@host:port/db

# API Configuration
CORS_ORIGINS=["https://yourusername.github.io"]
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration (optional)
DEFAULT_MODEL_PARAMS='{"embed_dim": 64, "num_heads": 2, "num_layers": 2}'
```

### CORS Configuration

Ensure your backend allows requests from your frontend domain. In `api.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourusername.github.io", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Add your GitHub Pages domain to CORS origins
   - Ensure HTTPS is used for both frontend and backend

2. **API Not Responding**:
   - Check if your Replit is sleeping (visit the URL to wake it)
   - Verify the API URL is correct
   - Check Replit logs for errors

3. **Training Fails**:
   - Ensure embed_dim is divisible by num_heads
   - Reduce batch_size if running out of memory
   - Check the training text is not empty

4. **GitHub Pages Not Updating**:
   - Check the Actions tab for deployment status
   - Clear your browser cache
   - Wait a few minutes for changes to propagate

### Performance Optimization

1. **Frontend**:
   - Enable browser caching for static assets
   - Minimize JavaScript bundle size
   - Use CDN for faster loading

2. **Backend**:
   - Implement response caching for frequently used endpoints
   - Use connection pooling for database
   - Add rate limiting to prevent abuse

## Security Considerations

1. **Never expose**:
   - Database credentials
   - API keys or secrets
   - Internal service URLs

2. **Use HTTPS**:
   - Always use HTTPS for production
   - Enable HSTS headers
   - Validate SSL certificates

3. **Rate Limiting**:
   - Implement rate limiting on training endpoints
   - Monitor for abuse patterns
   - Use authentication for sensitive operations

## Monitoring

### Health Checks

Add monitoring endpoints:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/metrics")
async def get_metrics():
    return {
        "models_trained": get_model_count(),
        "requests_today": get_request_count(),
        "uptime": get_uptime()
    }
```

### Logging

Configure structured logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

Your LLM system is now ready for production deployment!