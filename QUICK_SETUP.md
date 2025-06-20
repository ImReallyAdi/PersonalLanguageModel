# Quick Setup Guide

GitHub Pages can only host static files (HTML/CSS/JS), not Python servers. You need to deploy the backend separately.

## Option 1: Vercel Deployment (Recommended)

### Deploy Backend to Vercel

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Deploy from your project directory**:
   ```bash
   vercel --prod
   ```

3. **Your API will be available at**: `https://your-project-name.vercel.app`

4. **Test the deployment**:
   - Visit: `https://your-project-name.vercel.app/docs`
   - Test chat: `https://your-project-name.vercel.app/api/chat/hello`

### Vercel Benefits:
- Serverless functions (cost-effective)
- Auto-scaling
- Built-in HTTPS
- Fast global CDN
- No sleep mode (unlike free Replit)

## Option 2: Replit Deployment (Alternative)

1. **Create Replit Account**: Go to [replit.com](https://replit.com)

2. **Import Your Repository**:
   - Click "Create Repl"
   - Select "Import from GitHub" 
   - Enter: `https://github.com/imreallyadi/PersonalLanguageModel`

3. **Run the API**:
   - Click "Run" button
   - Your API URL: `https://PersonalLanguageModel.imreallyadi.repl.co`

## Frontend Setup (Already Working)

Your frontend is deployed at: `https://imreallyadi.github.io/PersonalLanguageModel/`

## Connect Frontend to Backend

1. **Visit your GitHub Pages site**
2. **Find the "API URL" field at the bottom**
3. **Enter your backend URL**:
   - Vercel: `https://your-project-name.vercel.app`
   - Replit: `https://PersonalLanguageModel.imreallyadi.repl.co`
4. **Click "Test Connection"**
5. **Start chatting!**

## URLs Summary

- **Frontend**: https://imreallyadi.github.io/PersonalLanguageModel/
- **Backend (Vercel)**: https://your-project-name.vercel.app
- **API Docs**: https://your-project-name.vercel.app/docs
- **Chat Endpoint**: https://your-project-name.vercel.app/api/chat/hello

## Files for Vercel

Your project includes:
- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless function entry point
- Lightweight model that trains quickly on startup

## Troubleshooting

**Vercel Deployment Issues**:
- Ensure PyTorch is compatible with Vercel's Python runtime
- Check function timeout limits (max 10 seconds for free tier)
- Monitor function logs in Vercel dashboard

**Connection Errors**:
- Wait 30 seconds after deployment for model to initialize
- Check CORS settings if frontend can't connect
- Verify API URL format in frontend

Your system architecture:
`GitHub Pages Frontend` → `Vercel Serverless API` → `AI Model Responses`