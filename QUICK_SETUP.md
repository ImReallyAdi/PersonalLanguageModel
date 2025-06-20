# Quick Setup Guide

Your deployment issue: GitHub Pages can only host static files (HTML/CSS/JS), not Python servers.

## Solution: Two-Part Deployment

### Part 1: Deploy API Server to Replit

1. **Create Replit Account**: Go to [replit.com](https://replit.com)

2. **Import Your Repository**:
   - Click "Create Repl"
   - Select "Import from GitHub" 
   - Enter: `https://github.com/imreallyadi/PersonalLanguageModel`
   - Choose "Python" template

3. **Run the API**:
   - Replit will auto-detect `main.py` or `api.py`
   - Click the "Run" button
   - Wait for "Application startup complete"
   - Your API URL will be: `https://PersonalLanguageModel.imreallyadi.repl.co`

4. **Test API**:
   - Visit: `https://PersonalLanguageModel.imreallyadi.repl.co/docs`
   - You should see the FastAPI documentation

### Part 2: GitHub Pages Frontend (Already Working)

Your frontend is already deployed at: `https://imreallyadi.github.io/PersonalLanguageModel/`

## Connect Frontend to Backend

1. **Visit your GitHub Pages site**
2. **Scroll to bottom** - find the "API URL" field
3. **Enter your Replit URL**: `https://PersonalLanguageModel.imreallyadi.repl.co`
4. **Click "Test Connection"**
5. **Start chatting!**

## URLs Summary

- **Frontend (GitHub Pages)**: https://imreallyadi.github.io/PersonalLanguageModel/
- **Backend (Replit)**: https://PersonalLanguageModel.imreallyadi.repl.co
- **API Docs**: https://PersonalLanguageModel.imreallyadi.repl.co/docs
- **Chat Endpoint**: https://PersonalLanguageModel.imreallyadi.repl.co/api/chat/hello

## Troubleshooting

**"Connection Error"**: 
- Ensure your Replit is running (green status)
- Copy the exact Replit URL into the frontend
- Wait a moment for the model to train on first startup

**"Character mappings not available"**:
- This happens on fresh startup
- The system auto-trains a demo model
- Wait 2-3 minutes then refresh

**CORS Issues**:
- Your Replit URL should automatically allow GitHub Pages
- If issues persist, check Replit console logs

## Keep Replit Running

- Replit sleeps after inactivity
- Visit your API URL periodically to keep it awake
- Consider upgrading to Replit Hacker plan for 24/7 uptime

Your system will work as:
`GitHub Pages Frontend` → `Replit API Backend` → `AI Model Responses`