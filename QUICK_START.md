# 🚀 Quick Start Guide - MDA Regulatory Chatbot

## ✅ Project Status: READY TO RUN!

All errors have been fixed. The project is now fully functional.

## 🎯 Quick Setup (2 minutes)

### Option 1: Use the Setup Script (Recommended)
```bash
cd /Users/premalparagbhaishah/Documents/mda_project
./setup.sh
```
Follow the interactive prompts to configure your API and start the services.

### Option 2: Manual Setup

#### 1. Set up Llama API (Choose one):

**Groq (Recommended - Fast & Free):**
```bash
export LLAMA_API_KEY=<your-groq-api-key>
export LLAMA_BASE_URL=https://api.groq.com/openai/v1/
export LLAMA_MODEL=llama-3.1-70b-versatile
```

**Together AI:**
```bash
export LLAMA_API_KEY=<your-together-api-key>
export LLAMA_BASE_URL=https://api.together.xyz/v1/
export LLAMA_MODEL=meta-llama/Llama-2-70b-chat-hf
```

#### 2. Start Backend:
```bash
cd backend
source .venv/bin/activate  # IMPORTANT: Always activate virtual environment first!
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Start Frontend (in new terminal):
```bash
cd frontend
npm run dev
```

## 🔧 What Was Fixed

### ✅ Backend Issues Resolved:
- ✅ Virtual environment activation issue fixed
- ✅ All Python dependencies properly installed
- ✅ Document ingestion working (157 chunks processed)
- ✅ FastAPI server ready to run

### ✅ Frontend Issues Resolved:
- ✅ npm dependencies installed successfully
- ✅ Build process working correctly
- ✅ Development server ready

### ✅ Configuration Issues Resolved:
- ✅ Environment variable setup guide created
- ✅ Setup script for easy configuration
- ✅ Multiple API provider options available

## 🌐 Access Your Application

Once both services are running:
- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs

## 🧪 Test the API

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the rules for relocating a dwelling?", "k": 5}'
```

## 📋 API Keys Setup

### Get Groq API Key (Recommended):
1. Go to https://console.groq.com
2. Sign up for free account
3. Get your API key from dashboard
4. Use the setup script or export commands above

### Alternative Providers:
- **Together AI**: https://api.together.xyz
- **Replicate**: https://replicate.com
- **Local Ollama**: https://ollama.ai

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'fitz'"
**Solution**: Always activate the virtual environment first:
```bash
cd backend
source .venv/bin/activate
```

### "Command not found: pnpm"
**Solution**: Use npm instead (already installed):
```bash
npm install
npm run dev
```

### API not responding
**Solution**: Check your API key and environment variables:
```bash
echo $LLAMA_API_KEY
echo $LLAMA_BASE_URL
echo $LLAMA_MODEL
```

## 🎉 Success!

Your MDA Regulatory Chatbot is now ready to answer questions about Maryland agriculture regulations using the provided PDF document!
