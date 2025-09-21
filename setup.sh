#!/bin/bash

# MDA Regulatory Chatbot Setup Script
# This script helps you set up the environment variables and run the project

echo "🚀 Setting up MDA Regulatory Chatbot..."

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Function to set up environment variables
setup_env() {
    echo "📝 Setting up environment variables..."
    echo ""
    echo "Choose your Llama API provider:"
    echo "1) Groq (Recommended - Fast & Free)"
    echo "2) Together AI"
    echo "3) Local Ollama"
    echo "4) Skip (use existing environment)"
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            read -p "Enter your Groq API key: " api_key
            export LLAMA_API_KEY="$api_key"
            export LLAMA_BASE_URL="https://api.groq.com/openai/v1/"
            export LLAMA_MODEL="llama-3.1-70b-versatile"
            echo "✅ Groq configuration set"
            ;;
        2)
            read -p "Enter your Together AI API key: " api_key
            export LLAMA_API_KEY="$api_key"
            export LLAMA_BASE_URL="https://api.together.xyz/v1/"
            export LLAMA_MODEL="meta-llama/Llama-2-70b-chat-hf"
            echo "✅ Together AI configuration set"
            ;;
        3)
            export LLAMA_API_KEY="dummy-key"
            export LLAMA_BASE_URL="http://localhost:11434/v1/"
            export LLAMA_MODEL="llama2:7b"
            echo "✅ Local Ollama configuration set"
            echo "⚠️  Make sure Ollama is running: ollama serve"
            ;;
        4)
            echo "⏭️  Skipping environment setup"
            ;;
        *)
            echo "❌ Invalid choice"
            exit 1
            ;;
    esac
}

# Function to start backend
start_backend() {
    echo "🔧 Starting backend..."
    cd backend
    if [ ! -d ".venv" ]; then
        echo "❌ Virtual environment not found. Please run the setup from README.md first."
        exit 1
    fi
    
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "🚀 Starting FastAPI server on http://localhost:8000"
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting frontend..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing frontend dependencies..."
        npm install
    fi
    echo "🚀 Starting frontend development server"
    npm run dev
}

# Main menu
echo ""
echo "What would you like to do?"
echo "1) Set up environment variables"
echo "2) Start backend only"
echo "3) Start frontend only"
echo "4) Start both (recommended)"
echo "5) Test backend setup"
read -p "Enter your choice (1-5): " main_choice

case $main_choice in
    1)
        setup_env
        echo ""
        echo "💡 To make these environment variables permanent, add them to your shell profile:"
        echo "   echo 'export LLAMA_API_KEY=$LLAMA_API_KEY' >> ~/.zshrc"
        echo "   echo 'export LLAMA_BASE_URL=$LLAMA_BASE_URL' >> ~/.zshrc"
        echo "   echo 'export LLAMA_MODEL=$LLAMA_MODEL' >> ~/.zshrc"
        echo "   source ~/.zshrc"
        ;;
    2)
        start_backend
        ;;
    3)
        start_frontend
        ;;
    4)
        setup_env
        echo "🚀 Starting both backend and frontend..."
        echo "Backend will be available at: http://localhost:8000"
        echo "Frontend will be available at: http://localhost:5173"
        echo ""
        # Start backend in background
        (cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) &
        BACKEND_PID=$!
        sleep 3
        # Start frontend
        (cd frontend && npm run dev) &
        FRONTEND_PID=$!
        
        echo "✅ Both services started!"
        echo "Press Ctrl+C to stop both services"
        
        # Wait for user interrupt
        trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
        wait
        ;;
    5)
        echo "🧪 Testing backend setup..."
        cd backend
        source .venv/bin/activate
        python -c "
import os
os.environ['LLAMA_API_KEY'] = 'test-key'
os.environ['LLAMA_BASE_URL'] = 'https://api.groq.com/openai/v1/'
os.environ['LLAMA_MODEL'] = 'llama-3.1-70b-versatile'
from app.rag import RAGEngine
rag = RAGEngine(data_dir='../data')
print('✅ RAGEngine imported successfully')
rag.ingest()
print(f'✅ Document ingestion successful - {len(rag.df)} chunks processed')
print('✅ Backend setup is working correctly!')
"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
