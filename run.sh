#!/bin/bash

echo "🚀 Starting Developer Research AI Agent..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Handle Qdrant container
echo "📦 Setting up Qdrant database..."
if docker ps -q -f name=qdrant | grep -q .; then
    echo "✅ Qdrant container already running"
else
    echo "🔄 Starting Qdrant container..."
    docker stop qdrant 2>/dev/null || true
    docker rm qdrant 2>/dev/null || true
    docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
    if [ $? -ne 0 ]; then
        echo "❌ Failed to start Qdrant container"
        exit 1
    fi
    echo "⏳ Waiting for Qdrant to be ready..."
    sleep 5
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔑 Creating .env file..."
    cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL_NAME=llama-3.1-8b-instant
GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
QDRANT_URL=http://localhost:6333
EOF
    echo "⚠️  Please edit .env file with your actual API keys!"
fi

# Verify setup
echo "🔍 Verifying setup..."
python verify_setup.py
if [ $? -ne 0 ]; then
    echo "❌ Setup verification failed. Please fix the issues above."
    exit 1
fi

# Start the application
echo "🎯 Starting Streamlit application..."
python -m streamlit run app.py --server.headless true --server.port 8503

