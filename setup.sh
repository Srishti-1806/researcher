#!/bin/bash

# Developer Research AI Agent - Setup Script
# ==========================================

echo "🔧 Setting up Developer Research AI Agent"
echo "=========================================="
echo ""

# Create .env from .env.example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env created. Please update it with your API keys."
    echo ""
else
    echo "✅ .env already exists"
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p output
mkdir -p qdrant_db
echo "✅ Directories created"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python $python_version found"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
echo "This may take a few minutes..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"
echo ""

echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Run: ./run.sh cli         (for CLI mode)"
echo "3. Run: ./run.sh streamlit   (for Streamlit UI)"
echo ""
echo "For more details, see README.md"
