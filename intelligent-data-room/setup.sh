#!/bin/bash

# Intelligent Data Room - Setup Script
# Run this to set up the development environment

echo "🚀 Setting up Intelligent Data Room..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment (platform-specific)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # macOS/Linux
    source venv/bin/activate
fi

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment
echo "🔑 Setting up API key configuration..."
cp .env.example .env
echo ""
echo "⚠️  IMPORTANT: Edit .env and add your GOOGLE_API_KEY"
echo ""

# Test imports
echo "✅ Testing imports..."
python -c "import streamlit; import pandas; import pandasai; import google.generativeai; print('✅ All imports successful!')"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your GOOGLE_API_KEY"
echo "2. Run: streamlit run app.py"
echo "3. Upload sample_data/sales_data.csv to test"
echo ""
