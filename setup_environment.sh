#!/bin/bash

# Setup script for Vietnamese Aspect-based Sentiment Analysis

echo "Setting up environment for Vietnamese ABSA..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

# Create virtual environment with Python 3.8 (compatible with underthesea)
echo "Creating virtual environment..."
if command -v python3.8 &> /dev/null; then
    python3.8 -m venv venv_absa
    echo "Using Python 3.8"
elif command -v python3.9 &> /dev/null; then
    python3.9 -m venv venv_absa
    echo "Using Python 3.9"
else
    python3 -m venv venv_absa
    echo "Using default Python 3"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_absa/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install basic dependencies first
echo "Installing basic dependencies..."
pip install numpy>=1.19.0
pip install pandas>=1.3.0
pip install scikit-learn>=0.20.0
pip install joblib>=0.14.1
pip install click>=7.0

# Install underthesea dependencies
echo "Installing underthesea dependencies..."
pip install python-crfsuite>=0.9.6
pip install fasttext>=0.9.1

# Install underthesea with specific method
echo "Installing underthesea..."
pip install underthesea==1.3.5a0 --no-deps
pip install underthesea==1.3.5a0

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other requirements..."
pip install emoji>=2.0.0
pip install beautifulsoup4>=4.10.0
pip install html2text>=2020.1.16
pip install regex>=2022.1.18
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install tqdm>=4.62.0
pip install gensim>=4.1.0
pip install googletrans==4.0.0rc1
pip install deep-translator>=1.8.0
pip install requests>=2.25.0
pip install urllib3>=1.26.0

# Clone Vietnamese preprocessing repository
echo "Cloning Vietnamese preprocessing repository..."
if [ ! -d "vietnamese-preprocess" ]; then
    git clone https://github.com/chienthan2vn/vietnamese-preprocess.git
fi

# Install Vietnamese preprocessing package
echo "Installing Vietnamese preprocessing package..."
cd vietnamese-preprocess
pip install -e .
cd ..

# Create necessary directories
echo "Creating directories..."
mkdir -p data_pr
mkdir -p models
mkdir -p logs

# Test underthesea installation
echo "Testing underthesea installation..."
python -c "
try:
    import underthesea
    print('✓ Underthesea imported successfully')

    # Test word tokenization
    test_text = 'Xin chào, tôi là người Việt Nam'
    tokens = underthesea.word_tokenize(test_text)
    print(f'✓ Word tokenization test: {tokens}')

    # Test POS tagging
    pos_tags = underthesea.pos_tag(test_text)
    print(f'✓ POS tagging test: {pos_tags}')

    print('✓ Underthesea installation successful!')

except ImportError as e:
    print(f'✗ Import error: {e}')
    print('Trying alternative installation...')

except Exception as e:
    print(f'✗ Runtime error: {e}')
    print('Underthesea imported but some functions may not work')
"

# Alternative underthesea installation if first attempt fails
if [ $? -ne 0 ]; then
    echo "Trying alternative underthesea installation..."
    pip uninstall underthesea -y
    pip install underthesea==1.3.4

    python -c "
    try:
        import underthesea
        print('✓ Alternative underthesea installation successful!')
        tokens = underthesea.word_tokenize('Xin chào')
        print(f'✓ Test result: {tokens}')
    except Exception as e:
        print(f'✗ Alternative installation also failed: {e}')
        print('Please install underthesea manually')
    "
fi

echo ""
echo "Environment setup complete!"
echo "To activate the environment, run: source venv_absa/bin/activate"
echo ""
echo "If underthesea installation failed, try:"
echo "1. source venv_absa/bin/activate"
echo "2. pip install underthesea==1.3.4"
echo "3. python -c \"import underthesea; print(underthesea.word_tokenize('test'))\""
