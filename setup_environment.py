"""
Setup script for running the AI vs Human text analysis pipeline in cloud environments.
Compatible with Google Colab and Kaggle notebooks.
"""

import os
import sys
import subprocess
from pathlib import Path


def detect_environment():
    """Detect if running in Colab, Kaggle, or local environment."""
    if 'google.colab' in sys.modules:
        return 'colab'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    else:
        return 'local'


def setup_colab():
    """Setup for Google Colab environment."""
    print("🔧 Setting up for Google Colab...")
    
    # Clone repository if not already present
    if not Path('/content/AIvsHuman').exists():
        print("📥 Cloning repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/YOUR_USERNAME/AIvsHuman.git',
            '/content/AIvsHuman'
        ], check=False)
        os.chdir('/content/AIvsHuman')
    else:
        os.chdir('/content/AIvsHuman')
        print("✓ Repository already cloned")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
    
    # Download spaCy model
    print("📥 Downloading spaCy model...")
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_lg'], check=True)
    
    # Create necessary directories
    print("📁 Creating directories...")
    Path('results').mkdir(exist_ok=True)
    Path('cache').mkdir(exist_ok=True)
    
    print("✅ Colab setup complete!")
    return '/content/AIvsHuman'


def setup_kaggle():
    """Setup for Kaggle environment."""
    print("🔧 Setting up for Kaggle...")
    
    # Set working directory
    work_dir = Path('/kaggle/working/AIvsHuman')
    
    # Clone repository if not already present
    if not work_dir.exists():
        print("📥 Cloning repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/YOUR_USERNAME/AIvsHuman.git',
            str(work_dir)
        ], check=False)
        os.chdir(str(work_dir))
    else:
        os.chdir(str(work_dir))
        print("✓ Repository already cloned")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
    
    # Download spaCy model
    print("📥 Downloading spaCy model...")
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_lg'], check=True)
    
    # Create necessary directories
    print("📁 Creating directories...")
    Path('results').mkdir(exist_ok=True)
    Path('cache').mkdir(exist_ok=True)
    
    print("✅ Kaggle setup complete!")
    return str(work_dir)


def setup_local():
    """Setup for local environment."""
    print("🔧 Setting up local environment...")
    
    # Check if virtual environment exists
    venv_path = Path('.venv')
    if not venv_path.exists():
        print("⚠️  No virtual environment found. Please create one:")
        print("    python -m venv .venv")
        print("    source .venv/bin/activate  # On Windows: .venv\\Scripts\\Activate.ps1")
        return None
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    
    # Download spaCy model
    print("📥 Downloading spaCy model...")
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_lg'], check=True)
    
    # Create necessary directories
    print("📁 Creating directories...")
    Path('results').mkdir(exist_ok=True)
    Path('cache').mkdir(exist_ok=True)
    
    print("✅ Local setup complete!")
    return str(Path.cwd())


def get_sample_data():
    """Download or create sample data for testing."""
    print("📊 Checking for sample data...")
    
    sample_file = Path('data/stub_train.csv')
    if sample_file.exists():
        print(f"✓ Sample data found: {sample_file}")
        return str(sample_file)
    else:
        print("⚠️  Sample data not found. Please add your data to data/ directory")
        return None


def main():
    """Main setup function."""
    print("=" * 60)
    print("AI vs Human Text Analysis - Environment Setup")
    print("=" * 60)
    
    env = detect_environment()
    print(f"🌍 Environment detected: {env.upper()}")
    print()
    
    # Setup based on environment
    if env == 'colab':
        work_dir = setup_colab()
    elif env == 'kaggle':
        work_dir = setup_kaggle()
    else:
        work_dir = setup_local()
    
    if work_dir is None:
        print("\n❌ Setup incomplete. Please follow the instructions above.")
        return False
    
    print()
    print("=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("📖 Quick Start:")
    print()
    print("1. Run basic analysis (without embeddings):")
    print("   python -m src.cli run \\")
    print("     --input data/stub_train.csv \\")
    print("     --output results/test_run \\")
    print("     --shards 2 --workers 2")
    print()
    print("2. Run statistical tests:")
    print("   python run_stats_tests.py results/test_run")
    print()
    print("3. Generate plots (optional):")
    print("   Plots are not produced by a repository helper. Use CSVs in results/<run>/lexical/ to create figures.")
    print()
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
