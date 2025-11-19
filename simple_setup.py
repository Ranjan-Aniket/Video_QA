#!/usr/bin/env python3
"""
Simple setup script - fixes all issues
"""
from pathlib import Path
import subprocess
import sys

def create_files():
    """Create all necessary files"""
    print("Creating files...")
    
    # Root __init__.py
    Path("__init__.py").write_text('"""Gemini Q&A"""\n__version__ = "0.1.0"\n')
    
    # config/__init__.py
    Path("config/__init__.py").write_text('''"""Configuration module"""
from config.settings import settings, get_settings
__all__ = ["settings", "get_settings"]
''')
    
    # database/__init__.py (copy from init.py if it exists)
    if Path("database/init.py").exists():
        Path("database/__init__.py").write_text(Path("database/init.py").read_text())
    else:
        Path("database/__init__.py").write_text('"""Database package"""\n')
    
    # utils/__init__.py
    Path("utils/__init__.py").write_text('''"""Utils package"""
from utils.cost_tracker import *
from utils.helpers import *
from utils.logger import *
''')
    
    # pytest.ini
    Path("pytest.ini").write_text('''[pytest]
pythonpath = .
testpaths = .
python_files = test_*.py
addopts = -v --tb=short
''')
    
    # setup.py
    Path("setup.py").write_text('''from setuptools import setup, find_packages

setup(
    name="gemini-qa",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
)
''')
    
    # .env file with FIXED cors_origins
    if not Path(".env").exists():
        Path(".env").write_text('''# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/video_qa

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GEMINI_API_KEY=

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# CORS - must be JSON format for List[str]
CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]
''')
        print("  ✅ Created .env file")
    else:
        # Fix existing .env if cors_origins is wrong
        env_content = Path(".env").read_text()
        if "CORS_ORIGINS=" in env_content:
            # Check if it's not in JSON format
            if not "CORS_ORIGINS=[" in env_content:
                lines = env_content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith("CORS_ORIGINS=") and not line.startswith("CORS_ORIGINS=["):
                        new_lines.append('CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]')
                    else:
                        new_lines.append(line)
                Path(".env").write_text('\n'.join(new_lines))
                print("  ✅ Fixed CORS_ORIGINS in .env")
        else:
            # Add CORS_ORIGINS if missing
            with open(".env", "a") as f:
                f.write('\n# CORS - must be JSON format\n')
                f.write('CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]\n')
            print("  ✅ Added CORS_ORIGINS to .env")
    
    print("✅ All files created")

def install_packages():
    """Install packages"""
    print("\nInstalling packages...")
    
    # Install package in editable mode
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Package installed")
    except subprocess.CalledProcessError:
        print("⚠️  Package installation failed (not critical)")
    
    # Install spacy model
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✅ Spacy model installed")
    except subprocess.CalledProcessError:
        print("⚠️  Spacy model installation failed - run manually: python -m spacy download en_core_web_sm")

def main():
    print("="*60)
    print("🚀 Gemini Q&A - Simple Setup")
    print("="*60)
    
    try:
        create_files()
        install_packages()
        
        print("\n" + "="*60)
        print("✅ Setup Complete!")
        print("="*60)
        print("\n📋 Next steps:")
        print("1. Update .env with your API keys")
        print("2. Run tests: pytest -v")
        print("\nIf spacy install failed, run:")
        print("  python -m spacy download en_core_web_sm")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())