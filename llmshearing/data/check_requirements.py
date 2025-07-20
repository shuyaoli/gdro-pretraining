#!/usr/bin/env python3
"""
Check requirements for optimized RedPajama preprocessing.
"""

import sys
import os
import subprocess

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description} found at {filepath}")
        return True
    else:
        print(f"✗ {description} NOT found at {filepath}")
        return False

def main():
    print("Checking requirements for optimized RedPajama preprocessing...")
    print("=" * 60)
    
    all_good = True
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 7):
        print("✗ Python 3.7+ is required")
        all_good = False
    else:
        print("✓ Python version is sufficient")
    
    print()
    
    # Check required Python packages
    print("Checking Python packages:")
    packages = [
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("streaming", "streaming"),
        ("transformers", "transformers"),
    ]
    
    for package, import_name in packages:
        if not check_python_package(package, import_name):
            all_good = False
    
    print()
    
    # Check optional packages
    print("Checking optional packages:")
    optional_packages = [
        ("torch", "torch"),
        ("tokenizers", "tokenizers"),
    ]
    
    for package, import_name in optional_packages:
        check_python_package(package, import_name)
    
    print()
    
    # Check required files
    print("Checking required files:")
    files_to_check = [
        ("../urls.txt", "URLs file"),
        ("tokenizer.model", "Tokenizer file (optional, will use HuggingFace if missing)"),
    ]
    
    for filepath, description in files_to_check:
        if "optional" in description.lower():
            check_file_exists(filepath, description)
        elif not check_file_exists(filepath, description):
            if "urls.txt" in filepath:
                all_good = False
    
    print()
    print("=" * 60)
    
    if all_good:
        print("✓ All requirements satisfied! You can run the optimized preprocessing.")
        print()
        print("To start preprocessing:")
        print("  bash run_complete_optimized_pipeline.sh")
    else:
        print("✗ Some requirements are missing. Please install missing packages:")
        print()
        print("Install missing packages with:")
        print("  pip install streaming numpy tqdm transformers")
        print()
        print("Make sure urls.txt file exists in the parent directory.")
    
    print()

if __name__ == "__main__":
    main()
