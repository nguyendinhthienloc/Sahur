"""
Verify reproducibility setup - checks for common issues that break reproducibility.
Run this before committing or sharing your project.
"""

import os
import sys
from pathlib import Path
import re
import subprocess


def check_gitignore():
    """Check if .gitignore properly excludes heavy files."""
    print("📝 Checking .gitignore...")
    
    gitignore = Path('.gitignore')
    if not gitignore.exists():
        print("  ❌ .gitignore not found!")
        return False
    
    content = gitignore.read_text()
    required_patterns = [
        '.venv/', 'venv/', '__pycache__/', 
        'cache/', '*.pyc', 'results/',
        '*.pt', '*.bin', '*.npy'
    ]
    
    missing = []
    for pattern in required_patterns:
        if pattern not in content:
            missing.append(pattern)
    
    if missing:
        print(f"  ⚠️  Missing patterns: {', '.join(missing)}")
        return False
    else:
        print("  ✓ .gitignore looks good")
        return True


def check_hardcoded_paths():
    """Check for hardcoded absolute paths in Python files."""
    print("\n🔍 Checking for hardcoded paths...")
    
    # Patterns to look for (with context to avoid false positives)
    dangerous_patterns = [
        (r'^[C-Z]:\\[^\\]', 'Windows absolute path at line start'),
        (r'["\']([C-Z]:\\\\[^"\']*)["\']', 'Windows absolute path in string'),
        (r'["\'](\\/Users\\/[a-zA-Z0-9_]+\\/[^"\']*)["\']', 'Mac user path'),
        (r'["\'](\\/home\\/[a-zA-Z0-9_]+\\/[^"\']*)["\']', 'Linux user path'),
    ]
    
    issues = []
    python_files = list(Path('src').rglob('*.py'))
    python_files.extend(list(Path('tests').rglob('*.py')))
    python_files.extend(Path('.').glob('*.py'))
    
    for py_file in python_files:
        if 'archive' in str(py_file) or py_file.name == 'check_reproducibility.py':
            continue
        
        try:
            content = py_file.read_text(encoding='utf-8')
            for pattern, description in dangerous_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Exclude comments and strings that might be documentation
                    for match in matches:
                        if match not in ['D:\\', 'C:\\']:  # Common examples in docs
                            issues.append({
                                'file': str(py_file),
                                'issue': description,
                                'match': match
                            })
        except Exception as e:
            print(f"  ⚠️  Could not read {py_file}: {e}")
    
    if issues:
        print("  ❌ Found hardcoded paths:")
        for issue in issues[:10]:  # Show first 10
            print(f"    - {issue['file']}: {issue['issue']} ({issue['match']})")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
        return False
    else:
        print("  ✓ No hardcoded paths found")
        return True


def check_requirements():
    """Check if requirements.txt exists and has essential packages."""
    print("\n📦 Checking requirements.txt...")
    
    req_file = Path('requirements.txt')
    if not req_file.exists():
        print("  ❌ requirements.txt not found!")
        return False
    
    content = req_file.read_text()
    essential = [
        'spacy', 'pandas', 'numpy', 'click',
        'lexicalrichness', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for pkg in essential:
        if pkg not in content.lower():
            missing.append(pkg)
    
    if missing:
        print(f"  ⚠️  Missing packages: {', '.join(missing)}")
        return False
    else:
        print("  ✓ requirements.txt looks good")
        return True


def check_relative_imports():
    """Check that imports use relative paths, not absolute."""
    print("\n🔗 Checking imports...")
    
    python_files = list(Path('src').rglob('*.py'))
    
    issues = []
    for py_file in python_files:
        try:
            content = py_file.read_text(encoding='utf-8')
            # Look for imports that might reference user-specific paths
            if 'sys.path.append' in content:
                issues.append(f"{py_file}: Uses sys.path.append")
        except Exception as e:
            pass
    
    if issues:
        print("  ⚠️  Found manual path modifications:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✓ No sys.path modifications found")
        return True


def check_documentation():
    """Check if setup documentation exists."""
    print("\n📚 Checking documentation...")
    
    docs = ['README.md', 'requirements.txt']
    missing = []
    
    for doc in docs:
        if not Path(doc).exists():
            missing.append(doc)
    
    if missing:
        print(f"  ❌ Missing: {', '.join(missing)}")
        return False
    else:
        print("  ✓ Essential documentation present")
        return True


def check_setup_script():
    """Check if setup script exists for cloud environments."""
    print("\n🚀 Checking cloud setup...")
    
    if Path('setup_environment.py').exists():
        print("  ✓ setup_environment.py found")
        return True
    else:
        print("  ⚠️  setup_environment.py not found (optional but recommended)")
        return False


def check_data_in_repo():
    """Check if large data files are accidentally tracked in git."""
    print("\n💾 Checking for large data files...")
    
    try:
        # Check if we're in a git repo
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("  ℹ️  Not a git repository")
            return True
        
        # Check git-tracked files in data directory
        result = subprocess.run(['git', 'ls-files', 'data/'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("  ℹ️  Could not check git-tracked files")
            return True
        
        tracked_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        large_tracked = []
        
        for file_path in tracked_files:
            full_path = Path(file_path)
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                if size_mb > 10:  # Files larger than 10MB
                    large_tracked.append((file_path, size_mb))
        
        if large_tracked:
            print("  ❌ Large files tracked in git:")
            for file, size in large_tracked[:5]:
                print(f"    - {file}: {size:.1f} MB")
            print("  💡 Run: git rm --cached <file> to untrack")
            return False
        else:
            print("  ✓ No large data files tracked in git")
            return True
            
    except FileNotFoundError:
        print("  ℹ️  Git not available, skipping check")
        return True
    except Exception as e:
        print(f"  ⚠️  Error checking git: {e}")
        return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("🔎 Reproducibility Check")
    print("=" * 60)
    print()
    
    if not Path('src').exists():
        print("❌ Not in project root directory! Run from AIvsHuman/")
        return False
    
    checks = [
        check_gitignore(),
        check_hardcoded_paths(),
        check_requirements(),
        check_relative_imports(),
        check_documentation(),
        check_setup_script(),
        check_data_in_repo(),
    ]
    
    print()
    print("=" * 60)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print("\n✨ Your project is reproducible!")
        return True
    else:
        print(f"⚠️  {total - passed} checks failed ({passed}/{total} passed)")
        print("\n💡 Fix the issues above to improve reproducibility")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
