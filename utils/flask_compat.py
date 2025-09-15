"""Flask/Werkzeug compatibility fixes for version mismatches."""

import sys

def fix_werkzeug_imports():
    """Fix Werkzeug import issues for Flask compatibility."""
    print("[COMPAT] Applying Werkzeug compatibility fixes...")
    
    try:
        # Try the old import location first
        from werkzeug.wrappers import BaseRequest
        print("[COMPAT] Werkzeug BaseRequest found in expected location")
        return True
    except ImportError:
        print("[COMPAT] BaseRequest not in old location, applying fix...")
        
    try:
        # Try new location (Werkzeug 2.1+)
        from werkzeug.wrappers.request import Request as BaseRequest
        from werkzeug.wrappers.response import Response as BaseResponse
        
        # Monkey-patch the old location
        import werkzeug.wrappers as wrappers
        wrappers.BaseRequest = BaseRequest
        wrappers.BaseResponse = BaseResponse
        
        print("[COMPAT] ✓ Successfully patched Werkzeug imports")
        return True
        
    except ImportError as e:
        print(f"[COMPAT] ✗ Could not fix Werkzeug imports: {e}")
        return False

def fix_flask_imports():
    """Apply all Flask compatibility fixes."""
    # Fix Werkzeug first
    werkzeug_fixed = fix_werkzeug_imports()
    
    if not werkzeug_fixed:
        print("[COMPAT] Warning: Flask may not work properly")
        return False
    
    return True

# Auto-apply fixes when imported
if __name__ != "__main__":
    fix_flask_imports()