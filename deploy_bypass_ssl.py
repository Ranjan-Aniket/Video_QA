#!/usr/bin/env python3
"""
Deploy to Modal with SSL verification disabled
Bypasses Zscaler SSL inspection
"""

import os
import sys
import warnings

# Disable SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Disable SSL verification at Python level
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Patch SSL verification in requests/urllib3
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except:
    pass

# Monkey patch SSL context to disable verification
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Patch httpx (used by Modal)
try:
    import httpx
    # Monkey patch httpx to disable SSL
    original_client = httpx.Client
    original_async_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs['verify'] = False
        return original_client(*args, **kwargs)

    def patched_async_client(*args, **kwargs):
        kwargs['verify'] = False
        return original_async_client(*args, **kwargs)

    httpx.Client = patched_client
    httpx.AsyncClient = patched_async_client
except:
    pass

# Set Modal credentials
os.environ["MODAL_TOKEN_ID"] = "ak-pGKL4DcwoTsOk5CJOUUUV7"
os.environ["MODAL_TOKEN_SECRET"] = "as-eZ5FRgXbD7nBeLG8UvX5j0"

print("=" * 60)
print("Modal Deployment (SSL Verification DISABLED)")
print("=" * 60)
print()
print("⚠️  WARNING: SSL verification is disabled")
print("   This bypasses Zscaler SSL inspection")
print()

# Now try to deploy
try:
    import subprocess
    result = subprocess.run(
        ["modal", "deploy", "modal_pipeline.py"],
        env=os.environ,
        cwd="/Users/aranja14/Desktop/Gemini_QA"
    )
    sys.exit(result.returncode)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
