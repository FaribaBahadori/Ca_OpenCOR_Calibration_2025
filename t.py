import sys, platform, site, os
print("Python executable :", sys.executable)
print("Python version    :", sys.version)
print("Implementation    :", platform.python_implementation())
print("sys.prefix        :", sys.prefix)
print("site-packages     :", getattr(site, "getsitepackages", lambda: [])())
print("PYTHONPATH env    :", os.environ.get("PYTHONPATH"))