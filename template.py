import os
from pathlib import Path

list_of_files = [
    f"SPEECHTOTEXTTENSORFLOW/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/cloud_storage/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/components/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/configuration/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/constants/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/entity/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/exceptions/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/logger/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/models/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/pipeline/__init__.py",
    f"SPEECHTOTEXTTENSORFLOW/utils/__init__.py",
    f'setup.py',
    f'requirements.txt'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")