import os
from pathlib import Path
print('imports done')
# Export the current project using paths relative to the working directory
PROJECT_ROOT = Path(__file__).resolve().parent
FOLDER_NAME = PROJECT_ROOT.name
OUTPUT_FILE = PROJECT_ROOT / "exported_code.txt"
print(f'Project Root: {FOLDER_NAME}')
# File types to include
INCLUDE_EXTENSIONS = {".py", ".txt", ".env", ".md"}
print('vars done')
def should_include(file: str) -> bool:
    _, ext = os.path.splitext(file)
    return ext in INCLUDE_EXTENSIONS
print('func done')
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if should_include(file):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)

                out.write(f"\n\n===== FILE: {rel_path} =====\n\n")

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"⚠️ Could not read file: {e}")

print(f"✅ Export complete: {OUTPUT_FILE}")

