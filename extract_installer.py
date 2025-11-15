import re
import os
import sys

print("\n==============================")
print("  Installer File Extractor")
print("  (Windows / Same Folder Mode)")
print("==============================\n")

if len(sys.argv) < 2:
    print("Usage: python extract_installer.py <installer.sh>")
    sys.exit(1)

installer_path = sys.argv[1]

if not os.path.exists(installer_path):
    print(f"Error: File not found → {installer_path}")
    sys.exit(1)

# Output base = current working directory
BASE_DIR = os.getcwd()

print(f"[+] Extracting into current directory:")
print(f"    {BASE_DIR}\n")

current_file = None
capture_mode = False
buffer = []

# Matches: cat > file << 'EOF'
start_pattern = re.compile(r"^cat\s*>\s*(.*?)\s*<<\s*[\'\"]?(\w+)[\'\"]?$")

# Allowed end tokens
end_tokens = {
    "EOF",
    "RECORDER_EOF",
    "STT_EOF",
    "TRANS_EOF",
    "SUMM_EOF",
    "LOGGER_EOF",
    "SERVER_EOF",
    "FRONTEND_EOF",
    "TEST_EOF"
}

with open(installer_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")

        if not capture_mode:
            match = start_pattern.match(line)
            if match:
                rel_path = match.group(1).strip()
                current_file = os.path.join(BASE_DIR, rel_path)

                print(f"[+] Writing → {rel_path}")

                os.makedirs(os.path.dirname(current_file), exist_ok=True)

                buffer = []
                capture_mode = True
            continue

        if capture_mode:
            stripped = line.strip()

            if stripped in end_tokens:
                with open(current_file, "w", encoding="utf-8") as out:
                    out.write("\n".join(buffer) + "\n")

                print(f"    [✓] File saved: {current_file}")

                capture_mode = False
                current_file = None
                buffer = []
                continue

            buffer.append(line)

print("\n========================================")
print(" Extraction Complete!")
print(" All files extracted in this folder.")
print("========================================\n")
