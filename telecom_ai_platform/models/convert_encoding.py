# convert_encoding.py
from pathlib import Path
p = Path(r"C:\Users\jcati\Desktop\Chinmay\Extras\Telecom-AI-Agent\telecom_ai_platform\models\enhanced_autoencoder.py")

if not p.exists():
    raise SystemExit(f"File not found: {p}")

data = p.read_bytes()
if b'\x00' not in data:
    print("No NUL bytes found; file appears fine. Exiting.")
    raise SystemExit

# try to decode using BOM-detected encodings
for enc in ("utf-16", "utf-16-le", "utf-16-be", "utf-32"):
    try:
        text = data.decode(enc)
        print("Successfully decoded using:", enc)
        break
    except Exception:
        text = None

if text is None:
    raise SystemExit("Could not decode file automatically. Open file in an editor to inspect.")

# backup original and write UTF-8
bak = p.with_suffix(".py.bak")
print("Backing up original to:", bak)
p.rename(bak)
p.write_text(text, encoding="utf-8")
print("Wrote UTF-8 file to:", p)
