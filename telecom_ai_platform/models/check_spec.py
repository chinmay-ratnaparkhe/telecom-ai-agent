# save as check_spec.py and run with the environment you use (telecom-kpi)
import importlib.util, importlib, sys, pathlib
spec = importlib.util.find_spec('telecom_ai_platform.models.enhanced_autoencoder')
print("spec:", spec)
if spec:
    print("spec.origin:", spec.origin)
    p = pathlib.Path(spec.origin)
    print("exists:", p.exists(), "size:", p.stat().st_size)
    b = p.read_bytes()
    print("contains NUL bytes?:", b'\x00' in b)
    print("first 120 bytes (repr):", repr(b[:120]))
else:
    print("Module spec not found; the package path or module is not on sys.path.")
print("sys.path (first 12):")
for i,p in enumerate(sys.path[:12]):
    print(i, p)
