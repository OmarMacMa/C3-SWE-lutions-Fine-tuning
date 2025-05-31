import ctypes
import json

# Load the shared library
lib = ctypes.CDLL("../Compiler/compiler/paradigm_analyzer.so")
lib.analyze_code_paradigm.argtypes = [ctypes.c_char_p]
lib.analyze_code_paradigm.restype = ctypes.c_char_p

# Process JSON
with open('training_lite.json', 'r') as f:
    data = json.load(f)

for entry in data:
    code = entry['input']['code_snippet'].encode('utf-8')
    print(f"Processing code snippet: {code.decode('utf-8')}")
    paradigm = lib.analyze_code_paradigm(code).decode('utf-8')
    entry['input']['paradigm'] = paradigm

with open('training_lite_paradigm.json', 'w') as f:
    json.dump(data, f, indent=2)
