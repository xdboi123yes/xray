import json
from pathlib import Path

notebook_dir = Path("/Users/alperen/Desktop/xray/notebooks")
notebook_files = [
    notebook_dir / "xray_colab_ablation_a100.ipynb",
    notebook_dir / "xray_colab_training_auto.ipynb"
]

cleanup_lines = [
    "import shutil\n",
    "shutil.rmtree('/content/sample_data', ignore_errors=True)  # Clean default Colab folder\n",
    "\n"
]

for nb_path in notebook_files:
    if not nb_path.exists():
        print(f"Skipping {nb_path} as it does not exist.")
        continue

    print(f"Modifying {nb_path}...")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        modified = False
        for cell in data.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                if source and any("CONFIGURATION" in line for line in source):
                    # Check if already inserted
                    if not any("sample_data" in line for line in source):
                        # Insert cleanup right after the first header block
                        # The header usually takes 3 lines: # ===, # CONFIGURATION, # ===
                        insert_idx = 0
                        for idx, line in enumerate(source):
                            if line.startswith("# ============================================================================"):
                                # If it's the second border line, insert right after it
                                if idx > 0:
                                    insert_idx = idx + 1
                                    break
                        if insert_idx == 0:
                            insert_idx = 3 if len(source) >= 3 else 0
                        
                        source[insert_idx:insert_idx] = cleanup_lines
                        cell["source"] = source
                        modified = True
                        print(f"  Successfully added cleanup command to configuration cell in {nb_path.name}")
                        break

        if modified:
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
                # Add trailing newline like original format
                f.write("\n")
            print(f"  Saved changes to {nb_path.name}")
        else:
            print(f"  No changes needed or configuration cell not found in {nb_path.name}")

    except Exception as e:
        print(f"  Error modifying {nb_path.name}: {e}")
