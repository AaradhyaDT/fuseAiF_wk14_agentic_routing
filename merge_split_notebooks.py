import json, os

def merge():
    print("=== Merging Split Notebook Results into support_routing.ipynb ===")
    
    enc_file = "encoder_results.json"
    slm_file = "slm_results.json"
    master_file = "support_routing.ipynb"
    
    if not os.path.exists(master_file):
        print(f"Error: {master_file} not found!")
        return

    with open(master_file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update SLM hyperparameters in master notebook
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "SLM_BATCH_SIZE = 8" in src:
            cell["source"] = [
                line.replace("SLM_BATCH_SIZE = 8", "SLM_BATCH_SIZE = 16").replace("SLM_EPOCHS = 3", "SLM_EPOCHS = 2")
                for line in cell["source"]
            ]
            print("  Updated master notebook SLM parameters (batch_size=16, epochs=2).")

    # 2. Add JSON cache loader to Section 4 (Comparison) in master notebook
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "comparison_rows = []" in src:
            loader_header = [
                "# Load pre-computed JSON results if present\n",
                "import os, json\n",
                "if ('encoder_results' not in locals() or not encoder_results) and os.path.exists('encoder_results.json'):\n",
                "    with open('encoder_results.json', 'r', encoding='utf-8') as f:\n",
                "        encoder_results = json.load(f)\n",
                "    valid_encoders = {k: v for k, v in encoder_results.items() if 'error' not in v}\n",
                "if ('slm_results' not in locals() or not slm_results) and os.path.exists('slm_results.json'):\n",
                "    with open('slm_results.json', 'r', encoding='utf-8') as f:\n",
                "        slm_results = json.load(f)\n",
                "    valid_slms = {k: v for k, v in slm_results.items() if 'error' not in v}\n",
                "\n"
            ]
            cell["source"] = loader_header + [line for line in cell["source"] if not line.startswith("valid_encoders =") and not line.startswith("valid_slms =")]
            print("  Added cache loading support to Section 4 Comparison cell in master notebook.")

    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"  Successfully updated {master_file}.")

if __name__ == "__main__":
    merge()
