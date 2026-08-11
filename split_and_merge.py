"""
support_routing.ipynb splitter/merger.
Split boundary: by Approach (matches TOC). Merge: nbformat, keyed on native cell IDs.

Usage:
  python split_and_merge.py split   support_routing.ipynb
      -> support_routing_partA_encoders.ipynb  (Setup+Data+Approach1, cells 0-24)
      -> support_routing_partB_slms.ipynb      (Setup+Data shared 0-11 + Approach2 25-41)

  python split_and_merge.py merge   support_routing.ipynb partA_run.ipynb partB_run.ipynb
      -> support_routing_merged.ipynb
      Rule: for any cell id present in a run file with non-empty outputs/execution_count,
      pull that cell (source+outputs+execution_count) from the run file into the base
      structure. Base structure = original support_routing.ipynb (preserves cells 42-51
      untouched, and preserves ordering). Conflicts (same id executed in both parts,
      e.g. Setup/Data cells) -> Part A wins, logged to stderr.
"""
import json, sys, copy

SETUP_DATA = list(range(0, 12))      # 0-11
APPROACH1  = list(range(12, 26))     # 12-25, includes export_encoder_json_master
APPROACH2  = list(range(26, 44))     # 26-43, includes export_slm_json_master
TAIL       = list(range(44, 55))     # 44-54, merge-only, not in either part


def load(path):
    return json.load(open(path))


def save(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")


def build_part(nb, indices, title_suffix):
    part = copy.deepcopy(nb)
    part["cells"] = [copy.deepcopy(nb["cells"][i]) for i in indices]
    # retitle first markdown cell so the split files are visually distinct in Colab
    if part["cells"] and part["cells"][0]["cell_type"] == "markdown":
        src = "".join(part["cells"][0]["source"])
        part["cells"][0]["source"] = [src + f"\n\n*(Split: {title_suffix})*"]
    return part


def do_split(src_path):
    nb = load(src_path)
    part_a = build_part(nb, SETUP_DATA + APPROACH1, "Part A — Approach 1 (Encoders)")
    part_b = build_part(nb, SETUP_DATA + APPROACH2, "Part B — Approach 2 (SLMs/LoRA)")
    save(part_a, "support_routing_partA_encoders_v2.ipynb")
    save(part_b, "support_routing_partB_slms_v2.ipynb")
    print("wrote support_routing_partA_encoders_v2.ipynb (cells 0-11,12-25)")
    print("wrote support_routing_partB_slms_v2.ipynb (cells 0-11,26-43)")


def cell_has_output(cell):
    if cell["cell_type"] != "code":
        return False
    ec = cell.get("execution_count")
    outs = cell.get("outputs") or []
    return ec is not None or len(outs) > 0


def index_by_id(nb):
    return {c["id"]: c for c in nb["cells"] if "id" in c}


def do_merge(base_path, run_a_path, run_b_path):
    base = load(base_path)
    run_a = load(run_a_path)
    run_b = load(run_b_path)

    a_by_id = index_by_id(run_a)
    b_by_id = index_by_id(run_b)

    merged = copy.deepcopy(base)
    conflicts = []
    filled = []
    missing = []

    for cell in merged["cells"]:
        cid = cell.get("id")
        if cid is None or cell["cell_type"] != "code":
            continue

        a_cell = a_by_id.get(cid)
        b_cell = b_by_id.get(cid)
        a_ok = a_cell is not None and cell_has_output(a_cell)
        b_ok = b_cell is not None and cell_has_output(b_cell)

        if a_ok and b_ok:
            conflicts.append(cid)
            src = a_cell  # Part A wins on conflict
        elif a_ok:
            src = a_cell
        elif b_ok:
            src = b_cell
        else:
            missing.append(cid)
            continue

        cell["outputs"] = copy.deepcopy(src.get("outputs", []))
        cell["execution_count"] = src.get("execution_count")
        filled.append(cid)

    save(merged, "support_routing_merged.ipynb")

    print(f"wrote support_routing_merged.ipynb")
    print(f"cells filled from runs: {len(filled)}")
    if conflicts:
        print(f"CONFLICTS (Part A wins, {len(conflicts)}): {conflicts}", file=sys.stderr)
    if missing:
        print(f"NO OUTPUT IN EITHER RUN ({len(missing)}): {missing}", file=sys.stderr)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "split":
        do_split(sys.argv[2])
    elif mode == "merge":
        do_merge(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("usage: split_and_merge.py [split|merge] ...", file=sys.stderr)
        sys.exit(1)
