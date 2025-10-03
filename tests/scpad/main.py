from sim import Scratchpad
from sim.common import * 
import os, json

def dump_shift_masks(scpad, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    masks = list(scpad.seen_masks)
    with open(os.path.join(OUT_DIR, path), "w", encoding="utf-8") as f:
        json.dump({"num_unique": len(masks), "masks": masks}, f, indent=2)
    print(f"[shift-masks] unique={len(masks)}  saved_to={path}")


sc0 = Scratchpad(slots_per_bank=128)

tid_ctr = 0
for r in range(1, 33):                
    for c in range(1, 33):
        tid = f"T{tid_ctr}"

        base_row = 0
        sc0.write_tile(tile_id=tid, rows=r, cols=c, base_row=base_row)

        for rr in range(r):
            sc0.read_tile(tile_id=tid, base_row=base_row, row_id=rr, row_based=True)

        for cc in range(c):
            sc0.read_tile(tile_id=tid, base_row=base_row, col_id=cc, row_based=False)

        tid_ctr += 1

dump_shift_masks(sc0, "shift_masks.json")
