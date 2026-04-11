#!/usr/bin/env python3
"""
Generate iPath3 input from filtered CSV + COBRA JSON model.

Flow:
  filtered CSV → gene_id + type(OE/KD) + sources
      ↓
  model JSON → find all reactions containing each gene
      ↓ each reaction's annotation:
      ├─ has kegg.reaction → validate via KEGG API
      │     ├─ valid      → use R##### directly (most precise)
      │     └─ obsolete   → fallback to ec-code → check obsolete → E#####
      └─ no kegg.reaction → use ec-code → check obsolete via KEGG API → E#####
      ↓
  iPath3 format:  ID  COLOR  WIDTH
      color: OE=#ff0000 (red), KD=#0000ff (blue)
      width: ecBridge+ecFSEOF=W30, ecBridge_only=W20, ecFSEOF_only=W15
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# Step 1: Read filtered CSV
# ---------------------------------------------------------------------------
def read_filtered_csv(path: str) -> pd.DataFrame:
    """Read filtered CSV, return DataFrame with gene_id, type, sources."""
    df = pd.read_csv(path)
    required = {"gene_id", "type"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")
    # Keep only OE / KD rows
    df = df[df["type"].isin(["OE", "KD"])].copy()
    df["type"] = df["type"].str.strip().str.upper()
    print(f"[Step 1] Loaded {len(df)} targets from {path}")
    print(f"         OE: {(df['type']=='OE').sum()},  KD: {(df['type']=='KD').sum()}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Build gene → reactions mapping from model JSON
# ---------------------------------------------------------------------------
def build_gene_reaction_map(model_path: str):
    """
    Returns:
        gene_rxn_map: {gene_id: [rxn_dict, ...]}
        Each rxn_dict has keys: id, ec_codes (list), kegg_reactions (list)
    """
    with open(model_path, "r") as f:
        model = json.load(f)

    reactions = model.get("reactions", [])
    gene_rxn_map = defaultdict(list)

    for rxn in reactions:
        rxn_id = rxn.get("id", "")
        annotation = rxn.get("annotation", {})

        # Extract ec-code(s)
        ec_raw = annotation.get("ec-code", [])
        if isinstance(ec_raw, str):
            ec_raw = [ec_raw]
        ec_codes = [e.strip() for e in ec_raw if e.strip()]

        # Extract kegg.reaction(s)
        kr_raw = annotation.get("kegg.reaction", [])
        if isinstance(kr_raw, str):
            kr_raw = [kr_raw]
        kegg_reactions = [r.strip() for r in kr_raw if r.strip()]

        # Parse gene_reaction_rule → extract gene IDs
        rule = rxn.get("gene_reaction_rule", "")
        genes_in_rule = set(re.findall(r"[A-Za-z0-9_]+", rule)) if rule else set()
        # Filter out boolean keywords
        genes_in_rule -= {"and", "or", "AND", "OR", "And", "Or"}

        rxn_info = {
            "id": rxn_id,
            "ec_codes": ec_codes,
            "kegg_reactions": kegg_reactions,
        }

        for g in genes_in_rule:
            gene_rxn_map[g].append(rxn_info)

    print(f"[Step 2] Model loaded: {len(reactions)} reactions, "
          f"{len(gene_rxn_map)} genes with GPR")
    return gene_rxn_map


# ---------------------------------------------------------------------------
# Step 3: KEGG API — shared fetch utility
# ---------------------------------------------------------------------------
def _kegg_get(url: str, retries: int = 3) -> str | None:
    """Fetch KEGG REST API with retry and rate limiting."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 403 or e.code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# Step 3a: KEGG API — validate KEGG Reaction IDs
# ---------------------------------------------------------------------------
_kr_cache: dict[str, str | None] = {}  # R##### → R##### (valid) or None (obsolete)


def resolve_kegg_reaction(rid: str) -> str | None:
    """
    Validate a KEGG Reaction ID via KEGG API.

    KEGG reactions are generally stable and do not get "transferred" the way
    EC numbers do, but entries can be removed/obsoleted.  A 404 response from
    the REST endpoint means the entry no longer exists; any other successful
    response means the ID is still valid.

    Returns:
        rid   – the same ID if it is valid
        None  – if the entry is not found / obsolete
    """
    if rid in _kr_cache:
        return _kr_cache[rid]

    body = _kegg_get(f"https://rest.kegg.jp/get/{rid}")
    if body is None:
        # 404 or unrecoverable error → treat as obsolete
        print(f"         KEGG Reaction {rid} → Not found / obsolete, will fallback to EC")
        _kr_cache[rid] = None
        return None

    # Entry exists — confirm it looks like a reaction entry (starts with ENTRY)
    if body.strip().startswith("ENTRY"):
        _kr_cache[rid] = rid
        return rid

    # Unexpected body format → conservative: treat as invalid
    print(f"         KEGG Reaction {rid} → Unexpected response format, will fallback to EC")
    _kr_cache[rid] = None
    return None


# ---------------------------------------------------------------------------
# Step 3b: KEGG API — check obsolete EC codes
# ---------------------------------------------------------------------------
_ec_cache: dict[str, str | None] = {}  # ec_code → resolved ec_code or None


def resolve_ec(ec_code: str) -> str | None:
    """
    Check if an EC code is obsolete via KEGG API.
    Returns the current valid EC code, or None if truly obsolete.
    """
    if ec_code in _ec_cache:
        return _ec_cache[ec_code]

    body = _kegg_get(f"https://rest.kegg.jp/get/ec:{ec_code}")
    if body is None:
        # Not found at all
        _ec_cache[ec_code] = None
        return None

    # Check for "Transferred to" or "Obsolete"
    for line in body.splitlines():
        if "Transferred to" in line:
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
            if m:
                new_ec = m.group(1)
                print(f"         EC {ec_code} → Transferred to {new_ec}")
                _ec_cache[ec_code] = new_ec
                return new_ec
        if "Obsolete" in line.lower() and "Transferred" not in line:
            print(f"         EC {ec_code} → Obsolete (no replacement)")
            _ec_cache[ec_code] = None
            return None

    # Valid EC
    _ec_cache[ec_code] = ec_code
    return ec_code


# ---------------------------------------------------------------------------
# Step 4: Map genes → iPath3 IDs
# ---------------------------------------------------------------------------
def _try_ec_fallback(rxn: dict, gene: str, gtype: str, sources: str,
                     check_ec_online: bool) -> list[dict]:
    """
    Helper: attempt to produce EC-based iPath3 entries for a single reaction.
    Called when kegg.reaction entries are all absent or obsolete.
    """
    entries = []
    for ec in rxn["ec_codes"]:
        if check_ec_online:
            resolved = resolve_ec(ec)
        else:
            resolved = ec  # skip API check
        if resolved:
            entries.append({
                "ipath_id": f"E{resolved}",
                "gene_id": gene,
                "type": gtype,
                "sources": sources,
                "origin": "ec-code",
                "rxn_id": rxn["id"],
            })
    return entries


def map_genes_to_ipath_ids(df: pd.DataFrame, gene_rxn_map: dict,
                           check_ec_online: bool = True):
    """
    For each gene in df, find all reactions in the model, then:
      - priority 1: kegg.reaction → validate via KEGG API
            valid   → use R##### directly
            obsolete → fallback to ec-code for THIS reaction
      - priority 2: ec-code → resolve obsolete → E#.#.#.#
            (used when no kegg.reaction exists, OR all kegg.reactions obsolete)

    Returns list of dicts:
        {ipath_id, gene_id, type, sources, origin, rxn_id}
    """
    results = []
    unmapped_genes = []

    for _, row in df.iterrows():
        gene = row["gene_id"]
        gtype = row["type"]
        sources = row.get("sources", "")

        rxns = gene_rxn_map.get(gene, [])
        if not rxns:
            unmapped_genes.append(gene)
            continue

        gene_has_id = False
        for rxn in rxns:
            rxn_has_id = False

            # ----------------------------------------------------------------
            # Priority 1: kegg.reaction — validate each ID, fallback if obsolete
            # ----------------------------------------------------------------
            if rxn["kegg_reactions"]:
                valid_kr_found = False
                for kr in rxn["kegg_reactions"]:
                    # Normalize: ensure R##### format
                    rid = kr.strip()
                    if not rid.startswith("R"):
                        rid = "R" + rid

                    if check_ec_online:
                        resolved_rid = resolve_kegg_reaction(rid)
                    else:
                        resolved_rid = rid  # skip API check

                    if resolved_rid:
                        # Valid KEGG Reaction ID → use it
                        results.append({
                            "ipath_id": resolved_rid,
                            "gene_id": gene,
                            "type": gtype,
                            "sources": sources,
                            "origin": "kegg.reaction",
                            "rxn_id": rxn["id"],
                        })
                        rxn_has_id = True
                        valid_kr_found = True
                    # If resolved_rid is None, this particular R##### is obsolete;
                    # keep looping — maybe other R##### entries for this reaction are valid.

                # All kegg.reaction IDs for this reaction are obsolete →
                # fall back to ec-code for this reaction
                if not valid_kr_found:
                    print(f"         Reaction {rxn['id']}: all KEGG Reaction IDs obsolete, "
                          f"falling back to EC code")
                    fallback_entries = _try_ec_fallback(
                        rxn, gene, gtype, sources, check_ec_online
                    )
                    if fallback_entries:
                        results.extend(fallback_entries)
                        rxn_has_id = True

            # ----------------------------------------------------------------
            # Priority 2: ec-code only (no kegg.reaction annotation at all)
            # ----------------------------------------------------------------
            else:
                fallback_entries = _try_ec_fallback(
                    rxn, gene, gtype, sources, check_ec_online
                )
                if fallback_entries:
                    results.extend(fallback_entries)
                    rxn_has_id = True

            if rxn_has_id:
                gene_has_id = True

        if not gene_has_id:
            unmapped_genes.append(gene)

    print(f"[Step 4] Mapped: {len(results)} entries from "
          f"{len(df)} genes")
    if unmapped_genes:
        print(f"         Unmapped genes ({len(unmapped_genes)}): "
              f"{unmapped_genes}")
    return results, unmapped_genes


# ---------------------------------------------------------------------------
# Step 5: Determine line width from sources
# ---------------------------------------------------------------------------
def _get_width(sources: str) -> str:
    """
    ecBridge + ecFSEOF → W30
    ecBridge only      → W20
    ecFSEOF only       → W15
    unknown/other      → W20
    """
    s = str(sources).lower()
    has_bridge = "ecbridge" in s
    has_fseof = "ecfseof" in s
    if has_bridge and has_fseof:
        return "W30"
    elif has_bridge:
        return "W20"
    elif has_fseof:
        return "W15"
    return "W20"


# ---------------------------------------------------------------------------
# Step 6: Generate iPath3 output
# ---------------------------------------------------------------------------
def generate_ipath3(results: list, output_path: str):
    """
    Write iPath3 selection file.
    Format: ID COLOR WIDTH
    OE → #ff0000 (red), KD → #0000ff (blue)
    Width from sources column.
    """
    color_map = {"OE": "#ff0000", "KD": "#0000ff"}

    # Deduplicate: same ipath_id may appear from multiple reactions
    # If same ID has both OE and KD → mark as conflict (orange)
    id_info = {}  # ipath_id → {types: set, sources: str, genes: list}
    for r in results:
        iid = r["ipath_id"]
        if iid not in id_info:
            id_info[iid] = {
                "types": set(),
                "sources": r["sources"],
                "genes": [],
            }
        id_info[iid]["types"].add(r["type"])
        id_info[iid]["genes"].append(r["gene_id"])

    lines = []
    oe_count, kd_count, conflict_count = 0, 0, 0
    for iid, info in sorted(id_info.items()):
        types = info["types"]
        width = _get_width(info["sources"])
        gene_list = ", ".join(sorted(set(info["genes"])))

        if len(types) > 1:
            # Both OE and KD point to same ID → orange
            color = "#ff8800"
            conflict_count += 1

        elif "OE" in types:
            color = color_map["OE"]
            oe_count += 1

        else:
            color = color_map["KD"]
            kd_count += 1

        lines.append(f"{iid} {color} {width}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Step 6] iPath3 file saved: {output_path}")
    print(f"         OE (red): {oe_count},  KD (blue): {kd_count},  "
          f"Conflict (orange): {conflict_count}")
    print(f"         Total unique IDs: {len(lines)}")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate iPath3 input from filtered CSV + COBRA model JSON"
    )
    parser.add_argument("--filtered", required=True,
                        help="Path to singletest_filtered.csv")
    parser.add_argument("--model", required=True,
                        help="Path to COBRA model JSON")
    parser.add_argument("--output", default=None,
                        help="Output iPath3 txt file (default: auto-named)")
    parser.add_argument("--no-api", action="store_true",
                        help="Skip KEGG API check for obsolete EC codes AND "
                             "KEGG Reaction IDs")

    args = parser.parse_args()

    if args.output is None:
        base = args.filtered.replace("_singletest_filtered.csv", "")
        args.output = base + "_ipath3.txt"

    # Step 1
    df = read_filtered_csv(args.filtered)

    # Step 2
    gene_rxn_map = build_gene_reaction_map(args.model)

    # Step 3 & 4
    results, unmapped = map_genes_to_ipath_ids(
        df, gene_rxn_map, check_ec_online=not args.no_api
    )

    # Step 5 & 6
    lines = generate_ipath3(results, args.output)

    # Print preview
    print(f"\n{'='*60}")
    print("Preview (first 10 lines):")
    print("="*60)
    for line in lines[:10]:
        print(line)
    if len(lines) > 10:
        print(f"... ({len(lines) - 10} more lines)")


if __name__ == "__main__":
    main()