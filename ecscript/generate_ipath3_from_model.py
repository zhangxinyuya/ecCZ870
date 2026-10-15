#!/usr/bin/env python
# Author: Zhang xinyu
# Date: 2026-10-15
"""Generate iPath3 selections from filtered gene targets and model annotations.

Prefer KEGG reaction IDs; fall back to EC codes. Online checks are optional.
Output: ID COLOR WIDTH. OE is red, KD blue, and conflicts orange."""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict

import pandas as pd


def read_filtered_csv(path: str) -> pd.DataFrame:
    """Read gene targets and retain OE/KD rows."""
    df = pd.read_csv(path)
    required = {"gene_id", "type"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")

    df = df[df["type"].isin(["OE", "KD"])].copy()
    df["type"] = df["type"].str.strip().str.upper()
    print(f"[Step 1] Loaded {len(df)} targets from {path}")
    print(f"         OE: {(df['type']=='OE').sum()},  KD: {(df['type']=='KD').sum()}")
    return df


def build_gene_reaction_map(model_path: str):
    """Map genes to reaction IDs, EC codes, and KEGG reaction annotations."""
    with open(model_path, "r") as f:
        model = json.load(f)

    reactions = model.get("reactions", [])
    gene_rxn_map = defaultdict(list)

    for rxn in reactions:
        rxn_id = rxn.get("id", "")
        annotation = rxn.get("annotation", {})


        ec_raw = annotation.get("ec-code", [])
        if isinstance(ec_raw, str):
            ec_raw = [ec_raw]
        ec_codes = [e.strip() for e in ec_raw if e.strip()]


        kr_raw = annotation.get("kegg.reaction", [])
        if isinstance(kr_raw, str):
            kr_raw = [kr_raw]
        kegg_reactions = [r.strip() for r in kr_raw if r.strip()]


        rule = rxn.get("gene_reaction_rule", "")
        genes_in_rule = set(re.findall(r"[A-Za-z0-9_]+", rule)) if rule else set()

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


def _kegg_get(url: str, retries: int = 3) -> str | None:
    """Fetch a KEGG entry with retries; return None on failure."""
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


_kr_cache: dict[str, str | None] = {}


def resolve_kegg_reaction(rid: str) -> str | None:
    """Return a KEGG reaction ID with an ENTRY response, or None."""
    if rid in _kr_cache:
        return _kr_cache[rid]

    body = _kegg_get(f"https://rest.kegg.jp/get/{rid}")
    if body is None:

        print(f"         KEGG Reaction {rid} → Not found / obsolete, will fallback to EC")
        _kr_cache[rid] = None
        return None


    if body.strip().startswith("ENTRY"):
        _kr_cache[rid] = rid
        return rid


    print(f"         KEGG Reaction {rid} → Unexpected response format, will fallback to EC")
    _kr_cache[rid] = None
    return None


_ec_cache: dict[str, str | None] = {}


def resolve_ec(ec_code: str) -> str | None:
    """Resolve EC transfers and cache the resulting code or None."""
    if ec_code in _ec_cache:
        return _ec_cache[ec_code]

    body = _kegg_get(f"https://rest.kegg.jp/get/ec:{ec_code}")
    if body is None:

        _ec_cache[ec_code] = None
        return None


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


    _ec_cache[ec_code] = ec_code
    return ec_code


def _try_ec_fallback(rxn: dict, gene: str, gtype: str, sources: str,
                     check_ec_online: bool) -> list[dict]:
    """Create EC-based entries, optionally checking codes online."""
    entries = []
    for ec in rxn["ec_codes"]:
        if check_ec_online:
            resolved = resolve_ec(ec)
        else:
            resolved = ec
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
    """Map genes using KEGG reaction IDs, falling back to EC codes per reaction.

    Return mapped entries and unmapped genes."""
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


            if rxn["kegg_reactions"]:
                valid_kr_found = False
                for kr in rxn["kegg_reactions"]:

                    rid = kr.strip()
                    if not rid.startswith("R"):
                        rid = "R" + rid

                    if check_ec_online:
                        resolved_rid = resolve_kegg_reaction(rid)
                    else:
                        resolved_rid = rid

                    if resolved_rid:

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


                if not valid_kr_found:
                    print(f"         Reaction {rxn['id']}: all KEGG Reaction IDs obsolete, "
                          f"falling back to EC code")
                    fallback_entries = _try_ec_fallback(
                        rxn, gene, gtype, sources, check_ec_online
                    )
                    if fallback_entries:
                        results.extend(fallback_entries)
                        rxn_has_id = True


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


def _get_width(sources: str) -> str:
    """Return W30 for both tools, W15 for ecFSEOF only, otherwise W20."""
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


def generate_ipath3(results: list, output_path: str):
    """Write unique ID COLOR WIDTH selections: OE red, KD blue, conflicts orange.

    Width uses the first source label recorded for each ID."""
    color_map = {"OE": "#ff0000", "KD": "#0000ff"}


    id_info = {}
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


    df = read_filtered_csv(args.filtered)


    gene_rxn_map = build_gene_reaction_map(args.model)


    results, unmapped = map_genes_to_ipath_ids(
        df, gene_rxn_map, check_ec_online=not args.no_api
    )


    lines = generate_ipath3(results, args.output)


    print(f"\n{'='*60}")
    print("Preview (first 10 lines):")
    print("="*60)
    for line in lines[:10]:
        print(line)
    if len(lines) > 10:
        print(f"... ({len(lines) - 10} more lines)")


if __name__ == "__main__":
    main()
