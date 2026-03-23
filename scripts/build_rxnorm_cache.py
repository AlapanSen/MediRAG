"""
FR-20: build_rxnorm_cache.py — Offline Drug Name Normalisation Cache Builder
=============================================================================
Accepts EITHER:
  A) DrugBank vocabulary CSV  (--drugbank-csv)  ← recommended, immediate
  B) DrugBank Open Data XML   (--drugbank-xml)  ← requires registration at drugbank.com

DrugBank vocabulary CSV is freely downloadable (no account needed) from:
  https://go.drugbank.com/releases/latest#open-data  →  "DrugBank Vocabulary"

Queries RxNorm REST API (single approximateTerm call per drug) and saves
results to data/rxnorm_cache.csv.

Runtime:
    ~14,000 names × 0.1s delay × 1 API call ≈ 24 minutes

Usage:
    python scripts/build_rxnorm_cache.py --drugbank-csv "data/drugbank vocabulary.csv"
    python scripts/build_rxnorm_cache.py --drugbank-csv "data/drugbank vocabulary.csv" --dry-run 50
    python scripts/build_rxnorm_cache.py --drugbank-xml data/raw/drugbank_open_data.xml
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_rxnorm_cache")

# RxNorm approximateTerm endpoint — returns rxcui + name in ONE call (v1.4 fix)
RXNORM_APPROX_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"

# DrugBank Open Data XML namespace (XML path only)
NS = {"db": "http://www.drugbank.ca"}


# ---------------------------------------------------------------------------
# Step 1A: Extract drug names from DrugBank Vocabulary CSV  ← preferred
# ---------------------------------------------------------------------------

def extract_drug_names_from_csv(csv_path: str) -> list[str]:
    """
    Parse the DrugBank vocabulary CSV and return all drug name strings.

    CSV columns: DrugBank ID | Accession Numbers | Common name | CAS | UNII
                 | Synonyms | Standard InChI Key

    Synonyms column is pipe-separated (e.g. "Drug A | Alias B | Trade Name C").

    Args:
        csv_path : path to the DrugBank vocabulary CSV file

    Returns:
        Sorted deduplicated list of drug name strings.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.error(
            "DrugBank vocabulary CSV not found at '%s'. "
            "Download it from https://go.drugbank.com/releases/latest#open-data "
            "(look for 'DrugBank Vocabulary' — no account needed).",
            csv_path,
        )
        sys.exit(1)

    logger.info("Parsing DrugBank vocabulary CSV: %s", path)
    names: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Common name
            common = row.get("Common name", "").strip()
            if common:
                names.add(common)

            # Pipe-separated synonyms
            synonyms_raw = row.get("Synonyms", "")
            if synonyms_raw:
                for syn in synonyms_raw.split("|"):
                    syn = syn.strip()
                    if syn:
                        names.add(syn)

    result = sorted(names)
    logger.info("Extracted %d unique drug names/synonyms from CSV", len(result))
    return result


# ---------------------------------------------------------------------------
# Step 1B: Extract drug names from DrugBank Open Data XML  ← needs account
# ---------------------------------------------------------------------------

def extract_drug_names_from_xml(xml_path: str) -> list[str]:
    """
    Parse DrugBank Open Data XML and extract all drug names + synonyms.

    Args:
        xml_path : Path to drugbank_open_data.xml

    Returns:
        Sorted deduplicated list of drug name strings.
    """
    logger.info("Parsing DrugBank XML: %s", xml_path)
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        logger.error(
            "DrugBank XML not found at '%s'. "
            "Download it from https://go.drugbank.com/releases/latest#open-data "
            "(free academic registration required), or use --drugbank-csv instead.",
            xml_path,
        )
        sys.exit(1)
    except ET.ParseError as exc:
        logger.error("Failed to parse DrugBank XML: %s", exc)
        sys.exit(1)

    root = tree.getroot()
    names: set[str] = set()

    for drug in root.findall("db:drug", NS):
        name_el = drug.find("db:name", NS)
        if name_el is not None and name_el.text:
            names.add(name_el.text.strip())
        for syn in drug.findall("db:synonyms/db:synonym", NS):
            if syn.text:
                names.add(syn.text.strip())
        for brand in drug.findall(
            "db:international-brands/db:international-brand/db:name", NS
        ):
            if brand.text:
                names.add(brand.text.strip())

    result = sorted(names)
    logger.info("Extracted %d unique drug names/synonyms from XML", len(result))
    return result



# ---------------------------------------------------------------------------
# Step 2: Query RxNorm (single API call per drug — v1.4)
# ---------------------------------------------------------------------------

def query_rxnorm(drug_name: str, timeout: int = 5) -> tuple[str, str]:
    """
    Look up a drug name in RxNorm using approximateTerm endpoint.
    Returns (rxcui, canonical_name).  Returns ("", "") on any failure.

    Uses /approximateTerm — single HTTP call returning both rxcui and name.
    (Previous 2-call approach was replaced in v1.4, cutting runtime by ~50%.)
    """
    try:
        resp = requests.get(
            RXNORM_APPROX_URL,
            params={"term": drug_name, "maxEntries": "1", "option": "1"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return "", ""

        candidates: list[dict] = (
            resp.json()
            .get("approximateGroup", {})
            .get("candidate", [])
        )
        if not candidates:
            return "", ""

        rxcui = candidates[0].get("rxcui", "")
        name  = candidates[0].get("name", drug_name)   # fallback to input
        return rxcui, name

    except Exception:
        return "", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline RxNorm cache from DrugBank data (FR-20)"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--drugbank-csv",
        metavar="PATH",
        default=None,
        help=(
            "Path to DrugBank vocabulary CSV  [RECOMMENDED — no account needed]. "
            "Download from https://go.drugbank.com/releases/latest#open-data"
        ),
    )
    source.add_argument(
        "--drugbank-xml",
        metavar="PATH",
        default=None,
        help="Path to DrugBank Open Data XML (requires free academic registration).",
    )
    parser.add_argument(
        "--output-csv",
        default="data/rxnorm_cache.csv",
        help="Path for output CSV",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds to wait between API calls (default 0.1 — ~24 min total)",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        metavar="N",
        help="Only process first N drug names (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previously interrupted run. Reads already-completed entries "
            "from --output-csv and skips them, appending only the missing ones."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Auto-detect source if neither flag was given
    # ------------------------------------------------------------------
    csv_default = "data/drugbank vocabulary.csv"
    xml_default = "data/raw/drugbank_open_data.xml"

    if args.drugbank_csv:
        drug_names = extract_drug_names_from_csv(args.drugbank_csv)
    elif args.drugbank_xml:
        drug_names = extract_drug_names_from_xml(args.drugbank_xml)
    elif Path(csv_default).exists():
        logger.info("Auto-detected DrugBank vocabulary CSV at '%s'", csv_default)
        drug_names = extract_drug_names_from_csv(csv_default)
    elif Path(xml_default).exists():
        logger.info("Auto-detected DrugBank XML at '%s'", xml_default)
        drug_names = extract_drug_names_from_xml(xml_default)
    else:
        logger.error(
            "No DrugBank source found. Pass --drugbank-csv or --drugbank-xml. "
            "See script docstring for download links."
        )
        sys.exit(1)

    if args.dry_run > 0:
        drug_names = drug_names[: args.dry_run]
        logger.info("Dry-run mode: processing %d names only", len(drug_names))

    # ------------------------------------------------------------------
    # Resume: skip names already in the output CSV
    # ------------------------------------------------------------------
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    if args.resume and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("drug_name", "").strip()
                    if name:
                        already_done.add(name)
            logger.info(
                "Resume mode: %d entries already in cache — skipping these.",
                len(already_done),
            )
        except Exception as exc:
            logger.warning("Could not read existing cache for resume: %s", exc)
            already_done = set()

    remaining = [n for n in drug_names if n not in already_done]
    skipped = len(drug_names) - len(remaining)
    if skipped:
        logger.info("Skipping %d already-resolved names. %d remaining.", skipped, len(remaining))

    total = len(remaining)
    if total == 0:
        logger.info("Nothing to do — cache is already complete.")
        sys.exit(0)

    est_minutes = total * (args.delay + 0.05) / 60
    logger.info(
        "Starting cache build: %d names to process, delay=%.2fs, estimated %.0f minutes",
        total, args.delay, est_minutes,
    )

    # ------------------------------------------------------------------
    # Write CSV — append if resuming, overwrite otherwise
    # ------------------------------------------------------------------
    file_mode = "a" if args.resume and out_path.exists() and already_done else "w"
    write_header = file_mode == "w"

    found = len(already_done)  # count previously resolved entries too
    new_found = 0

    with open(out_path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["drug_name", "rxcui", "canonical_name"])

        for i, name in enumerate(remaining):
            rxcui, canonical = query_rxnorm(name)
            writer.writerow([name, rxcui, canonical])
            if rxcui:
                new_found += 1
                found += 1

            if i % 25 == 0 or i == total - 1:
                pct = 100 * (i + 1) / total
                logger.info(
                    "Progress: %d/%d (%.1f%%) — %d resolved this run (%d total)",
                    i + 1, total, pct, new_found, found,
                )

            time.sleep(args.delay)

    logger.info(
        "Cache saved to %s — %d/%d names resolved to RxNorm IDs (this run: +%d)",
        out_path, found, len(drug_names), new_found,
    )
    logger.info(
        "Commit this file to the repo: git add %s && git commit -m 'Add RxNorm cache'",
        out_path,
    )


if __name__ == "__main__":
    main()
