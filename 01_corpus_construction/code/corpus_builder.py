#!/usr/bin/env python3
"""
Download TCP files listed in a CSV column from GitHub repos like:
  https://github.com/textcreationpartnership/A00723

Edits the CSV by adding/updating a status column (default: "downloaded")
with values: "downloaded" or "not downloaded".

CONFIGURE THESE VARIABLES BELOW (not via command line):
- CSV_PATH
- TCP_COLUMN
- STATUS_COLUMN
- GITHUB_OWNER
- OUTPUT_DIR
- OVERWRITE
- LIMIT (optional testing)
- GITHUB_TOKEN (optional)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

# =========================
# USER SETTINGS (EDIT HERE)
# =========================
CSV_PATH = Path("/Users/grace/Desktop/new meta/deep_tcp.csv")
TCP_COLUMN = "TCP"                           # column containing IDs like A00723
STATUS_COLUMN = "included"                 # new/updated column written back to CSV
GITHUB_OWNER = "textcreationpartnership"     # GitHub org/user
OUTPUT_DIR = Path("tcp_drama")               # where to save .xml files
OVERWRITE = False                            # overwrite existing local files?
LIMIT = 0                                    # 0 = no limit; else only first N (testing)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # optional: avoid rate limits
TIMEOUT_SECONDS = 10
# =========================


def sanitize_tcp_id(value) -> Optional[str]:
    """Extract TCP id like A00723 from a cell, URL, or mixed string."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "na", "<na>"}:
        return None

    # URL -> repo name
    m = re.search(r"/([A-Z]\d{5})(?:/)?$", s)
    if m:
        return m.group(1)

    # plain id
    m = re.fullmatch(r"([A-Z]\d{5})", s)
    if m:
        return m.group(1)

    # embedded id
    m = re.search(r"\b([A-Z]\d{5})\b", s)
    if m:
        return m.group(1)

    return None


def try_raw_download(session: requests.Session, owner: str, repo: str, branch: str, filename: str) -> Optional[bytes]:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
    r = session.get(raw_url, timeout=TIMEOUT_SECONDS)
    if r.status_code == 200 and r.content:
        return r.content
    return None


def find_xml_via_api(session: requests.Session, owner: str, repo: str) -> Optional[Tuple[str, bytes]]:
    """
    GitHub API fallback: list root contents and download an XML file.
    Prefers "{repo}.xml" if present, else first .xml in root.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    r = session.get(api_url, timeout=TIMEOUT_SECONDS)
    if r.status_code != 200:
        return None

    try:
        items = r.json()
    except Exception:
        return None

    xml_files = [
        it for it in items
        if isinstance(it, dict)
        and it.get("type") == "file"
        and str(it.get("name", "")).lower().endswith(".xml")
    ]
    if not xml_files:
        return None

    chosen = None
    for it in xml_files:
        if it.get("name") == f"{repo}.xml":
            chosen = it
            break
    if chosen is None:
        chosen = xml_files[0]

    name = chosen.get("name")
    download_url = chosen.get("download_url")
    if not name or not download_url:
        return None

    r2 = session.get(download_url, timeout=TIMEOUT_SECONDS)
    if r2.status_code == 200 and r2.content:
        return name, r2.content
    return None


def download_repo_xml(session: requests.Session, owner: str, tcp_id: str) -> Optional[Tuple[str, bytes]]:
    """
    Download "{tcp_id}.xml" from repo {owner}/{tcp_id}.
    Tries branches master/main; falls back to GitHub API root listing.
    Returns (filename, content) or None.
    """
    filename = f"{tcp_id}.xml"

    for branch in ("master", "main"):
        content = try_raw_download(session, owner, tcp_id, branch, filename)
        if content is not None:
            return filename, content

    return find_xml_via_api(session, owner, tcp_id)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if TCP_COLUMN not in df.columns:
        raise ValueError(f"Column '{TCP_COLUMN}' not found. Available: {list(df.columns)}")

    # Ensure status column exists
    if STATUS_COLUMN not in df.columns:
        df[STATUS_COLUMN] = ""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "tcp-csv-downloader/1.0",
        "Accept": "application/vnd.github+json",
    })
    if GITHUB_TOKEN:
        session.headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    # Build list of (row_index, tcp_id)
    targets = []
    for idx, val in df[TCP_COLUMN].items():
        tcp_id = sanitize_tcp_id(val)
        if tcp_id:
            targets.append((idx, tcp_id))
        else:
            # If it's blank/invalid, mark as not downloaded
            df.at[idx, STATUS_COLUMN] = "not downloaded"

    if LIMIT and LIMIT > 0:
        targets = targets[:LIMIT]

    ok = skip = fail = 0

    for n, (idx, tcp_id) in enumerate(targets, start=1):
        # If we got a weird duplicate id in many rows, we still mark per-row status.
        default_out = OUTPUT_DIR / f"{tcp_id}.xml"

        if default_out.exists() and not OVERWRITE:
            df.at[idx, STATUS_COLUMN] = "downloaded"
            print(f"[{n}/{len(targets)}] SKIP (exists) {tcp_id} -> {default_out.name}")
            skip += 1
            continue

        try:
            result = download_repo_xml(session, GITHUB_OWNER, tcp_id)
            if result is None:
                df.at[idx, STATUS_COLUMN] = "not downloaded"
                print(f"[{n}/{len(targets)}] FAIL {tcp_id}: XML not found")
                fail += 1
                continue

            fname, content = result
            out_path = OUTPUT_DIR / fname

            if out_path.exists() and not OVERWRITE:
                df.at[idx, STATUS_COLUMN] = "downloaded"
                print(f"[{n}/{len(targets)}] SKIP (exists) {tcp_id} -> {out_path.name}")
                skip += 1
                continue

            out_path.write_bytes(content)
            df.at[idx, STATUS_COLUMN] = "downloaded"
            print(f"[{n}/{len(targets)}] OK   {tcp_id} -> {out_path.name}")
            ok += 1

        except requests.RequestException as e:
            df.at[idx, STATUS_COLUMN] = "not downloaded"
            print(f"[{n}/{len(targets)}] FAIL {tcp_id}: network error: {e}")
            fail += 1
        except Exception as e:
            df.at[idx, STATUS_COLUMN] = "not downloaded"
            print(f"[{n}/{len(targets)}] FAIL {tcp_id}: error: {e}")
            fail += 1

    # Write updated CSV back (keeps same filename)
    df.to_csv(CSV_PATH, index=False)

    print("\n=== Summary ===")
    print(f"CSV updated: {CSV_PATH}")
    print(f"Output dir:  {OUTPUT_DIR.resolve()}")
    print(f"Downloaded:  {ok}")
    print(f"Skipped:     {skip}")
    print(f"Failed:      {fail}")


if __name__ == "__main__":
    main()