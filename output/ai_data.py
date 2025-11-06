#!/usr/bin/env python3
"""
synthea_to_csv_chunked.py

Faster, chunked converter from Synthea CSVs -> per-patient CSV.
If --debug_limit N > 0, only processes the first N patients (fast).
"""

import os
import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ---------- Helpers ----------
def lowercase_cols(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def detect_id_column_from_columns(cols):
    candidates = ['patient', 'patient_id', 'patientid', 'id']
    for c in candidates:
        if c in cols:
            return c
    # fallback to first col
    return cols[0] if cols else None

def get_csv_columns(path):
    # peek at header only
    try:
        df = pd.read_csv(path, nrows=0, dtype=str, encoding="utf-8-sig", low_memory=False)
    except Exception:
        df = pd.read_csv(path, nrows=0, dtype=str, low_memory=False)
    return [c.strip().lower() for c in df.columns]

def parse_timestamp_single(v):
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v, utc=True, errors='coerce', infer_datetime_format=True)
        if pd.isna(ts):
            return None
        return ts.isoformat()
    except Exception:
        return None

def pick_timestamp_for_row(row, ts_candidates):
    # ts_candidates is a list of lowercased column names
    for c in ts_candidates:
        if c in row and row[c] not in (None, ""):
            ts = parse_timestamp_single(row[c])
            if ts:
                return ts
    # fallback: look for any column with date/time substring
    for k, v in row.items():
        if v and ('date' in k or 'time' in k or 'timestamp' in k):
            ts = parse_timestamp_single(v)
            if ts:
                return ts
    return None

def detect_timestamp_candidates(cols):
    # return prioritized list of columns that probably contain timestamps
    candidates = []
    keys = ['start','stop','date','time','effective','issued','authored','recorded','period','timestamp','datetime']
    for key in keys:
        for c in cols:
            if key in c and c not in candidates:
                candidates.append(c)
    # fallback: any column with date/time in name
    for c in cols:
        if ('date' in c or 'time' in c or 'timestamp' in c or 'datetime' in c) and c not in candidates:
            candidates.append(c)
    return candidates

def build_event_from_record(rec, category, ts_candidates, source_name, pid_col):
    # rec is a dict (lowercased keys)
    ts = pick_timestamp_for_row(rec, ts_candidates)
    subtype = None
    for key in ('description','code','type','name','category','value','status','reason'):
        if key in rec and rec[key]:
            subtype = rec[key]
            break
    data = {}
    for k, v in rec.items():
        if k == pid_col or v is None:
            continue
        # skip timestamp-like columns from data inclusion
        if any(tok in k for tok in ('start','stop','date','time','issued','authored','recorded','period','timestamp','datetime')):
            continue
        data[k] = v
    return {
        "timestamp": ts,
        "category": category,
        "subtype": subtype if subtype else source_name,
        "data": data,
        "meta": {"source": source_name}
    }

# ---------- Main conversion (chunked) ----------
def convert_chunked(input_dir, output_path, debug_limit=0, chunksize=200_000, anchors='latest', label_window_days=90):
    input_dir = str(Path(input_dir))
    print("Input folder:", input_dir)

    # Find CSVs (only the ones we commonly need)
    files = {
        "patients": "patients.csv",
        "encounters": "encounters.csv",
        "conditions": "conditions.csv",
        "observations": "observations.csv",
        "medications": "medications.csv",
        "procedures": "procedures.csv",
        "immunizations": "immunizations.csv",
        "devices": "devices.csv"
    }

    # Validate presence of patients.csv
    patients_path = os.path.join(input_dir, files["patients"])
    if not os.path.exists(patients_path):
        raise FileNotFoundError(f"patients.csv not found under {input_dir}")

    # Read patients list (only headers + N rows if debug_limit > 0)
    if debug_limit and debug_limit > 0:
        patients_df = pd.read_csv(patients_path, dtype=str, encoding="utf-8-sig", nrows=debug_limit, low_memory=False)
        # But also ensure we know the PID column for mapping later; get header separately
        all_patients_cols = get_csv_columns(patients_path)
    else:
        patients_df = pd.read_csv(patients_path, dtype=str, encoding="utf-8-sig", low_memory=False)
        all_patients_cols = get_csv_columns(patients_path)

    patients_df = lowercase_cols(patients_df)
    patients_df = patients_df.where(pd.notnull(patients_df), None)
    pid_col = detect_id_column_from_columns(list(patients_df.columns))
    print(f"Detected patient-id column: '{pid_col}' (patients)")

    # Build list / set of patient ids we will process
    patients_list = patients_df[pid_col].tolist()
    if not patients_list:
        raise RuntimeError("No patient ids found in patients.csv")

    pid_set = set(patients_list)
    print(f"Processing {len(pid_set)} patients (debug_limit={debug_limit})")

    # For each other large CSV, read in chunks and keep only rows where patient id is in pid_set
    events_by_patient = defaultdict(list)
    outcome_by_patient = defaultdict(list)

    for key, fname in files.items():
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            print(f"Skipping {fname}: not found.")
            continue
        if key == "patients":
            # we already loaded patient records, skip adding as event table
            continue

        print(f"Scanning {fname} in chunks for matching patients...")
        # detect columns and pid column name in this file
        cols = get_csv_columns(path)
        pid_col_table = detect_id_column_from_columns(cols)
        ts_candidates = detect_timestamp_candidates(cols)
        category = key
        source_name = fname

        # chunked reader
        reader = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False, chunksize=chunksize)
        chunk_count = 0
        matched_rows = 0
        for chunk in reader:
            chunk_count += 1
            chunk = lowercase_cols(chunk)
            chunk = chunk.where(pd.notnull(chunk), None)
            # filter rows where pid column matches our pid_set
            if pid_col_table in chunk.columns:
                # use .isin on series (fast)
                filtered = chunk[chunk[pid_col_table].isin(pid_set)]
            else:
                # try fallback columns
                possible = [c for c in chunk.columns if c in ('patient','id','patient_id','patientid')]
                if possible:
                    col0 = possible[0]
                    filtered = chunk[chunk[col0].isin(pid_set)]
                else:
                    # no patient column, skip chunk
                    filtered = pd.DataFrame(columns=chunk.columns)

            if filtered.empty:
                # continue; still keep scanning other chunks
                continue

            matched_rows += len(filtered)
            # convert filtered rows to dicts and build events
            for rec in filtered.to_dict(orient="records"):
                # determine pid robustly
                pid_val = rec.get(pid_col_table) or rec.get('patient') or rec.get('id') or rec.get('patient_id')
                if not pid_val:
                    continue
                event = build_event_from_record(rec, category, ts_candidates, source_name, pid_col_table)
                events_by_patient[pid_val].append(event)

            # optional small progress print
            if chunk_count % 5 == 0:
                print(f"  processed {chunk_count} chunks, matched_rows so far: {matched_rows}")

        print(f"  done scanning {fname}: matched_rows={matched_rows}")

    # Build and write per-patient CSV
    print("Writing per-patient CSV...")
    written = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        # iterate over originally captured patients (in order)
        for i, p_row in enumerate(patients_df.to_dict(orient="records")):
            if debug_limit and i >= debug_limit:
                break
            pid = p_row.get(pid_col)
            if not pid:
                continue
            block = {"patient_id": pid, "demographics": {}, "events": [], "anchors": []}
            # copy basic demographics from patients file
            for fld in ['first','last','birthdate','gender','race','ethnicity','city','state','zip','income','healthcare_expenses','healthcare_coverage']:
                if fld in p_row:
                    block['demographics'][fld] = p_row[fld] if p_row[fld] not in (None, "") else None
            # attach events (sorted)
            evs = events_by_patient.get(pid, [])
            # sort by timestamp (None or unparsable timestamps go last)
            def sk(e):
                t = e.get('timestamp')
                if not t:
                    return pd.Timestamp.max
                try:
                    return pd.to_datetime(t, utc=True)
                except Exception:
                    return pd.Timestamp.max
            evs_sorted = sorted(evs, key=sk)
            block['events'] = evs_sorted

            fout.write(json.dumps(block, default=str) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"  wrote {written} patients...")

    print(f"Finished. Wrote {written} patient blocks to {output_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunked Synthea CSV -> CSV (debug_limit helps).")
    parser.add_argument("--input_dir", required=True, help="Synthea CSV folder")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--debug_limit", type=int, default=0, help="If >0, only process the first N patients (fast)")
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunk size for scanning big files")
    parser.add_argument("--label_window", type=int, default=90, help="label window days for anchor")
    args = parser.parse_args()

    convert_chunked(args.input_dir, args.output, debug_limit=args.debug_limit, chunksize=args.chunksize, label_window_days=args.label_window)
