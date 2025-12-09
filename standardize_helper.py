import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import glob
import time
from tqdm import tqdm
import sys
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import warnings
import matplotlib.pyplot as plt
import csv
import pyproj
from datetime import datetime
import json
import os
import json
import numpy as np
import pandas as pd
import re

def parse_depth(val):
    """
    Parse a depth entry into:
      - (dep, NaN, NaN) if single value
      - (NaN, top, btm) if interval
      - (NaN, NaN, NaN) if NaN/invalid
    """
    if pd.isna(val):
        return (np.nan, np.nan, np.nan)

    s = str(val).lower().replace("cm", "").strip()

    # interval case
    if "-" in s:
        parts = s.split('-')
        if len(parts) == 2:
            try:
                top = float(parts[0].strip())
                btm = float(parts[1].strip())
                return (np.nan, top, btm)
            except ValueError:
                return (np.nan, np.nan, np.nan)
    else:
        # single value
        try:
            dep = float(s)
            return (dep, np.nan, np.nan)
        except ValueError:
            return (np.nan, np.nan, np.nan)


def attach_metadata(df, meta_file):
    # read in meta json
    if isinstance(meta_file, str): # meta_file is a file path
        with open(meta_file, 'r') as f:
            meta_json = json.load(f)
    elif isinstance(meta_file, dict): # meta_file is already a dictionary
        meta_json = meta_file
        
    # attach meta data
    for key in ["dataset_id", "subset_id", "time", "country", "depth", "lc_survey",]:
        val = meta_json.get(key)
        if val is not None:
            df[key] = val
            
    # parse depth column if exists
    if "depth" in df.columns:
        deps, tops, btms = zip(*df["depth"].map(parse_depth))
        df["hzn_dep"] = deps
        df["hzn_top"] = tops
        df["hzn_btm"] = btms
                
    # attach property info
    for prop, meta in meta_json.get("properties", {}).items():
        if prop == "soc":
            if meta.get("som?") == 1:
                df["soc_som"] = 1
        for suffix in ["unit", "method"]:
            value = meta.get(suffix)
            if value is not None:
                df[f"{prop}_{suffix}"] = value

    return df

def standardize_column_types(df):
    """
    Enforce standard types and formats for a harmonized soil dataset.

    - Props: convert to float
    - Meta string columns: force to string
    - 'site_id': force to string and remove trailing '.0', '.00'
    - Meta float columns: force to float
    - prop_unit and prop_method columns: force to string
    - Drop any unexpected columns
    - Return both cleaned df and any mismatches
    """
    import pandas as pd

    props = [
        'soc', 'total.n', 'carbonates',
        'ph.h2o', 'ph.cacl2',
        'clay', 'silt', 'sand',
        'extractable.k', 'extractable.p',
        'cec', 'ec',
        'bd.fe', 'bd.tot',
        'cf.mass', 'cf.vol',
        'soc_som'
    ]

    meta_str = ['dataset_id', 'subset_id', 'site_id', 'country', 'lc_survey', 'time']
    meta_float = ['hzn_top', 'hzn_btm', 'hzn_dep', 'lon', 'lat']

    # Generate prop_info columns (unit/method for each prop)
    prop_info = [f"{p}_{suffix}" for p in props for suffix in ['unit', 'method']]
    allowed_cols = set(props + meta_str + meta_float + prop_info)

    mismatches = {"props": [], "meta_str": [], "meta_float": [], "prop_info": []}

    df = df.copy()

    # 1. Props → float
    dfot = df.copy() 
    for col in props: 
        if col in df.columns: 
            if df[col].dtype=='object': 
                converted = pd.to_numeric(dfot[col], errors='coerce') 
                bad_values = dfot.loc[converted.isna(), col].dropna().unique() 
                if len(bad_values) > 0: 
                    print(col, bad_values) 
                    mismatches["props"].append(col) 
                else:# df[col].dtype=='int64': 
                    df[col] = df[col].astype(float)

    # 2. Meta string columns
    for col in meta_str:
        if col in df.columns:
            if col == 'site_id':
                def normalize_site_id(val):
                    try:
                        if pd.notna(val):
                            fval = float(val)
                            if fval.is_integer():
                                return str(int(fval))
                        return str(val).strip()
                    except:
                        return str(val).strip()
                        
                df[col] = df[col].apply(normalize_site_id)
            else:
                try:
                    df[col] = df[col].astype(str)
                except Exception:
                    mismatches["meta_str"].append(col)
        else:
            print(f"Missing meta column: {col}, double check if you could add it...")

    # 3. Meta float columns
    for col in meta_float:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                mismatches["meta_float"].append(col)

    # 4. Property info columns → string
    for col in prop_info:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str)
            except Exception:
                mismatches["prop_info"].append(col)

    # 5. Drop unexpected columns
    unexpected = set(df.columns) - allowed_cols
    if unexpected:
        print(f"Dropping unexpected columns: {sorted(unexpected)}")
        df = df.drop(columns=list(unexpected))

    return df, mismatches

import re
import numpy as np

def extract_year(val):
    """
    Extract year from mixed formats:
      - DD[-.]MM[-.](YY|YYYY) (year-last)-> take the last (year-last)     e.g. '25.08.2016', '30-07-17'
      - YYYY.MM, YYYY.0, YYYY            -> take the first (year-first)   e.g. '2021.05'
    If multiple distinct years are found across tokens, returns NaN.
    """
    if pd.isna(val):
        return np.nan

    s = str(val).strip()
    if not s:
        return np.nan

    # commas/semicolons/whitespace to handle multiple years "2021.05 30-07-17"
    token = re.split(r"[,\s;/]+", s)
    if len(token)>1:
        return np.nan
    if not token[0]:
        return np.nan
            
    p = token[0].strip().strip("'\"")  # trim quotes if present
    # print(p)
    
    # extract 4-digit year 
    m = re.search(r'\b(19|20)\d{2}\b', p)
    if m:
        y = int(m.group(0))
        return y

    # DD[-.]MM[-.](YY) (year-last)
    m = re.match(r'^\.*(\d{2})[.-]+(\d{2})[.-]+(\d{2})\.?$', p)
    if m:
        y = int(m.group(3))
        if y < 25:
            y = y+2000
        else:
            y = y+1900
        
        return y
        
    # YYYY.MM, YYYY.0, YYYY
    # print("year")
    try:
        num = float(p)
        y = int(num)
        if 1900 <= y <= 2025:
            return y
    except ValueError:
        pass

    return np.nan