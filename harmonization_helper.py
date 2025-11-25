# distribution_checker.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from scipy.stats import ks_2samp, mannwhitneyu, entropy
from shapely.geometry import Point
input_path = '/home/xuemeng/work_xuemeng/ai4sh_data.harmo/raw_data'  
output_path = '/home/xuemeng/work_xuemeng/ai4sh_data.harmo/data_v2'

# # create EU boundary file
# eu_countries = gpd.read_file(f'{input_path}/EU/EU_nuts/NUTS_RG_20M_2021_3035.shp')
# if eu_countries.crs != 'EPSG:4326':
#     eu_countries = eu_countries.to_crs('EPSG:4326')
# eu_boundary = eu_countries.dissolve()
# eu_boundary = eu_boundary.drop(columns=['NUTS_ID', 'LEVL_CODE', 'CNTR_CODE', 'NAME_LATN','NUTS_NAME', 'MOUNT_TYPE', 'URBN_TYPE', 'COAST_TYPE', 'FID'])
# eu_boundary['description'] = 'EU boundary'

import time
import random
import gspread
from gspread.exceptions import APIError

def register_methods(sheet, df, tab, prop, version):
    """
    sheet: gspread Worksheet
    df:    standardized pandas DataFrame (has 'dataset_id' and f'{prop}_method')
    tab:   pandas DataFrame of current registry (B:D), with columns ['src','method','version'] (recommended)
    prop:  property key, e.g. 'ph', 'soc'  -> uses df[f'{prop}_method']
    version: string to write in column D
    chunk_size: how many rows per write (tune if you still hit quota)
    """
    col = f"{prop}_method"

    # 1) existing pairs in the registry (B=src, C=method)
    existing_pairs = set(zip(tab["src"], tab["method"]))

    # 2) candidate pairs from df -> unique & sorted
    pairs = (
        df[["dataset_id", col]]
        .dropna()
        .drop_duplicates()
        .sort_values(["dataset_id", col])
        .to_numpy()
    )

#     # 3) first remove outdated ones
#     outdated_pairs = existing_pairs - set(tuple(p) for p in pairs)
#     if outdated_pairs:
#         print('Removing outdated entries:')
#         colA = sheet.col_values(1)[1:]  # skip header
#         colB = sheet.col_values(2)[1:]
#         row_map = {(a, b): i+2 for i, (a, b) in enumerate(zip(colA, colB))}
#         rows_to_del = []
#         for p in outdated_pairs:
#             if p in row_map:
#                 rows_to_del.append(row_map[p])
#                 print('Removing ', p)
                
#         rows_to_del = sorted(rows_to_del, reverse=True)
#         for r in rows_to_del:
#             sheet.delete_rows(r)
            
#     else:
#         print('No outdated entries to be removed')
 
    # 4) only new ones and shape as [A,B,C], add new ones
    new_pairs = [tuple(p) for p in pairs if tuple(p) not in existing_pairs]
    counts = (
        df.groupby(['dataset_id', col])
          .size()
          .rename('count')
    )
    rows_to_append = [
        [src, method, version, int(counts.get((src, method), 0))]
        for src, method in new_pairs
    ]
    
    if not rows_to_append:
        print("No new methods to register.")
        return None

    # 5) first empty row in column B
    colB_vals = sheet.col_values(2)  
    start_row = len(colB_vals) + 1
    end_row = start_row + len(rows_to_append) - 1
    range_str = f"A{start_row}:D{end_row}"

    # 6) batch, safe write with mild retry
    for attempt in range(2):
        try:
            # sheet.update(range_str, rows_to_append, value_input_option="RAW")
            sheet.update(
                range_name=range_str,
                values=rows_to_append,
                value_input_option="RAW"
            )

            print(f"✅ Appended {len(rows_to_append)} methods to {range_str}")
            return None
        except APIError as e:
            if "RATE_LIMIT_EXCEEDED" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                sleep_s = 2 ** attempt + random.uniform(0, 0.5)
                print(f"Rate limit hit — retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
            else:
                raise

    print("❌ Failed after retries.")
    return None

            
def is_in_eu(df):
    # Load the unified EU boundary shapefile (GeoPackage format)
    eu_boundary = gpd.read_file(f'{input_path}/EU/EU_nuts/eu_boundary.gpkg', driver='GPKG')
    
    # Extract the geometry of the EU boundary from the GeoDataFrame
    eu_polygon = eu_boundary.unary_union  # This combines all parts of the EU into a single shape
    
    # Function to check if a point is in the EU
    def check_point_in_eu(row):
        point = Point(row['lon'], row['lat'])
        return 1 if eu_polygon.contains(point) else 0
    
    # Apply the function to each row in the DataFrame
    df['eu'] = df.apply(check_point_in_eu, axis=1)
    
    return df

import matplotlib.pyplot as plt
import math

def plot_subplots_histogram(df, column, ref_column, filt=None, value_range=None, bins=30):
    """
    Plots histograms for a specified column in a DataFrame within a given range,
    for each unique value in the ref_column, in separate subplots within the same figure.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        column (str): The column to plot the histogram for.
        ref_column (str): The reference column containing different groups.
        value_range (tuple): A tuple specifying the (min, max) range of values to plot.
        bins (int): Number of bins for the histogram. Default is 30.
        filt (list of string): filter list
    """
    if filt is None:
        unique_refs = df.loc[df[column].notna()][ref_column].dropna().unique()
        num_refs = len(unique_refs)
    else:
        unique_refs = filt
        num_refs = len(filt)
    
    # Determine the layout of subplots (e.g., 2x2, 3x3) based on the number of unique ref values
    cols = 3  # Define a fixed number of columns for better layout
    rows = math.ceil(num_refs / cols)
    
    # Filter the DataFrame based on the specified range
    if value_range:
        df_filtered = df[(df[column] >= value_range[0]) & (df[column] <= value_range[1])]
    else:
        df_filtered = df
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    fig.suptitle(f'Histograms of {column} by {ref_column}', fontsize=16)
    
    # Loop over each unique ref value and plot in a separate subplot
    for i, ref_value in enumerate(unique_refs):
        row, col = divmod(i, cols)
        
        # Filter the DataFrame for the current ref value
        subset = df_filtered[df_filtered[ref_column] == ref_value]
        
        # Plot histogram in the respective subplot
        axes[row, col].hist(subset[column], bins=bins)
        axes[row, col].set_title(f'{ref_value}, {len(subset)}')
        # axes[row, col].set_xlabel(column)
        # axes[row, col].set_ylabel('Frequency')
    
    # Hide any unused subplots
    for i in range(num_refs, rows * cols):
        fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the title
    plt.show()

    
## do automatic conversion
def conversion(data_df, method_df, prop, keep=False):
    """
    Harmonize property measurements to comparable units using conversion formulas.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Standardized dataset (with columns like `dataset_id`, f"{prop}_method", and `prop`).
    method_df : pandas.DataFrame
        Central registry table (Google Sheet export, with `ref` and `method`).
    prop : str
        Property name (e.g., "soc", "ph.h2o", "clay").

    Returns
    -------
    merged_df : pandas.DataFrame
        DataFrame with converted property values and QA scores.
    """

    mtod = f"{prop}_method"

    # --- 1. check for missing methods ---
    existing_pairs = set(zip(method_df["src"], method_df["method"]))
    new_pairs = set(zip(data_df["dataset_id"], data_df[mtod]))

    missing_pairs = new_pairs - existing_pairs
    if missing_pairs:
        print(f"Missing in registry for {prop}: {missing_pairs}")
        return None
    else:
        print(f"All methods for {prop} are in registry.")

    # --- 2. merge method info ---
    method_df = method_df.rename(columns={"src": "dataset_id","method": mtod})
    merged_df = pd.merge(data_df, method_df, on=["dataset_id", mtod], how="left")

    # --- 3. apply conversion ---
    def apply_conversion(row):
        if pd.isna(row[prop]):
            return row[prop]
        formula = row["formula"]
        if pd.isna(formula) or formula.strip() in ["y", "y = x"]:
            return row[prop]
        if pd.isna(formula) or formula.strip() in [""]:
            return np.nan
        try:
            expr = formula.split("=")[-1].strip().replace("x", str(row[prop]))
            return eval(expr, {"__builtins__": {}}, {})
        except Exception as e:
            print(f"Conversion error for {prop}: {formula} with value {row[prop]} → {e}")
            return row[prop]

    merged_df[f"{prop}_converted"] = merged_df.apply(apply_conversion, axis=1)

    # --- 4. rename columns ---
    merged_df = merged_df.rename(columns={prop: f"{prop}_original",f"{prop}_converted": prop})
    # merged_df.loc[merged_df['score']<3,prop] = np.nan

    merged_df = merged_df.drop(columns= ['formula', 'count', 'decision','ref', 'note', 'score', "version", f"{prop}_method"])
    if '' in merged_df.columns:
        merged_df = merged_df.drop(columns= [''])
        
    if keep:
        return merged_df
    else:
        return merged_df.drop(columns= [f"{prop}_original"])


    
    
def check_distribution(df, prop, ref):
    lucas_data = df[df['ref'] == 'LUCAS'][prop].dropna()
    other_data = df[df['ref'] == ref][prop].dropna()
    
    # Calculate statistics
    mw_stat, mw_p_value = mannwhitneyu(lucas_data, other_data) # use Mann-Whitney U Test to see whether these distributions are similar
    ks_stat, ks_p_value = ks_2samp(lucas_data, other_data) # Kolmogorov-Smirnov test
#     min_diff = abs(lucas_data.min() - other_data.min()) / lucas_data.min()
#     mean_diff = abs(lucas_data.mean() - other_data.mean()) / lucas_data.mean()
#     median_diff = abs(lucas_data.median() - other_data.median()) / lucas_data.median()
    
    # Calculate statistics for title
    lucas_mean = lucas_data.mean()
    lucas_median = lucas_data.median()
#     lucas_min = lucas_data.min()
    
    other_mean = other_data.mean()
    other_median = other_data.median()
#     other_min = other_data.min()
    
    # Plot distributions
    plt.figure(figsize=(8, 4))
    plt.hist(lucas_data, bins=30, alpha=0.4, label='LUCAS', density=True)
    plt.hist(other_data, bins=30, alpha=0.4, label=ref, density=True)
    plt.legend(loc='upper right',fontsize=14)
    plt.title(f'mw: {mw_p_value:.2f}, ks: {ks_p_value:.2f}\n'
              f'LUCAS - Mean: {lucas_mean:.2f}, Median: {lucas_median:.2f}, Size: {len(lucas_data):.2f}\n'
              f'{ref} - Mean: {other_mean:.2f}, Median: {other_median:.2f}, Size: {len(other_data):.0f}',fontsize=14)
    plt.xlabel(prop,fontsize=16)
    plt.ylabel('Normalized Frequency',fontsize=16)
    plt.show()
    
#     # Check if distributions are similar
#     if min_diff > 0.2 or mean_diff > 0.2 or median_diff > 0.1:
#         print(f'Distribution of LUCAS and {ref} are not comparable!')
#         df.loc[df['ref'] == ref, f'{prop}_qa'] = df.loc[df['ref'] == ref, f'{prop}_qa'] - 1
    
    return df


    
# used in combination with exam_para
# used for examine the parameters from a compiled dataset, with each property a row
def print_overview(nnlist,filt=0):
    if filt==0:
        print('overview:')
        for i in range(len(nnlist)):
            print(f'{nnlist[i][1]} - {nnlist[i][0]}')
    else:
        print(f'overview with filter {filt}:')
        for i in range(len(nnlist)):
            if filt in nnlist[i][1]:
                print(f'{nnlist[i][1]} - {nnlist[i][0]}')
                

                
# plot spatial distribution over EU
def plot_spatial_distribution(data,title,latbox=[33,72],lonbox=[-12,35]):
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(11, 8))
    hexbin = ax.hexbin(data['lon'], data['lat'], gridsize=150, cmap='RdYlGn', mincnt=1, norm=mcolors.LogNorm())
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(f'{title} - {len(data)} data', fontsize=16)
    
    if latbox is not None:
        ax.set_ylim(latbox)
    if lonbox is not None:
        ax.set_xlim(lonbox)
    colorbar = plt.colorbar(hexbin)
    colorbar.set_label(f'count', fontsize=14)
    
    plt.grid(True)
    plt.show()
    
# # define a function to clean
# def clean_prop(df, prop, limit):
#     print(f'\033[1mCleaning {prop}\033[0m')
#     tot = len(df)
#     # Clean NaN
#     num = df[prop].isna().sum()
#     ccol = df.loc[df[prop].isna()]['ref'].unique()
#     print(f'{num} ({num/tot*100:.2f}%) rows with NaN, from {ccol}')
#     df = df.dropna(subset=[prop])

#     #df.loc[:,prop] = pd.to_numeric(df.loc[:,prop], errors='coerce')
#     df[prop] = pd.to_numeric(df[prop], errors='coerce')
#     num = df[prop].isna().sum()
#     ccol = df.loc[df[prop].isna()]['ref'].unique()
#     print(f'{num} ({num/tot*100:.2f}%) rows with invalid strings, from {ccol}')
#     df = df.dropna(subset=[prop])
    
#     # Check for values below 0, which are invalid for all properties
#     num = len(df.loc[df[prop] < 0])
#     ccol = df.loc[df[prop] < 0]['ref'].unique()
#     print(f'{num} ({num/tot*100:.2f}%) rows with {prop} < 0, from {ccol}')
#     df = df[df[prop] >= 0]
    
#     # check for values higher than plausible limit
#     if limit:
#         num = len(df.loc[df[prop]>limit])
#         ccol = df.loc[df[prop]>limit]['ref'].unique()
#         print(f'{num} ({num/tot*100:.2f}%) rows with {prop} > limit values, from {ccol}')
#         df = df[df[prop] < limit]
    
#     print(f'{len(df)} valid data records left')
#     return df

# clean the data out of valid range
def valid_range_check(data, prop, upper, lower=0):
    # Below lower limit
    low_mask = data[prop] < lower
    low_ref = data.loc[low_mask, 'dataset_id'].unique()
    low_len = low_mask.sum()

    print(f'{low_len} data points are below the lower possible limit of {prop}: {lower}')
    if low_len > 0:
        low_counts = data.loc[low_mask, 'dataset_id'].value_counts()
        for ds_id, count in low_counts.items():
            print(f'  - {ds_id}: {count}')
        data.loc[low_mask, prop] = np.nan

    # Above upper limit
    up_mask = data[prop] > upper
    up_ref = data.loc[up_mask, 'dataset_id'].unique()
    up_len = up_mask.sum()

    print(f'{up_len} data points exceed the upper possible limit of {prop}: {upper}')
    if up_len > 0:
        up_counts = data.loc[up_mask, 'dataset_id'].value_counts()
        for ds_id, count in up_counts.items():
            print(f'  - {ds_id}: {count}')
        data.loc[up_mask, prop] = np.nan

    
    return data

