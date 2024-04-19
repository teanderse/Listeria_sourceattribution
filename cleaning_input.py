# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np

#%%

# Cleaning input data for cgMLST
# cgMLST_input is the CHEWBBACA_results_allels.tsv file from the cgMLST track of the ALPPACA pipeline.
# Sourcelabel_input is a file with the id column matching the number in the file column of the cgMLST_input 
# and one column for the source of the isolate.

cgMLST_input = pd.read_csv("CHEWBBACA_results_alleles.tsv", sep="\t")
Sourcelabel_input = pd.read_csv("raw_source_lables.csv")

# rename FILE column to SRA_no to match Sourcelabel_input
cgMLST_input.rename(columns={"FILE": "SRA_no"}, inplace=True)

# removing string from SRA number to match Sourcelabel_input
cgMLST_input["SRA_no"] = cgMLST_input["SRA_no"].str.replace("_pilon_spades", "")

# joining cgMLST_input with sourcelabel to add a class variable for the isolates sources
cgMLST_in_data = cgMLST_input.join(Sourcelabel_input.set_index("SRA_no"), on="SRA_no")

cgMLST_cleaned_data = cgMLST_in_data.copy()

# remove flags from the cgMLST data
cgMLST_cols = [col for col in cgMLST_cleaned_data if col.startswith('Pasteur')]
for col in cgMLST_cols: 
    cgMLST_cleaned_data[col] = cgMLST_cleaned_data[col].astype(str).str.replace("INF-", "", regex = False)
    cgMLST_cleaned_data[col] = cgMLST_cleaned_data[col].astype(str).str.replace("*", "", regex = False)
    cgMLST_cleaned_data[col] = pd.to_numeric(cgMLST_cleaned_data[col], errors="coerce")

#%%
    
# remove and store clinical isolates in seperat variable
cgMLST_clinical_samples = cgMLST_cleaned_data.loc[cgMLST_cleaned_data["Source"] == "clinical"].copy()
# replacing nan-values with -1 in the clinical isolates
cgMLST_clinical_samples.fillna(-1, inplace=True)

# cgMLST data without clinical isolates
cgMLST_filtered_data = cgMLST_cleaned_data.loc[cgMLST_cleaned_data.Source != "clinical"].copy()
                                            
#%% 

# filtering out nan-values 
# removing columns and rows with 10% or more missing values

# values before drop based on missing values
before_row = cgMLST_filtered_data.shape[0]
before_col = cgMLST_filtered_data.shape[1]

# removing rows
cgMLST_filtered_data.dropna(thresh=(round(before_col*0.9)), axis=0, inplace=True)
print("Dropped {} rows with over 10% missing values.".format(before_row - cgMLST_filtered_data.shape[0]))

# removing columns
cgMLST_filtered_data.dropna(thresh=(round(before_row*0.9)), axis=1, inplace=True)
print("Dropped {} columns with over 10% missing values.".format(before_col - cgMLST_filtered_data.shape[1]))

# remove sources with less than threshold isolats
isolate_threshold = 15
cgMLST_filtered_data = cgMLST_filtered_data[cgMLST_filtered_data.groupby(cgMLST_filtered_data.Source)["Source"].transform('size')>isolate_threshold]

# replacing remaining nan-values with -1
cgMLST_filtered_data.fillna(-1, inplace=True)

#%%

# remove the filtered out columns from clinical isolates
cols_notdroped = np.intersect1d(cgMLST_clinical_samples.columns, cgMLST_filtered_data.columns)
cgMLST_clinical_samples = cgMLST_clinical_samples[cols_notdroped]

# saving cleaned cgMLST data and clinical isolates
cgMLST_filtered_data.to_csv("cgMLSTcleaned_data_forML.csv", index=False)
cgMLST_clinical_samples.to_csv("cgMLST_clinical_samples.csv", index=False)

#%%----------------------------------------------------------------------------------------------------------

# Cleaning input data for wgMLST
# wgMLST_input is the results_allels_NoParalogs.tsv file from the allele call and remove gene of chewbbaca.
# Sourcelabel_input is a file with the id column matching the number in the file column of the wgMLST_input 
# and one column for the source of the isolate.

wgMLST_input = pd.read_csv("results_alleles_NoParalogs.tsv", sep="\t")
Sourcelabel_input = pd.read_csv("raw_source_lables.csv")

# rename FILE column to SRA_no to match Sourcelabel_input
wgMLST_input.rename(columns={"FILE": "SRA_no"}, inplace=True)

# removing string from SRA number to match Sourcelabel_input
wgMLST_input["SRA_no"] = wgMLST_input["SRA_no"].str.replace("_pilon_spades", "")

# joining wgMLST_input with sourcelabel to add a class variable for the isolates sources
wgMLST_in_data = wgMLST_input.join(Sourcelabel_input.set_index("SRA_no"), on="SRA_no")

wgMLST_cleaned_data = wgMLST_in_data.copy()

# remove flags from the wgMLST data
wgMLST_cols = wgMLST_cleaned_data.iloc[:, 1:-1].columns.tolist()
for col in wgMLST_cols: 
    wgMLST_cleaned_data[col] = wgMLST_cleaned_data[col].astype(str).str.replace("INF-", "", regex = False)
    wgMLST_cleaned_data[col] = wgMLST_cleaned_data[col].astype(str).str.replace("*", "", regex = False)
    wgMLST_cleaned_data[col] = pd.to_numeric(wgMLST_cleaned_data[col], errors="coerce")

#%%

# remove and store clinical isolates in seperat variable
wgMLST_clinical_samples = wgMLST_cleaned_data.loc[wgMLST_cleaned_data["Source"] == "clinical"].copy()
# replacing nan-values with -1 in the clinical isolates
wgMLST_clinical_samples.fillna(-1, inplace=True)

# wgMLST data without clinical isolates
wgMLST_filtered_data = wgMLST_cleaned_data.loc[wgMLST_cleaned_data["Source"] != "clinical"].copy()

#%% 

# filtering out nan-values
# removing columns with 10% or more missing values

# values before drop based on missing values
wgMLSTbefore_col = wgMLST_filtered_data.shape[1]
wgMLSTbefore_row = wgMLST_filtered_data.shape[0]

# removing columns
wgMLST_filtered_data.dropna(thresh=(round(wgMLSTbefore_row*0.9)), axis=1, inplace=True)
print("Dropped {} columns with over 10% missing values.".format(wgMLSTbefore_col - wgMLST_filtered_data.shape[1]))

# remove sources with less than threshold isolats
isolate_threshold = 15
wgMLST_filtered_data = wgMLST_filtered_data[wgMLST_filtered_data.groupby(wgMLST_filtered_data.Source)["Source"].transform('size')>isolate_threshold]

# replacing nan-values with -1
wgMLST_filtered_data.fillna(-1, inplace=True)

# remove the filtered out columns from clinical isolates
cols_notdroped = np.intersect1d(wgMLST_clinical_samples.columns, wgMLST_filtered_data.columns)
wgMLST_clinical_samples = wgMLST_clinical_samples[cols_notdroped]

# saving cleaned wgMLST data and clinical isolates
wgMLST_filtered_data.to_csv("wgMLSTcleaned_data_forML.csv", index=False)
wgMLST_clinical_samples.to_csv("wgMLST_clinical_samples.csv", index=False)
