# -*- coding: utf-8 -*-


# imports
import pandas as pd

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

# joining cgMLST_input with sourcelabel
cgMLST_in_data = cgMLST_input.join(Sourcelabel_input.set_index("SRA_no"), on="SRA_no")

# remove sources with less than threshold isolats
isolate_threshold = 15
cgMLST_in_data = cgMLST_in_data[cgMLST_in_data.groupby(cgMLST_in_data.Source)["Source"].transform('size')>isolate_threshold]

# remove clinical isolates
cgMLST_in_data = cgMLST_in_data[cgMLST_in_data.Source != "clinical"]
cgMLST_cleaned_data = cgMLST_in_data.copy()

# remove flags from the cgMLST data
cgMLST_cols = [col for col in cgMLST_cleaned_data if col.startswith('Pasteur')]
for col in cgMLST_cols: 
    cgMLST_cleaned_data[col] = cgMLST_cleaned_data[col].astype(str).str.replace("INF-", "", regex = False)
    cgMLST_cleaned_data[col] = cgMLST_cleaned_data[col].astype(str).str.replace("*", "", regex = False)
    cgMLST_cleaned_data[col] = pd.to_numeric(cgMLST_cleaned_data[col], errors="coerce")
                                            
#%% 

# removing columns and rows with 10% or more missing values

# values before drop based on missing values
before_col = cgMLST_cleaned_data.shape[1]
before_row = cgMLST_cleaned_data.shape[0]

# removing columns
cgMLST_cleaned_data.dropna(thresh=(round(before_row*0.9)), axis=1, inplace=True)
print("Dropped {} columns with over 10% missing values.".format(before_col - cgMLST_cleaned_data.shape[1]))

# removing rows
cgMLST_cleaned_data.dropna(thresh=(round(before_col*0.9)), axis=0, inplace=True)
print("Dropped {} rows with over 10% missing values.".format(before_row - cgMLST_cleaned_data.shape[0]))

# replacing nan-values with -1
cgMLST_cleaned_data.fillna(-1, inplace=True)

# saving cleaned data
cgMLST_cleaned_data.to_csv("cgMLSTcleaned_data_forML.csv", index=False)

#%%

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

# joining wgMLST_input with sourcelabel
wgMLST_in_data = wgMLST_input.join(Sourcelabel_input.set_index("SRA_no"), on="SRA_no")

# remove sources with less than threshold isolats
isolate_threshold = 15
wgMLST_in_data = wgMLST_in_data[wgMLST_in_data.groupby(wgMLST_in_data.Source)["Source"].transform('size')>isolate_threshold]

# remove clinical isolates
wgMLST_in_data = wgMLST_in_data[wgMLST_in_data.Source != "clinical"]
wgMLST_cleaned_data = wgMLST_in_data.copy()

# remove flags from the wgMLST data
wgMLST_cols = wgMLST_cleaned_data.iloc[:, 1:-1].columns.tolist()
for col in wgMLST_cols: 
    wgMLST_cleaned_data[col] = wgMLST_cleaned_data[col].astype(str).str.replace("INF-", "", regex = False)
    wgMLST_cleaned_data[col] = pd.to_numeric(wgMLST_cleaned_data[col], errors="coerce")

#%% 

# removing columns with 10% or more missing values

# values before drop based on missing values
wgMLSTbefore_col = wgMLST_cleaned_data.shape[1]
wgMLSTbefore_row = wgMLST_cleaned_data.shape[0]

# removing columns
wgMLST_cleaned_data.dropna(thresh=(round(wgMLSTbefore_row*0.9)), axis=1, inplace=True)
print("Dropped {} columns with over 10% missing values.".format(wgMLSTbefore_col - wgMLST_cleaned_data.shape[1]))

# replacing nan-values with -1
wgMLST_cleaned_data.fillna(-1, inplace=True)

# saving cleaned data
wgMLST_cleaned_data.to_csv("wgMLSTcleaned_data_forML.csv", index=False)
