# -*- coding: utf-8 -*-


# imports
import pandas as pd

#%%

# Cleaning input data.
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
in_data = cgMLST_input.join(Sourcelabel_input.set_index("SRA_no"), on="SRA_no")

# remove sources with less than threshold isolats
isolate_threshold = 15
in_data = in_data[in_data.groupby(in_data.Source)["Source"].transform('size')>isolate_threshold]

# remove clinical isolates
in_data = in_data[in_data.Source != "clinical"]
cleaned_data = in_data.copy()

# remove flags from the cgMLST data
cgMLST_cols = [col for col in cleaned_data if col.startswith('Pasteur')]
for col in cgMLST_cols: 
    cleaned_data[col] = cleaned_data[col].astype(str).str.replace("INF-", "", regex = False)
    cleaned_data[col] = cleaned_data[col].astype(str).str.replace("*", "", regex = False)
    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors="coerce")
                                            
#%% 

# removing columns and rows with 10% or more missing values

# values before drop based on missing values
before_col = cleaned_data.shape[1]
before_row = cleaned_data.shape[0]

# removing columns
cleaned_data.dropna(thresh=(round(before_row*0.9)), axis=1, inplace=True)
print("Dropped {} columns with over 10% missing values.".format(before_col - cleaned_data.shape[1]))

# removing rows
cleaned_data.dropna(thresh=(round(before_col*0.9)), axis=0, inplace=True)
print("Dropped {} rows with over 10% missing values.".format(before_row - cleaned_data.shape[0]))

# replacing nan-values with -1
cleaned_data.fillna(-1, inplace=True)

# saving cleaned data
# cleaned_data.to_csv("cleaned_data_forML.csv", index=False)

