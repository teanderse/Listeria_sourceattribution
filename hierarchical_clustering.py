# -*- coding: utf-8 -*-


# imports
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hc 
import seaborn as sns

#%%

# importing cleaned data for cgMLST or wgMLST
MLST_type = "cg" # cg or wg
cleaned_data = pd.read_csv(f"cleaned_data_forML/{MLST_type}MLSTcleaned_data_forML.csv")

#%%

# spliting source labels, MLST-data and SRA id-number
# (assuming SRA_no and Source is first and last column)
MLST_data = cleaned_data.iloc[:, 1:-1]
labels = cleaned_data.Source

# calculating hamming distances
# normalised hamming distances
distances = pdist(MLST_data, metric= "hamming")
dist_tbl = squareform(distances)
# not normalised hamming distances
wg_featureLength = 2496
cg_featureLength = 956
nonNormalised_dist_tbl = dist_tbl.copy()
nonNormalised_dist_tbl = nonNormalised_dist_tbl*wg_featureLength

# defining linkage for clustering
linkage = hc.linkage(distances, method='average')

#%%

# giving label a colour in the dendrogram
colour = sns.color_palette("colorblind", 5)
colour_map = dict(zip(labels.unique(), colour))
row_colors = labels.map(colour_map).to_numpy()

# hierarchical clustering using hamming distances visualised in a heatmap and dendrogram
clusterplot = sns.clustermap(nonNormalised_dist_tbl, row_linkage=linkage, col_linkage=linkage, row_colors=row_colors,
              cmap="mako", cbar_kws={"orientation": "horizontal",'label':'Hamming distance'}, 
              cbar_pos=(.25, 0, .7, .03))

# adding plot details 
clusterplot.ax_heatmap.set_title(f"Hierarchical clustering heatmap for {MLST_type}MLST data", pad=70, size=22)
for (label, colour) in colour_map.items():
    clusterplot.ax_row_dendrogram.bar(0, 0, color=colour, label="{}".format(label))
clusterplot.ax_row_dendrogram.legend(title="Sources", ncol=1, loc='upper left', bbox_to_anchor=(0, 1.2))
clusterplot.ax_col_dendrogram.set_visible(False)
clusterplot.ax_heatmap.tick_params(bottom = False, right = False , labelright = False ,labelbottom = False)
