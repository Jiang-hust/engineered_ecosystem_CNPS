import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

# Load the data
file_path = 'SR_train.csv'
data = pd.read_csv(file_path)

# Calculate Pearson rank-order correlations
pearson_corr = data.corr(method='pearson')

# Perform hierarchical clustering
linkage_matrix = hierarchy.linkage(pearson_corr, method='ward')

# Create a list of feature names
feature_names = data.columns

# Display hierarchical clustering with feature names
fig = plt.figure(figsize=(10, 8))
dendrogram = hierarchy.dendrogram(linkage_matrix, labels=feature_names,
                                  orientation='top', leaf_font_size=15
                                  )

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=90)

plt.title("Hierarchical Clustering Dendrogram (pearson)")
plt.xlabel("Data Features")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

