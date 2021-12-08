# Changelog

## 0.0.7

#### Updates

* Changed **mira.topics.BaseModel.get_umap_features** to pull topic compositions 
from anndata instead of recomputing them. Faster and less redundant.
* Added **box_cox** parameter to **mira.topics.BaseModel.get_umap_features**. Tuning
box_cox transformation constant can influence the connectiveness of joint KNN graph.

#### Fixes
* Fixed residual featurization only learning statistics of data from first batch.
Now, statistics are calculated from full dataset as intended and consistent
with models used in manuscript.


## 0.0.0 - 0.0.6

* Various fixes while preparing package for distribution.