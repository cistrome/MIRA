# Changelog

## 0.1.1

#### Updates

* Added method: `mira.pl.plot_eigengap`, which shows the eigengap heuristic
for a dataset. User must now specify how many diffusion components to use for
downstream analysis.
* Added mira.info for tracking important links and attributes for the project.
* Set default joint representation box-cox transformation to 0.5.


## 0.1.0

#### Updates

* Changed mira.time API to be able to take arbitrary connectivity/distance
matrices from anndata object for more flexibility.
* Changed saving names from cis to LITE for LITE model. Added methods to convert
savenames from older models to new models and added methods to detect when a 
user should run the convert command.
* Added method ``mira.rp.LITE_Model.load_dir`` to load in a bunch of RP models at
once without having to specify the genes up-front.
* Changed signature of ``mira.topics.AccessibilityModel.plot_compare_module_enrichments``
to ``mira.topics.AccessibilityModel.plot_compare_topic_enrichments`` to reflect change
in vernacular from "modules" to "topics".
* Changed all references to "modules" in function signatures to "topics".

#### Fixes

* Fixed bug where ``mira.time.normalize_diffmap`` was taking one less diffusion map
component than expected.


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