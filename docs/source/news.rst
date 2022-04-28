News
====

----

MIRA version 1.0 - 03/15/2022
-------------------------------

This documentation website will be launched alongside the first 
official version of MIRA! The models/statistical code is stable, and most
new features will be devoted towards accessing
information more easily, decreasing runtime, and adding more
flexibility to plotting functions. Releasing version 1.0.0
reflects that this code embdoies what we set out to design,
and represents a comprehensive base of features on which users can
build multiomics analyses. The biggest changes with this release are:

* Multiprocessing and tensorboard support for topic model training
* New topic model trial pruner which saves time and improves results
* Multiprocessing for RP model training
* Facilities for visualizing regulatory potential models
* Bug fixes and tweaks to streamplot functions
* Added seeding to topic model tuning and terminal point selection to facilitate deterministic analyses
* Accessors for various attributes from AnnDatas (motif hits, TSS-peak distances, etc.)
* Extended support for MIRA streamgraphs and lineage inference to velocity experiments via CellRank
* Added pre-built TSS and chromosome size annotations for mouse (mm10) and human (hg38)

The only breaking changes are some plotting functions had an `add_legend` parameter,
while others had `show_legend`. We changed this so that all plots have `show_legend`, consistent with
Seaborn conventions.

Code Ocean Capsule - 03/15/2022
-------------------------------

We packaged our analysis of the SHARE-seq skin dataset (Ma et al. 2020) as a
`code ocean capsule <https://codeocean.com/>`_, which enables users to check out the
code used to generate figures and conduct the analysis.

New website! - 03/15/2022
-------------------------

Welcome to MIRA's new documentation website! We hope you find it
informative and aesthetic. Future additions to documentation and
expansions to the MIRA analysis suite will live here.