News
====

----

MIRA version 2.0 - 06/20/2023
-----------------------------

With our next publication in Nature Communications, we introduced a new batch
effect correction method which is purpose built for multiomic analysis. The algorithm,
called `CODAL <https://rdcu.be/dgCQF>`_, uses a novel objective function to "disentangle" technical effects and
maximally preserve biological variance. This makes the CODAL algorithm especially well
suited to studying the effects of perturbations on single cells, where the biological
change induced by the perturbation could be obscured by technical noise. In benchmarking
tests, the CODAL algorithm produced results on-par with current state-of-the-art supervised 
methods, while requiring no prior knowledge about the sample.

With the CODAL update, we completely re-vamped the hyperparameter tuning method
to be 10-20x faster, more robust, and highly parallelizable. Of note, we introduced the
`gradient_tune` method, which uses gradient descent optimization to automatically select the
hyperparameters for the model, reducing the time taken for hyperparameter selection many-fold.

Coming next, the Regulatory Potential modeling section of the package is due for a major update which
will dramatically increase its speedy, scalability, and interpretability.


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