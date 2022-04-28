include: "rules/align_rna.rules"

rule all:
    input:
        [
            rules.aggregate_countmatrix.output[0].format(
            directory = config['directory'],
            feature = feature)
            for feature in config['mapping']['features']
        ]