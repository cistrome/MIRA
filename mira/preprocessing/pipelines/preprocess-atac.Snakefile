
include: "rules/align_fastqs.rule"
include: "rules/make_countmatrix.rule"

rule all:
    input:
        rules.aggregate_peakcounts.output[0].format(
            directory = config['directory'],
            batch = 'bulk', sample = 'bulk'
        )