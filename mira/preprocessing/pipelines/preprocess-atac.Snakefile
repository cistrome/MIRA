
include: "rules/align_atac.rules"
include: "rules/make_countmatrix.rules"

rule all:
    input:
        '{directory}/Peak_counts.h5ad'.format(directory = config['directory'])

rule rename_countmatrix:
    input:
        rules.aggregate_peakcounts.output[0].format(
            directory = config['directory'],
            batch = 'bulk', sample = 'bulk'
        )
    output:
        '{directory}/Peak_counts.h5ad'
    shell:
        'cp {input} {output}'

