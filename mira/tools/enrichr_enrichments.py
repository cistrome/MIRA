import requests
import json
from mira.plots.enrichment_plot import plot_enrichment
from functools import partial
from mira.plots.base import map_plot


ENRICHR_URL = 'http://maayanlab.cloud/Enrichr/'
POST_ENDPOINT = 'addList'
GET_ENDPOINT = 'enrich?userListId={list_id}&backgroundType={ontology}'
HEADERS = 'rank,term,pvalue,zscore,combined_score,genes,adj_pvalue'.split(',')
LEGACY_ONTOLOGIES = 'WikiPathways_2019_Human,WikiPathways_2019_Mouse,KEGG_2019_Human,KEGG_2019_Mouse,GO_Molecular_Function_2018,GO_Cellular_Component_2018,GO_Biological_Process_2018,BioPlanet_2019'.split(',')

def post_genelist(genelist):
    
    payload = {
        'list': (None, '\n'.join(genelist)),
    }

    response = requests.post(ENRICHR_URL + POST_ENDPOINT, files=payload)
    if not response.ok:
        raise Exception('Error analyzing gene list')

    list_id = json.loads(response.text)['userListId']
    return list_id


def fetch_ontology(list_id, ontology = 'WikiPathways_2019_Human'):

    url = ENRICHR_URL + GET_ENDPOINT.format(
        list_id = str(list_id),
        ontology = str(ontology)
    )

    response = requests.get(url)
    if not response.ok:
        raise Exception('Error fetching enrichment results: \n' + str(response))
    
    data = json.loads(response.text)[ontology]
    
    return {ontology : [dict(zip(HEADERS, x)) for x in data]}


def fetch_ontologies(list_id, ontologies = LEGACY_ONTOLOGIES):

    results = {}

    for ontology in ontologies:
        results.update(
            fetch_ontology(list_id, ontology)
        )

    return results


def plot_enrichments(enrichment_results, show_genes = True, show_top = 10, barcolor = 'lightgrey', label_genes = [],
        text_color = 'black', return_fig = False, enrichments_per_row = 2, height = 4, aspect = 2.5, max_genes = 15,
        pval_threshold = 1e-5, color_by_adj = True, palette = 'Reds', gene_fontsize = 10):

    func = partial(plot_enrichment, text_color = text_color, label_genes = label_genes, pval_threshold = pval_threshold,
            show_top = show_top, barcolor = barcolor, show_genes = show_genes, max_genes = max_genes,
            color_by_adj = color_by_adj, palette = palette, gene_fontsize=gene_fontsize)

    fig, ax = map_plot(func, list(enrichment_results.items()), plots_per_row = enrichments_per_row, 
        height =height, aspect = aspect)  

    if return_fig:
        return fig, ax

