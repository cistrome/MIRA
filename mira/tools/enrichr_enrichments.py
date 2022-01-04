import requests
import json
from functools import partial
from mira.plots.base import map_plot
from collections.abc import Iterable
import logging
requests.logging.getLogger().setLevel(logging.WARN)


ENRICHR_URL = 'http://maayanlab.cloud/Enrichr/'
POST_ENDPOINT = 'addList'
GET_ENDPOINT = 'enrich?userListId={list_id}&backgroundType={ontology}'
HEADERS = 'rank,term,pvalue,zscore,combined_score,genes,adj_pvalue'.split(',')
LEGACY_ONTOLOGIES = 'WikiPathways_2019_Human,WikiPathways_2019_Mouse,KEGG_2019_Human,KEGG_2019_Mouse,GO_Molecular_Function_2018,GO_Cellular_Component_2018,GO_Biological_Process_2018,BioPlanet_2019'.split(',')

def post_genelist(genelist):
    '''
    Post genelist to Enrichr for comparison against pre-compiled ontologies.

    Parameters
    ----------
    genelist : Iterable
        List of genes

    Returns
    -------
    list_id : str
        ID for genelist. Used to retrieve enrichment results.
    '''
    assert(isinstance(genelist, Iterable)), 'Genelist must be an iterable object'

    payload = {
        'list': (None, '\n'.join(genelist)),
    }

    response = requests.post(ENRICHR_URL + POST_ENDPOINT, files=payload)
    if not response.ok:
        raise Exception('Error analyzing gene list')

    list_id = json.loads(response.text)['userListId']
    return list_id


def fetch_ontology(list_id, ontology = 'WikiPathways_2019_Human'):
    '''
    Fetch enrichment results from an ontology.

    Parameters
    ----------
    list_id : str
        genelist ID returned by `post_genelist`
    onotology : str, default = "WikiPathways_2019_Human"
        Retrieve results for this ontology. For a full list of 
        ontologies, see [enrichr](https://maayanlab.cloud/Enrichr/#libraries).

    Returns
    -------
    results : dict
        Dictionary with schema:
            {
                <ontology> : {
                    [
                        {'rank' : <rank>,
                        'term' : <term>,
                        'pvalue' : <pval>,
                        'zscore': <zscore>,
                        'combined_score': <combined_score>,
                        'genes': [<gene1>, ..., <geneN>],
                        'adj_pvalue': <adj_pval>},
                        ...,
                    ]
                }
            }   
    '''

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
    '''
    Fetch enrichment results from ontologies.

    Parameters
    ----------
    list_id : str
        genelist ID returned by `post_genelist`
    onotologies : Iterable[str], default = mira.tl.LEGACY_ONTOLOGIES
        Retrieve results for these ontologies. For a full list of 
        ontologies, see [enrichr](https://maayanlab.cloud/Enrichr/#libraries).

    Returns
    -------
    results : dict
        Dictionary with schema:
            {
                <ontology> : {
                    [
                        {'rank' : <rank>,
                        'term' : <term>,
                        'pvalue' : <pval>,
                        'zscore': <zscore>,
                        'combined_score': <combined_score>,
                        'genes': [<gene1>, ..., <geneN>],
                        'adj_pvalue': <adj_pval>},
                        ...,
                    ]
                }
            }   
    '''

    results = {}
    assert(isinstance(ontologies, Iterable)), 'Ontologies must be an iterable object'

    for ontology in ontologies:
        results.update(
            fetch_ontology(list_id, ontology)
        )

    return results