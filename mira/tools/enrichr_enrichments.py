import requests
import json
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