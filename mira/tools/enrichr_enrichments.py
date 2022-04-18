import requests
import json
from collections.abc import Iterable
import logging


ENRICHR_URL = 'http://maayanlab.cloud/Enrichr/'
POST_ENDPOINT = 'addList'
GET_ENDPOINT = 'enrich?userListId={list_id}&backgroundType={ontology}'
HEADERS = 'rank,term,pvalue,zscore,combined_score,genes,adj_pvalue'.split(',')
LEGACY_ONTOLOGIES = 'WikiPathways_2019_Human,WikiPathways_2019_Mouse,KEGG_2019_Human,KEGG_2019_Mouse,GO_Molecular_Function_2018,GO_Cellular_Component_2018,GO_Biological_Process_2018,BioPlanet_2019'.split(',')

example_genelist = ['SKINT2', 'FMN1', 'NOS1AP', 'AKAP17B', 'SPHK2', '9330185C12RIK',
       'EPHB2', 'FOXP3', 'AMIGO1', 'MKNK2', 'RIMS2', 'PAK3', 'MROH3',
       'A130014A01RIK', 'GUK1', 'GM14051', 'SERPINB3B', 'FAM163B',
       'SLC11A2', 'SULT2B1', 'ADCY5', 'PPL', 'MOGS', 'GM17089', 'RC3H1',
       'AU041133', 'RUSC1', 'FUT8', 'FAM57A', 'TMEM154', 'TJP3', 'HIP1R',
       'SH3KBP1', 'FHDC1', '4933406C10RIK', '2310046K23RIK', 'KRT17',
       'CYP2B19', 'LYPD6B', 'DUSP22', 'HMGCR', 'SKINT6', 'SPAG1', 'IL1F6',
       'RDH12', 'NRTN', 'SKINT10', 'RGS20', 'ZFP790', 'HOPX', 'ADD2',
       'CLASRP', 'VMN2R6', 'LRRC48', 'FAAH', 'VAMP2', 'SQLE', 'DOCK2',
       'BICD2', 'MAPRE2', 'VSIG10L', 'TPTE', 'SC5D', 'TMEM232', 'NDUFA6',
       'ESR2', 'GPR87', 'ENDOD1', 'PERM1', 'TMEM19', 'GM26550', 'TMEM116',
       'ABHD6', 'MEGF10', 'DHPS', 'SORBS1', 'DPP3', 'SLC6A14', 'DOHH',
       'KRTDAP', 'PLEKHN1', 'RBAK', 'LIN28A', 'ANO9', 'CIB1', 'PTPN21',
       'CC2D1A', 'GRID2IP', 'KCNJ6', 'GM44293', 'GM12132', 'GM15594',
       'TPRG', 'TMEM51OS1', 'SMDT1', 'SAMD10', 'GM281', 'PTGS1', 'TEX264',
       'DNAJC25', 'ANKRD27', 'HOMER2', 'A330069K06RIK', 'CCR9', 'KRT23',
       'CYP51', 'DSG1C', 'HMGCS1', 'PPP1R10', 'MAP4K3', 'TMEM62', 'CHP1',
       'AADACL2', 'YPEL4', 'APTX', 'SKINT5', 'LIN7B', 'MYL10', 'IRAK2',
       'CTTNBP2', 'OTOP3', 'RBP2', 'CYFIP2', 'CBARP', 'IGF2BP3', 'ELOVL4',
       'SLC5A9', 'GM20618', 'MSMO1', 'ZFP954', 'SLC17A5', 'KLHL14',
       'PAF1', 'GM10549', 'CCDC9', 'ZKSCAN14', 'CSK', 'SFT2D2', 'TTC7',
       'HBB-BS', 'IDI1', 'RAB24', 'MICU1', 'KCND2', 'FDFT1', 'DRC1',
       'PCSK6', 'KIF9', 'CARD14', 'VMN2R60', '5031414D18RIK', 'ARHGEF33',
       'DAPL1', 'GBA', 'ID4', 'MPND', 'GM12648', 'MYO5B', 'MARCH3',
       'TEAD3', 'ANXA9', 'MAP7D1', 'NACC1', 'ARRDC2', 'AGPAT4', 'FAM188A',
       'CDC26', 'RP24-222O22.4', 'NOX1', 'SBSN', 'TGM1', 'SYCP2L',
       'PRAMEF12', '4930562F17RIK', 'SLC9A7', 'GM27007', 'PCYT1A',
       'CHD3OS', 'GM26935', 'PLXDC2', 'RAB11FIP1', 'GAN', 'KDSR', 'MYZAP',
       'ADH6A', 'ASPG', 'GDPD3', '9530059O14RIK', 'GM12968', 'DMKN',
       'LMO7', 'GGH', 'EPHB1', 'PTGR1', 'CDSN', 'RP23-458C8.2',
       'RALGAPA2', 'PEPD', 'PNPO', 'LRRC51', 'GM9821', 'ARHGAP40',
       'RP23-350G1.5', 'GANC', 'TOMM34', 'EPS8L1', 'ACAP2', 'IVL',
       'DGAT2', 'LYPD5', 'CACNB4', 'PRRC1', 'DKKL1', 'FLG2', 'FAM107B',
       'MROH6', 'ASAH2', 'MAL2', 'C130079G13RIK', 'HMG20B', 'HAL', 'LOR',
       'ABTB2', 'FAM65C', 'PEX13', 'EML3', 'MXI1', 'PLA2G4E', 'SPINK5',
       'CSNK2A2', 'DPP6', 'GM12436', 'EREG', 'DNASE1L3', 'INPP5B', 'MAP2',
       'ABCA5', 'MBOAT2', 'CASP14', 'IL1F5', 'TRIOBP', 'FAM3B', 'SLC5A1',
       'SBF1', 'GRAMD3', 'TSPAN8', 'GM12766', 'SRPK1', 'FLG', 'MVB12A']


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
        ontologies, see `Enrichr <https://maayanlab.cloud/Enrichr/#libraries>`_.

    Returns
    -------
    results : dict
        Dictionary with schema:

        .. code-block::

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

    try:
        import charset_normalizer
        charset_normalizer.logging.getLogger().setLevel(logging.WARN)
    except ModuleNotFoundError:
        pass

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
        ontologies, see `Enrichr <https://maayanlab.cloud/Enrichr/#libraries>`_.

    Returns
    -------
    results : dict
        Dictionary with schema:

        .. code-block::
    
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