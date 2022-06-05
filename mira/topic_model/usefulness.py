
from scipy.stats import gmean
import numpy as np
from sklearn.metrics.cluster import silhouette_samples
from collections import defaultdict

def _check_topics(model, data):
    
    if not 'X_topic_compositions' in data.obsm:
        model.predict(data, bar = False)


def MutualInformation(other_data):
    
    def get_MI_metric(x, y):

        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert len(x) == len(y)

        x_marg = x.mean(0,keepdims = True)
        y_marg = y.mean(0, keepdims = True)

        joint = (x[:, np.newaxis, :] * y[:,:, np.newaxis]).mean(axis = 0)
        marg = (x_marg * y_marg.T)[np.newaxis, :,:]

        mutual_information = np.sum(joint*np.log(joint/marg))

        return mutual_information
    
    
    def get_joint_topics(data, other_data):
        
        shared_cells = np.intersect1d(data.obs_names, other_data.obs_names)
        
        return np.array(data[shared_cells].obsm['X_topic_compositions']), \
                np.array(other_data[shared_cells].obsm['X_topic_compositions'])
    
    
    def mutual_information(model, train, test, data):
        
        _check_topics(model, data)
        
        return get_MI_metric(
            *get_joint_topics(data, other_data)
        )
    
    return mutual_information


def ActiveTopics(min_threshold = 0.1):
    
    def active_topics(model, train, test, data):
        
        _check_topics(model, data)
        
        topics = np.array(data.obsm['X_topic_compositions'])
        
        return (topics.max(0) >= min_threshold).sum()/topics.shape[1]
    
    return active_topics


def _stratified_silhouette_score(
        latent_space, 
        cluster,
        agg_fn = lambda x : (np.array(x).mean() + 1)/2,
        metric = 'manhattan',
    ):

    sil_score = silhouette_samples(
        latent_space,
        cluster,
        metric = metric
    )

    d = defaultdict(list)
    for _celltype, _sil_score in zip(cluster, sil_score):
        d[_celltype].append(_sil_score)

    celltype_score = {
        _celltype : agg_fn(scores)
        for _celltype, scores in d.items()
    }

    overall_score = gmean(list(celltype_score.values()))

    return overall_score


def CellTypeSilhouette(cell_type_col, box_cox = 0.5, metric = 'manhattan'):

    def celltype_silhouette(
        model, train, test, data
    ):

        _check_topics(model, test)

        model.get_umap_features(test, box_cox = box_cox)
        celltype = test.obs_vector(cell_type_col).astype(str)

        return _stratified_silhouette_score(
            test.obsm['X_umap_features'],
            celltype,
            metric = metric,
        )
        
    return celltype_silhouette


def BatchSilhouette(batch_col, cell_type_col = None, 
    box_cox = 0.5, metric = 'manhattan'):

    def batch_stratified_score(model, train, test, data):
        
        _check_topics(model, test)

        model.get_umap_features(test, box_cox = box_cox)
        batch = test.obs_vector(batch_col).astype(str)

        _stratified_silhouette_score(
            test.obsm['X_umap_features'], 
            batch,
            agg_fn = lambda x : np.mean( 1 - np.abs(np.array(x)) ),
            metric = metric,
        )

    def cell_stratified_score(model, train, test, data):
        
        _check_topics(model, test)

        model.get_umap_features(model, box_cox = box_cox)
        batch = test.obs_vector(batch_col).astype(str)
        celltype = test.obs_vector(cell_type_col).astype(str)

        intersect_clusters = batch + '::' + celltype

        _stratified_silhouette_score(
            test.obsm['X_umap_features'], 
            intersect_clusters,
            agg_fn = lambda x : np.mean( 1 - np.abs(np.array(x)) ),
            metric = metric
        )

    if cell_type_col is None:
        return batch_stratified_score
    else:
        return cell_stratified_score


def aggregate_usefulness(*usefulness_functions):
    
    def calc_usefulness(model, train, test, data):
        
        model.predict(data, bar = False)
        
        metrics = {
            fn.__name__ : fn(model, train, test, data)
            for fn in usefulness_functions
        }
        
        return gmean(list(metrics.values())), metrics
    
    return calc_usefulness