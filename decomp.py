from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, FastICA, LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler

def get_decomp(features, decomp_fn, components=2, seed=None):
    model = None
    if 'ica':
        model = FastICA(n_components=components, random_state=seed)
    elif 'lda':
        model = LatentDirichletAllocation(n_components=components, random_state=seed)
    elif 'pca':
        model = PCA(n_components=components, random_state=seed)
    elif 'grp':
        model = GaussianRandomProjection(n_components=components, random_state=seed)
    else:
        raise AttributeError("Function not found.")
    
    transformed = model.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)