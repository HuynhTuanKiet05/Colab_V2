import numpy as np


def _as_symmetric_float(matrix):
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected a 2D similarity matrix, got shape {arr.shape}')
    return 0.5 * (arr + arr.T)


def _fuse_nonzero_mean(matrices):
    stack = np.stack([_as_symmetric_float(matrix) for matrix in matrices], axis=0)
    valid = (np.abs(stack) > 1e-12).astype(np.float32)
    weighted = stack * valid
    count = valid.sum(axis=0)
    fused = np.divide(weighted.sum(axis=0), np.clip(count, 1.0, None), where=count > 0, out=np.zeros_like(weighted[0]))
    return fused.astype(np.float32)


def _restore_diagonal(fused, source_views):
    diag = np.maximum.reduce([np.diag(_as_symmetric_float(view)) for view in source_views]).astype(np.float32)
    np.fill_diagonal(fused, diag)
    return fused


def repair_similarity_views(data, mode='nonzero_mean'):
    if mode == 'legacy':
        return data

    drug_views = [data['drf'], data['drg']]
    disease_views = [data['dip'], data['dig']]

    if mode == 'mean':
        drug_fused = _as_symmetric_float(drug_views[0] + drug_views[1]) / 2.0
        disease_fused = _as_symmetric_float(disease_views[0] + disease_views[1]) / 2.0
    elif mode == 'nonzero_mean':
        drug_fused = _fuse_nonzero_mean(drug_views)
        disease_fused = _fuse_nonzero_mean(disease_views)
    else:
        raise ValueError(f'Unsupported similarity fusion mode: {mode}')

    data['drs'] = _restore_diagonal(drug_fused, drug_views)
    data['dis'] = _restore_diagonal(disease_fused, disease_views)
    return data


def collect_similarity_views(data, entity):
    if entity == 'drug':
        keys = ['drf', 'drg', 'drs']
    elif entity == 'disease':
        keys = ['dip', 'dig', 'dis']
    else:
        raise ValueError(f'Unsupported entity: {entity}')

    views = []
    for key in keys:
        matrix = data.get(key)
        if matrix is not None:
            views.append(_as_symmetric_float(matrix))
    if not views:
        raise ValueError(f'No similarity views found for entity: {entity}')
    return views
