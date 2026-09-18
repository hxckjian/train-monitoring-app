"""Small ridge ranking baseline with all learned state serialized as JSON."""
import numpy as np

def fit_model(features, labels, alpha=10.0):
    array = features.to_numpy(float)
    medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in array.T])
    array = np.where(np.isfinite(array), array, medians)
    mean, scale = array.mean(axis=0), array.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (array - mean) / scale
    intercept = float(np.mean(labels))
    coef = np.linalg.solve(normalized.T @ normalized + alpha * np.eye(array.shape[1]), normalized.T @ (labels - intercept))
    return dict(version=1, features=list(features.columns), medians=medians.tolist(), mean=mean.tolist(), scale=scale.tolist(), coefficients=coef.tolist(), intercept=intercept, alpha=alpha)

def score_cars(model, features):
    array = features.loc[:, model['features']].to_numpy(float)
    array = np.where(np.isfinite(array), array, model['medians'])
    return ((array - model['mean']) / model['scale']) @ np.array(model['coefficients']) + model['intercept']

def rank_cars(model, features):
    scores = score_cars(model, features)
    return [str(features.index[i]) for i in np.argsort(-scores, kind='stable')]

def rank_decay(ranking, faulty_car, n=8):
    if faulty_car not in ranking:
        return 0.0
    if len(ranking) != n or len(set(ranking)) != n:
        raise ValueError('Ranking must contain each car exactly once.')
    return (n - ranking.index(faulty_car)) / n
