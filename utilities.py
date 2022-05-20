"""
utilities associated with the paper "Random projections and kernelised
Leave One Cluster Out Cross-Validation: Universal baselines and evaluation
tools for supervised machine learning for materials properties"

"""
import regex as re
from sklearn.cluster import KMeans
import pandas as pd
try:
    import domain_knowledge.utils.composition as dk
    dk.path=r'domain_knowledge/data/element_properties/'
except Exception as e:
    raise Exception('domain_knowledge library not found. Please read and follow the instructions in the README.md to download it') from e
with open('periodictable.csv') as f:
    periodic_table = [x.rstrip() for x in f.readlines()]
element_regex = re.compile(r'(([A-Z][a-z]?)\s?([0-9]*(\.[0-9]*)?)?\s?)')
def find_clusterings(data, formulae):
    """clusters data using kmeans clustering for values of k between 2 and 10

    Parameters:
    data (pandas Dataframe or np.ndarray): data to cluster
    formulae (pandas Series or list of strings): formulae associated with each row of data
    Returns:
    list: clusters in the form [{'k':2, 'formulae':['H2O','NaCl'....],
          'clusters':[0,1...]},{'k':2...]

   """
    clusters = []
    for k in range(2,11):
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(data)
        clusters.append({'k':k, # How many clusters
                         'labels':labels.tolist(), #Labels for each data point
                         'formula':list(formulae), # Associated formula
                         'centers':km.cluster_centers_.tolist() #Cluster centers
                        })
    return clusters

def do_loco_cv(clusters, data, model, metric, return_score_breakdown=False):
    """Performs LOCO-CV given predefined clusters

    Parameters:
    clusters (python list): Clusters with which to apply LOCO-CV
         in the form [{'k':2, 'formulae':['H2O','NaCl'....],'clusters':[0,1...]},{'k':2...]
    data (pandas DataFrame): data to apply LOCO-CV to
    model (any model that uses sklearn style .fit, .predict interface): the model to evaluate
    metric (function that takes in true and predicted values): metric to evaluate model with
    return_score_breakdown (bool): whether to return per cluster scores as well
         as the overall mean score of the model
    Returns:
    float or float, list if return_score_breakdown: the performance of the model
           (and the associated per cluster scores if return_score_breakdown)

   """
    all_scores = []
    #For each value of k
    for clustering in clusters:

        clustering_scores = []
        labels = pd.Series(clustering['labels'])
        #For each cluster
        for i in range(clustering['k']):
            train_df = data[labels!=i]
            test_df = data[labels==i]

            #Train the model
            train_x = train_df.drop(['target','formula'], axis=1)
            train_y = train_df['target']
            model.fit(train_x, train_y)

            #Make predictions on test set
            test_x = test_df.drop(['target','formula'], axis=1)
            test_y = test_df['target']
            predictions = model.predict(test_x)

            clustering_scores.append(metric(test_y,predictions))
        all_scores.append(clustering_scores)

    per_clustering_means = [sum(x)/len(x) for x in all_scores]
    overall_mean = sum(per_clustering_means)/len(per_clustering_means)

    if return_score_breakdown:
        return overall_mean, per_clustering_means

    return overall_mean



def apply_random_projections(projection_matrix, df, out_file=None):
    """Given a projection matrix and a compvec representation, applies the random
    projection and either returns the result (aligned with targets and formulae)
    or outputs result to file
    Parameters:
    projection_matrix (numpy array): matrix to project with
    df (pandas DataFrame): compvec representation of data set to project
    out_file (None or string): optional place to save the result
    Returns:
    pandas DataFrame or None: resulting projection

   """
    if out_file is not None:
        print("creating file", out_file)
    representation = df.drop('formula',axis=1)
    if 'target' in representation.columns:
        representation = representation.drop('target',axis=1)
    projection = representation.to_numpy() @ projection_matrix
    projection = pd.DataFrame(projection, columns=range(projection.shape[1]))
    projection['formula'] = df['formula']
    if 'target' in df.columns:
        projection['target'] = df['target']
    
    if out_file is None:
        return projection
    
    projection.to_csv(out_file,index=False)

    
def mp_random_projections_helper(args):
    """A helper function for multiprocessing random projection
    Parameters:
    args (tupple in for (args, kwargs)): args and kwargs 
    for apply_random_projections function
    Returns:
    pandas DataFrame or None: resulting projection

   """
    return apply_random_projections(*args[0], **args[1])
    

    
    
def mp_featurisation_helper(args):
    """A helper function for multiprocessing featurisation
    Parameters:
    args (tupple in for (args, kwargs)): args and kwargs 
    for featurise_data function
    Returns:
    pandas DataFrame or None: resulting featurisation

   """
    return featurise_data(*args[0], **args[1])
    
    

def featurise_data(formulae, style='jarvis', target=None, out_file=None):
    """Featurises data, with optional target to align data to

    Parameters:
    formulae (pandas DataFrame): the formulae to featurise
    style ('jarvis', 'compVec','onehot','random_200','magpie', 'oliynyk'):
         the featurisation method to use
    target (pandas Series): target values to be assigned into a column
         (called target) in the featurised data
    out_file (str or None): if string is passed this function will
        save DataFrame to file at this path rather than returning it
    Returns:
    pandas DataFrame of featurised data

   """
    if out_file is not None:
        print('creating',out_file)
    if style.lower() == 'compvec':
        df = generate_features_compVec(formulae, target=target)
        if out_file is None:
            return df
        df.to_csv(out_file, index=False)
        return

    if not isinstance(formulae, pd.DataFrame):
        df = pd.DataFrame(data=formulae, columns=['formula'])
    else:
        df = formulae

    if target is None:
        df['target'] = 0
    else:
        df['target'] = target

    X, Y, formulae = dk.generate_features(df, features_style=style, reset_index=True)
    if target is not None:
        X['target'] = target
    X['formula'] = formulae
    if out_file is None:
            return X
    X.to_csv(out_file, index=False)


#This function is pretty basic but also runs pretty fast because regex is fast
def generate_features_compVec(formulae, target = None):
    """Creates a fractional encoding of input formulae.
    Note: this function does not support formulae containing brackets

    Parameters:
    formulae (pandas DataFrame): the formulae to featurise
    target (pandas Series): target values to be assigned into a column (called target) in the featurised data
    Returns:
    pandas DataFrame of featurised data

   """
    out = []
    for i, formula in enumerate(formulae['formula']):

        matches = element_regex.findall(formula)
        formula_dict = {x:0.0 for x in periodic_table}
        total=0
        for match in matches:
            #Deal with dutereum and tritium
            el = 'H' if match[1] == 'D' or match[1] == 'T'  else match[1]
            if el not in formula_dict:
                print(f'Unkown element {el} in formula {formula}')
                continue
            formula_dict[el] = 1.0 if len(match[2]) == 0 else float(match[2])
            total +=formula_dict[el]
        try:
            formula_dict = {x:formula_dict[x]/total for x in formula_dict}
        except Exception as e:
            print(formula, formula_dict)
            break

        if target is not None:
            formula_dict['target'] = list(target)[i] #cast to a list as this could be a series with non-numeric different indices
        formula_dict['formula'] = formula
        out.append(formula_dict)

    return pd.DataFrame.from_dict(out, orient='columns')
