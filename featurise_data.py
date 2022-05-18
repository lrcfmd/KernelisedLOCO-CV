from utilities import mp_featurisation_helper, featurise_data, mp_random_projections_helper
import os
import pandas as pd
import multiprocessing
import numpy as np
import time
pool = multiprocessing.Pool(multiprocessing.cpu_count())

rp_locations = 'data/random_projections'
featurisation_styles = ['jarvis','compVec','onehot','magpie','oliynyk','random_200']

#Random projections are small enough I'm happy to leave them in ram to save reading them in repeatedly
random_projections = {style: np.genfromtxt(os.path.join(rp_locations,f'{style}_projection.csv'), delimiter=',') for style in featurisation_styles}

def apply_random_projections(folder_name, df,file_suffix="_projection.csv"):
    
    representation = featurise_data(df[['formula']], style='compVec').drop('formula',axis=1).to_numpy()
    for style in featurisation_styles:
        print("Applying random projection",style)
        projection = representation @ random_projections[style]
        projection = pd.DataFrame(projection.T, columns=range(projection.shape[0]))
        projection['formula'] = df['formula']
        if 'target' in df.columns:
            projection['target'] = df['target']
            
        target_file = os.path.join(folder_name,f'{style}{file_suffix}')
        projection.to_csv(target_file,index=False)
    

def featurise_random_projection_file(folder_name, file_name):
    """Given a folder and a file in that folder, featurises that 
    file into every available size of random projection
    Parameters:
    folder_name (string): path to folder
    file_name (string): name of file
    returns:
    list(MapResult) object of async tasks
   """
    
    full_file_name = os.path.join(folder_name, file_name)
    df = pd.read_csv(full_file_name)
    
    df = featurise_data(df[['formula']], style='compVec')
    if 'test' in full_file_name:
        file_suffix='_test_projection.csv'
    elif 'train' in full_file_name:
        file_suffix='_train_projection.csv'
    else:
        file_suffix= '_projection.csv'
    args = []
    for style in featurisation_styles:
        target_file = os.path.join(folder_name,f'{style}{file_suffix}')
        args.append(([random_projections[style], df],{'out_file':target_file}))
    
    return [pool.map_async(mp_random_projections_helper,args)]



def featurise_CBFV_file(folder_name, file_name):
    """Given a folder and a file in that folder, featurises that 
    file into every available CBFV style
    Parameters:
    folder_name (string): path to folder
    file_name (string): name of file
    returns:
    list(MapResult) object of async tasks
   """
    
    full_file_name = os.path.join(folder_name, file_name)
    df = pd.read_csv(full_file_name)
    
    target = df['target'] if 'target' in df.columns else None
    args = []
    for style in featurisation_styles:
        if 'test' in full_file_name:
            suffix = '_test_CBFV.csv'
        elif 'train' in full_file_name:
            suffix = '_train_CBFV.csv'
        else:
            suffix = '_CBFV.csv'
            
        target_file = os.path.join(folder_name, f'{style}{suffix}')
        args.append(([df[['formula']]], {
            'style':style,
            'target':target,
            'out_file':target_file
        }))
    return [pool.map_async(mp_featurisation_helper, args)]
    
    
def featurise_folder(folder_name):
    """featurises data in this folder and all subfolders
    Parameters:
    folder_name (string): path to folder
    returns:
    pandas None
   """
    print('Featurising folder', folder_name)
    async_tasks = []
    for file in os.listdir(folder_name):
        full_file_name = os.path.join(folder_name, file)
        if os.path.isdir(full_file_name):
            #recursive call
            async_tasks += featurise_folder(full_file_name)
        if not ('unfeaturised_data.csv' in full_file_name):
            continue
        
        if 'random_projection' in full_file_name:
            async_tasks += featurise_random_projection_file(folder_name, file)
        else:
            async_tasks += featurise_CBFV_file(folder_name, file)
    return async_tasks
            
async_tasks = featurise_folder('data')
while (True):
    all_finished = True
    for task in async_tasks:
        all_finished = all_finished and task.ready()
        if not all_finished:
            break
    if all_finished:
        break
    print('Featurising is still processing this can take an hour or so')
    time.sleep(30)
    
                