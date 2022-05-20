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



def featurise_random_projection_file(folder_name, file_name, force_refresh=False):
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
    target = df['target'] if 'target' in df.columns else None
    df = featurise_data(df[['formula']], style='compVec', target=target)
    if 'test' in full_file_name:
        file_suffix='_test_projection.csv'
    elif 'train' in full_file_name:
        file_suffix='_train_projection.csv'
    else:
        file_suffix= '_projection.csv'
    args = []
    for style in featurisation_styles:
        target_file = os.path.join(folder_name,f'{style}{file_suffix}')
        if (not os.path.isfile(target_file)) or force_refresh:
            print('scheduling creation of',target_file)
            args.append(([random_projections[style], df],{'out_file':target_file}))
        else:
            print(target_file, 'exists. Skipping it')
    
    return [pool.map_async(mp_random_projections_helper,args)]



def featurise_CBFV_file(folder_name, file_name, force_refresh=False):
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
        if (not os.path.isfile(target_file)) or force_refresh:
            print('scheduling creation of',target_file)
            args.append(([df[['formula']]], {
                'style':style,
                'target':target,
                'out_file':target_file
            }))
        else:
            print(target_file, 'exists. Skipping it')
    return [pool.map_async(mp_featurisation_helper, args)]
    
    
def featurise_folder(folder_name, force_refresh=False):
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
            async_tasks += featurise_folder(full_file_name, force_refresh=force_refresh)
        if not ('unfeaturised_data.csv' in full_file_name):
            continue        
        if "random_projection" in full_file_name:
            async_tasks += featurise_random_projection_file(folder_name, file, force_refresh=force_refresh)
        else:
            async_tasks += featurise_CBFV_file(folder_name, file, force_refresh=force_refresh)
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
    
                