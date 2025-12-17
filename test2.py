import pickle
from typing import List
import numpy as np

GRAPH_FEATURE_MAP_PATH = "C:\\Users\\HI\\Desktop\\ProjectX\\models1\\graph_feature_map.pkl"

def load():
    with open(GRAPH_FEATURE_MAP_PATH, "rb") as file:
        mapping = pickle.load(file=file)
    return dict(mapping)

def jaccard_similarity(new_user_feature, mapping, top_most:int=5):
    
    similarity = []
    new_user_set = np.array(new_user_feature)

    for user in mapping.keys():
        all_user_set = np.array(mapping[user])
        common = np.intersect1d(all_user_set, new_user_set, assume_unique=True)
        total = np.union1d(all_user_set, new_user_set)
        
        jaccard_score = len(common)/len(total)
        
        similarity.append((user, jaccard_score))
    
    similarity.sort(key=lambda x : x[1], reverse=True)
    
    return similarity[0:top_most]
        

if __name__=="__main__":
    all_user_features = load()
    new_user_features = [
        'Feat: #FF',
        'Feat: #hayleywilliams'
    ]
    
    jaccard_similarity(new_user_feature=new_user_features, mapping=all_user_features)