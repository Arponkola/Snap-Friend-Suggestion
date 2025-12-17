from gensim.models import Word2Vec
import numpy as np
import pickle


class TwitterRecommender:
    def __init__(self, model_path):
        # Load the pre-trained model
        self.model = Word2Vec.load(model_path)
        
    def find_friends_for_existing_user(self, user_id, all_neighbors, top_n=5):
        """
        Finds similar users based on graph structure.
        """
        if user_id not in self.model.wv:
            return []
            
        # Get most similar nodes
        similar_nodes = self.model.wv.most_similar(user_id, topn=top_n+50)
        
        # Filter: We only want to recommend Users, not Features (hashtags)
        recommendations = []
        for node, score in similar_nodes:
            if "Feat:" not in node and node != user_id and node not in all_neighbors:
                recommendations.append((node, score))
                if len(recommendations) >= top_n:
                    break
        return recommendations

    def cold_start_recommendation(self, interest_list, top_n=5):
        """
        The 'Vector Averaging' Strategy.
        interest_list: ['Feat: Tech', 'Feat: Politics']
        """
        valid_vectors = []
        for interest in interest_list:
            if interest in self.model.wv:
                valid_vectors.append(self.model.wv[interest])
        
        if not valid_vectors:
            return []

        # Calculate the "Centroid" (Average Vibe)
        user_vector = np.mean(valid_vectors, axis=0)
        
        # Find closest users to this synthetic vector
        similar_nodes = self.model.wv.similar_by_vector(user_vector, topn=top_n+10)
        
        recommendations = []
        for node, score in similar_nodes:
            if "Feat:" not in node: # Only recommend people
                recommendations.append((node, score))
                if len(recommendations) >= top_n:
                    break
        return recommendations
    
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
        