from Recommender_app.recommender import TwitterRecommender
import pickle
import networkx as nx;

MODEL_PATH = "C:\\Users\\HI\\Desktop\\ProjectX\\models1\\node2vec.model"
GRAPH_PATH = "C:\\Users\\HI\\Desktop\\ProjectX\\models1\\graph.pkl"

def load_data():
    with open(GRAPH_PATH, "rb") as file:
        G = pickle.load(file)
    return G

twitter = TwitterRecommender(MODEL_PATH)
G = load_data()

user_lists = []

for node, data in G.nodes(data=True):
    if data['type']=='user':
        user_lists.append(node)

graph_feature_map = {}

for user_id in user_lists:
    u = list(G.neighbors(user_id))
    if user_id not in graph_feature_map:
        graph_feature_map[user_id] = []
    for node, data in G.nodes(data=True):
        if data['type']=='feature' and node in u and '#' in node:
            graph_feature_map[user_id].append(node)

# def jaccard_similarities():

new_graph_feature_map = {}

for user_id in graph_feature_map.keys():
    if graph_feature_map[user_id]:
        new_graph_feature_map[user_id] = graph_feature_map[user_id].copy()

with open("C:\\Users\\HI\\Desktop\\ProjectX\\models1\\graph_feature_map.pkl", "wb") as file:
    pickle.dump(new_graph_feature_map, file)