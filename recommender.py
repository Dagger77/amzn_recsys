import numpy as np
from typing import Dict, Set, List, Any
from pydantic import BaseModel, conint, ValidationError
from utils import pickle_load

# Load pre-trained models and mappings
model_path = 'model\\lightfm_model.pkl'
user_id_mapping_path = 'model\\model_user_id_map.pkl'
item_id_mapping_path = 'model\\model_item_id_map.pkl'
user_name_mapping_path = 'model\\user_to_name_map.pkl'
item_name_mapping_path = 'model\\item_to_name_map.pkl'
known_interactions_path = 'model\\user_item_interactions.pkl'
top_items_path = 'model\\top_20_items.pkl'


class RecommendationInput(BaseModel):
    user_id: conint(gt=0, lt=1000000)
    num_recommendations: conint(gt=0, le=100) = 5


class Recommender:
    def __init__(self):
        self.lfm_model: Any = pickle_load(model_path)
        self.user_id_mapping: Dict[int, int] = pickle_load(user_id_mapping_path)
        self.item_id_mapping: Dict[int, int] = pickle_load(item_id_mapping_path)
        self.item_id_inverse_mapping = {v: k for k, v in self.item_id_mapping.items()}
        self.user_name_mapping: Dict[int, str] = pickle_load(user_name_mapping_path)
        self.item_name_mapping: Dict[int, str] = pickle_load(item_name_mapping_path)
        self.known_interactions: Dict[int, Set[int]] = pickle_load(known_interactions_path)
        self.top_items: List[int] = pickle_load(top_items_path)

    def recommend_items(self, user_id: int, num_recommendations: int = 5):
        try:
            inputs = RecommendationInput(user_id=user_id, num_recommendations=num_recommendations)
        except ValidationError as e:
            return {"error": "Input validation failed", "details": e.errors()}, 400

        validated_user_id = inputs.user_id
        validated_num_recommendations = inputs.num_recommendations

        try:
            model_user_id = self.user_id_mapping.get(validated_user_id, None)

            if model_user_id is None:
                print('User unknown. Most popular items were recommended.')
                return self.top_items[:validated_num_recommendations]

            n_items = len(self.item_id_mapping)
            scores = self.lfm_model.predict(model_user_id, np.arange(n_items))

            # Exclude items the user has already interacted with
            known_items = self.known_interactions.get(validated_user_id, set())
            known_item_indices = np.array([self.item_id_mapping[item] for item in known_items if item in self.item_id_mapping])
            mask = np.ones(n_items, dtype=bool)
            mask[known_item_indices] = False

            filtered_scores = scores[mask]
            filtered_item_indices = np.arange(n_items)[mask]

            top_filtered_indices = np.argsort(-filtered_scores)[:validated_num_recommendations]
            top_item_ids = [self.item_id_inverse_mapping[filtered_item_indices[i]] for i in top_filtered_indices]

            return top_item_ids
        except Exception as e:
            print(f"Error during recommendation for user {user_id}: {e}")
            return {"error": "An error occurred while processing the recommendation."}, 500