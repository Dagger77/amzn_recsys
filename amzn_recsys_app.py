from flask import Flask, request, jsonify
import numpy as np
import pickle
import random
from typing import Dict, Set, List, Any
from pydantic import BaseModel, conint, ValidationError

model_path = 'model\\lightfm_model.pkl'
user_id_mapping_path = 'model\\model_user_id_map.pkl'
item_id_mapping_path = 'model\\model_item_id_map.pkl'
user_name_mapping_path = 'model\\user_to_name_map.pkl'
item_name_mapping_path = 'model\\item_to_name_map.pkl'
known_interactions_path = 'model\\user_item_interactions.pkl'
top_items_path = 'model\\top_20_items.pkl'


def pickle_load(path: str) -> Any:
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError(f"Error loading the pickle file: {path}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading {path}: {e}")


class RecommendationInput(BaseModel):
    user_id: conint(gt=0, lt=1000000)
    num_recommendations: conint(gt=0, le=100) = 5


class RecommendationApp:
    def __init__(self):
        # Load pre-trained LightFM model, mappings, known interactions and top items
        self.lfm_model: Any = pickle_load(model_path)

        self.user_id_mapping: Dict[int, int] = pickle_load(user_id_mapping_path)
        self.item_id_mapping: Dict[int, int] = pickle_load(item_id_mapping_path)
        self.item_id_inverse_mapping = {v: k for k, v in self.item_id_mapping.items()}

        self.user_name_mapping: Dict[int, str] = pickle_load(user_name_mapping_path)
        self.item_name_mapping: Dict[int, str] = pickle_load(item_name_mapping_path)

        self.known_interactions: Dict[int, Set[int]] = pickle_load(known_interactions_path)
        self.top_items: List[int] = pickle_load(top_items_path)

        # Initialize Flask app
        self.app = Flask(__name__)
        self.add_routes()

    def recommend_items(self, user_id: int, num_recommendations: int = 5):

        try:
            inputs = RecommendationInput(user_id=user_id, num_recommendations=num_recommendations)
        except ValidationError as e:
            return {"error": "Input validation failed", "details": e.errors()}, 400

        # Use the validated inputs
        validated_user_id = inputs.user_id
        validated_num_recommendations = inputs.num_recommendations

        try:
            model_user_id = self.user_id_mapping.get(validated_user_id, None)

            if model_user_id is None:
                # If the user is unknown, return the precomputed top global items
                print(f"Unknown user {validated_user_id}. Returning top global items.")
                return self.top_items[:validated_num_recommendations]

            n_items = len(self.item_id_mapping)
            scores = self.lfm_model.predict(model_user_id, np.arange(n_items))

            # Excluding items that users already interacted with
            known_items = self.known_interactions.get(validated_user_id, set())
            known_item_indices = np.array(
                [self.item_id_mapping[item] for item in known_items if item in self.item_id_mapping])
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

    def add_routes(self):
        # Define the recommendation route
        @self.app.route('/recommend/<user_id>', methods=['GET'])
        def recommend(user_id):

            try:
                user_id = int(user_id)
            except ValueError:
                return jsonify({"error": "Invalid user_id. It should be an integer."}), 400

            recommendations = self.recommend_items(user_id)

            # Check if recommendations is an error response
            if isinstance(recommendations, tuple):
                return jsonify(recommendations[0]), recommendations[1]

            user_name = self.user_name_mapping.get(user_id, "Unknown User")

            recommended_item_names = [
                {"item_id": item_id, "item_name": self.item_name_mapping.get(item_id, "Unknown Item")}
                for item_id in recommendations
            ]

            return jsonify({
                "user_id": user_id,
                "user_name": user_name,
                'recommendations': recommendations,
                "recommendations_ext": recommended_item_names
            })

        # Define a route to get a sample of model's user_id: userName pairs
        @self.app.route('/user_sample', methods=['GET'])
        def get_user_sample():
            try:
                # Get a sample of user_id: userName pairs
                sample_size = 10
                user_sample = random.sample(list(self.user_name_mapping.items()), sample_size)
                return jsonify(user_sample)
            except Exception as e:
                print(f"Error retrieving user sample: {e}")
                return {"error": "An error occurred while retrieving the user sample."}, 500

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    recommendation_app = RecommendationApp()
    recommendation_app.run()
