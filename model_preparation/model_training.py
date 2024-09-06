from preproc import preproc_data
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
import pickle
import os

random_seed = 42
path = 'C:\\Users\\1\\Downloads\\Amazon Dataset\\amazon_reviews.csv'

amzn_dataset = pd.read_csv(path)
df = preproc_data(amzn_dataset)
print(f'Data set size after preprocessing: {df.shape}')


def filter_data_by_interactions(data, min_user_ratings=10, min_item_ratings=10):
    """
    Filters dataset by number of user and item interactions.
    """
    data = data.groupby('user_id').filter(lambda x: len(x) >= min_user_ratings)
    data = data.groupby('item_id').filter(lambda x: len(x) >= min_item_ratings)
    print(f'dataset was filtered by interactions number. dataset shape: {data.shape}')
    return data


def build_interactions(data):
    """
    Build interaction matrices for LightFM.
    """
    dataset = Dataset()
    dataset.fit(users=data['user_id'].unique(), items=data['item_id'].unique())
    interactions, weights = dataset.build_interactions(
        zip(data['user_id'].values, data['item_id'].values, data['rating'].values)
    )
    user_id_map, _, item_id_map, _ = dataset.mapping()
    return interactions, user_id_map, item_id_map


def get_top_items(data, rating_threshold=4, top_n=20):
    """
    Get top N items based on user ratings.
    """
    filtered_df = data[data['rating'] >= rating_threshold]
    return filtered_df['item_id'].value_counts().head(top_n).index.tolist()


def build_mappings(filtered_df):
    """
    Build user-item mappings for names and ids + known user-item interactions
    """

    user_to_name_map = dict(zip(filtered_df['user_id'], filtered_df['userName']))
    item_to_name_map = dict(zip(filtered_df['item_id'], filtered_df['itemName']))
    known_interactions = filtered_df.groupby('user_id')['item_id'].apply(set).to_dict()

    return user_to_name_map, item_to_name_map, known_interactions


def train_model(data, random_seed=42):
    """
    Train the LightFM model and build mappings for users and items.
    """
    interactions, user_id_map, item_id_map = build_interactions(data)

    model = LightFM(no_components=128,
                    learning_schedule='adagrad',
                    loss='warp',
                    learning_rate=0.02,
                    item_alpha=0.0,
                    user_alpha=0.0,
                    max_sampled=10,
                    random_state=random_seed)

    print('Training started')
    model.fit(interactions, epochs=50, num_threads=1, verbose=True)
    print('Training finished')
    return model, user_id_map, item_id_map


def save_model(directory, model, user_id_map, item_id_map, user_to_name_map, item_to_name_map, known_interactions, top_20_items):
    """
    Save model components (model, mappings, interactions, etc.) to disk in the specified directory.
    Ensures the correct filenames are used for each file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define explicit file paths
    file_paths = {
        'lightfm_model.pkl': model,
        'model_user_id_map.pkl': user_id_map,
        'model_item_id_map.pkl': item_id_map,
        'user_to_name_map.pkl': user_to_name_map,
        'item_to_name_map.pkl': item_to_name_map,
        'user_item_interactions.pkl': known_interactions,
        'top_20_items.pkl': top_20_items
    }

    # Save each component to the directory with the specified filenames
    for file_name, value in file_paths.items():
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(value, file)
        print(f'Saved {file_name} to {file_path}')


filtered_data = filter_data_by_interactions(df)
model, user_id_map, item_id_map = train_model(filtered_data)
top_20_items = get_top_items(filtered_data)
user_to_name_map, item_to_name_map, known_interactions = build_mappings(filtered_data)

save_model('model', model, user_id_map, item_id_map, user_to_name_map,
           item_to_name_map, known_interactions, top_20_items)

