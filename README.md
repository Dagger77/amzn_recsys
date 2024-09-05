# amzn_recsys
EDA/MODELING/FLASK APP based on Amazon Reviews 2018 dataset

##Report of analysis and model development
### Description
Located in the folder 'eda_notebooks'
1. Amazon_Reviews_2018_Cleaning.ipynb 				- Preliminary dataset cleaning
2. Amazon_Reviews_2018_EDA.ipynb 					- Exploratory Data Analysis
3. Amazon_Reviews_2018_Feature_extraction.ipynb		- Notebook with feature extraction 
4. Amazon_Reviews_2018_Modeling.ipynb				- Notebook with model selection and evaluation


##Recommender App
### Description
This application serves as a recommendation engine using the pre-trained LightFM model. The app provides personalized recommendations for known users and general top item recommendations for unknown users.

### Install

####Clone the Repository: 
git clone https://github.com/Dagger77/amzn_recsys.git

####Install Dependencies: 
Navigate to the project directory and install the necessary dependencies using pip
cd amzn_recsys
pip install -r requirements.txt

####Ensure Model and Mappings Are Available.
The application relies on the following pre-trained models and mappings (ensure these files are in the model/ directory):

lightfm_model.pkl: Pre-trained LightFM model.
model_user_id_map.pkl: User ID to internal model ID mapping.
model_item_id_map.pkl: Item ID to internal model ID mapping.
user_to_name_map.pkl: User ID to user name mapping.
item_to_name_map.pkl: Item ID to item name mapping.
user_item_interactions.pkl: Known user-item interactions.
top_20_items.pkl: Precomputed top item recommendations.

### Run the Application
Start the Flask application by running:
python amzn_recsys_app.py

### Usage
1. Recommendations for a User
To get personalized recommendations for a known user, send a GET request to:
curl http://127.0.0.1:5000/recommend/<user_id>
Replace <user_id> with the user’s ID.

Example of response:
{
    "user_id": "123",
    "user_name": "John Doe",
    "recommendations": ["item_id_22", "item_id_45", "item_id_80"],
    "recommendations_ext": [
        {"item_id": "item_id_22", "item_name": "Camera 123"},
        {"item_id": "item_id_45", "item_name": "Laptop XYZ"},
        {"item_id": "item_id_80", "item_name": "Monitor ABC"}
    ]
}

2. Get a Sample of Users
To retrieve a sample of user IDs and their corresponding names:
curl http://127.0.0.1:5000/user_sample

Example of response:
[
    {"123": "John Doe"},
    {"124": "Jane Smith"},
    {"125": "Alice Johnson"}
]

### Additional Notes
Cold user Handling: If a user is not found in the mapping, the application returns the top global items recommendation.
Debug Mode: The application runs in debug mode by default (debug=True).
