import os
import pandas as pd
import html

path = 'C:\\Users\\1\\Downloads\\Amazon Dataset\\amazon_reviews.csv'

amzn_dataset = pd.read_csv(path)


def preproc_data(dataset):
    df = dataset.copy()
    df = df.drop_duplicates()

    #print(df.shape)

    columns_to_drop = ['verified', 'description', 'image', 'brand', 'feature', 'category', 'price', 'reviewTime',
                       'summary', 'reviewText', 'vote']
    df.drop(columns=columns_to_drop, inplace=True)


    df = df.dropna(subset=['userName', 'itemName'])
    #print(df.shape)

    for i in range(5):
        df.loc[:, 'userName'] = df['userName'].apply(html.unescape)
        df.loc[:, 'itemName'] = df['itemName'].apply(html.unescape)

    bad_value_pattern = r'var aPageStart'
    df.loc[df['itemName'].str.contains(bad_value_pattern,
                                       na=False), 'itemName'] = 'Nike Womens Flex Supreme TR 4 training shoe'

    df = df.drop_duplicates()
    #print(df.shape)

    # Map the userName column to the user_id column
    user_to_id_map = {name: idx for idx, name in enumerate(df['userName'].unique())}
    df['user_id'] = df['userName'].map(user_to_id_map)

    # Map the itemName column to the item_id column
    items_to_id_map = {name: idx for idx, name in enumerate(df['itemName'].unique())}
    df['item_id'] = df['itemName'].map(items_to_id_map)

    return df