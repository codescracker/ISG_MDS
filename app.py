from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
from distance import *
import numpy as np
import json

# /home/shaowei/PycharmProjects/ISG_MDS/input/wine.csv
def loadData():
    # Read the wine data into pandas dataframe, and name the feature
    df = pd.read_csv('/home/shaowei/PycharmProjects/ISG_MDS/input/wine.csv', names=['Class', 'Alcohol', 'Malic acid',
                                                                                    'Ash', 'Alcalinity of ash ',
                                                                                    'Magnesium',
                                                                                    'Total phenols', 'Flavanoids',
                                                                                    'Nonflavanoid phenols',
                                                                                    'Proanthocyanins', 'Color intensity',
                                                                                    'Hue',
                                                                                    'OD280/OD315 of diluted wines',
                                                                                    'Proline'
                                                                                    ])
    # Extract the name of features
    columnName = df.columns[1:]
    # Extract the feature from pandas dataframe according to the feature names
    dt4Scale = df[columnName]
    # StandardScaler the data
    scaled_dt = StandardScaler().fit_transform(dt4Scale)
    # Store the scaled data into pandas dataframe
    scaled_df = pd.DataFrame(scaled_dt, columns=columnName)
    # Extract the class of records
    className = df['Class']
    # Combine the class and features according to the index
    df_clean = pd.concat([className, scaled_df], axis=1, join_axes=[df.index])
    # Measure the similarity of records using euclidean distance
    WEIGHT_INITIAL = np.array([1/len(scaled_dt[0]) for i in range(0, len(scaled_dt[0]))])
    similarities = dist_func(scaled_df, WEIGHT_INITIAL)
    seed = np.random.RandomState(seed=3)
    # set up the parameter of MDS
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                       random_state=seed, dissimilarity="precomputed", n_jobs=1)
    # Reduce the dimensions of dataset to 2 dimensions
    dt_2Dims = mds.fit(similarities).embedding_
    # Store the dimensions reduced data in pandas dataframe
    df_2Dims = pd.DataFrame(dt_2Dims, columns=['x', 'y'])
    # Combine and store all useful features
    dt_allFeatures = pd.concat([df_clean, df_2Dims], axis=1, join_axes=[df.index])
    # Extract features for Visualization
    df4Vis = dt_allFeatures[['Class', 'x', 'y']]
    # Convert the data from dataframe to standard Json
    df_json = df4Vis.to_json(orient='records')
    return df_json

dt_new = None
dt_json = loadData()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def get_data():
    if dt_new:
        return dt_new
    return dt_json


@app.route('/post', methods=['POST'])
def post():
    json_dic = request.json
    global dt_new
    dt_new = json.dumps(json_dic)
    return jsonify(({
            "status": "success",
            "message": "Your post is successful"
        }))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
