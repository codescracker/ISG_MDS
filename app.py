from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
from sklearn.metrics import euclidean_distances
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# /home/shaowei/PycharmProjects/ISG_MDS/input/wine.csv
@app.route('/data')
def get_data():
    df = pd.read_csv('/home/shaowei/PycharmProjects/ISG_MDS/input/wine.csv', names=['Class', 'Alcohol', 'Malic acid',
                                          'Ash', 'Alcalinity of ash ', 'Magnesium',
                                          'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                                          'Proanthocyanins', 'Color intensity', 'Hue',
                                          'OD280/OD315 of diluted wines', 'Proline'
                                          ])

    colu = df.columns[1:]
    predt = df[colu]
    scale_dt = StandardScaler().fit_transform(predt)
    scale_df = pd.DataFrame(scale_dt, columns=colu)
    classcolu = df['Class']
    ndt = pd.concat([classcolu, scale_df], axis=1, join_axes=[df.index])

    similarities = euclidean_distances(scale_df)
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                       random_state=seed, dissimilarity="precomputed", n_jobs=1)

    pos = mds.fit(similarities).embedding_
    rdt = pd.DataFrame(pos, columns=['x', 'y'])

    rs = pd.concat([ndt, rdt], axis=1, join_axes=[df.index])

    pdt = rs[['Class', 'x', 'y']]

    df_json = pdt.reset_index().to_json(orient='records')

    return df_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
