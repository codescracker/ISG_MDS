from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
from distance import *
import numpy as np
from scipy.optimize import minimize
import time


# /home/shaowei/PycharmProjects/ISG_MDS/input/wine.csv
def getOriginalDf():
    # Read the wine data into pandas dataframe, and name the feature
    df = pd.read_csv('./input/wine_clean.csv')
    return df


def getScaledDt(df):
    # Extract the name of features
    columnName = df.columns[2:]
    # Extract the feature from pandas dataframe according to the feature names
    dt4Scale = df[columnName]
    # StandardScaler the data
    scaled_dt = StandardScaler().fit_transform(dt4Scale)
    return scaled_dt


# function that reduce the dimensionality to 2 dimensions in np.array format
def get2DimsDt(dt, weight):
    similarity = dist_func(dt, weight)
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                       random_state=seed, dissimilarity="precomputed", n_jobs=1)
    dt_2Dims = mds.fit(similarity).embedding_
    # print(dt_2Dims)
    return dt_2Dims


# return json
def get2DimDtWithClass(dt_2dims, df):
    df_2dims = pd.DataFrame(dt_2dims, columns=['x', 'y'])
    classname = df[["Class","id"]]
    df_withclass = pd.concat([df_2dims, classname], axis=1)
    return df_withclass.to_json(orient="records")


# return numpy array
def format2dtWithoutClass(post_dict):
    res = [[point["x"], point["y"]] for point in post_dict]
    return np.array(res)

WEIGHT_4_2DIMS = [0.5, 0.5]
DF = getOriginalDf()
DT = getScaledDt(DF)
INITIAL_WEIGHT = np.array([1/len(DT[0]) for i in range(0, len(DT[0]))])

DT_2Dims_before = get2DimsDt(DT, INITIAL_WEIGHT)
DT_2Dims_after = None


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/highDims_data')
def get_highDims_data():
    return DF.to_json(orient='records')


@app.route('/weight_data', methods=['GET'])
def get_weight():
    return jsonify(list(INITIAL_WEIGHT))


@app.route('/data')
def get_data():
    return get2DimDtWithClass(DT_2Dims_before, DF)


@app.route('/post', methods=['POST'])
def post():
    json_dic = request.json
    global DT_2Dims_after
    DT_2Dims_after = format2dtWithoutClass(json_dic)

    print("Post DT_2Dims_after", DT_2Dims_after)
    return jsonify(({
            "status": "success",
            "message": "Your post is successful"
        }))


@app.route('/calculate_weight', methods=['GET'])
def new_weight():
    global DT_2Dims_after
    global DT_2Dims_before
    global WEIGHT_4_2DIMS
    global INITIAL_WEIGHT

    if DT_2Dims_after is None:
        print("DT_2Dims_after:\n", DT_2Dims_after)
        return jsonify({
            "message": "Please post the new position first"
        })
    else:
        dist_after = dist_func(DT_2Dims_after, WEIGHT_4_2DIMS)
        dist_before = dist_func(DT_2Dims_before, WEIGHT_4_2DIMS)
        print("dist_after:", dist_after)
        print("dist_before:", dist_before)
        u = umatrix(dist_after, dist_before)
        l = lmatrix(u)
        print('u:', u)
        print('l:', l)

        def object_function(x, sign=1.0):
            n = len(DT)
            m = len(DT[0])
            UMatrix = u["U"]
            target = 0
            objective_weight = np.array([x[i] for i in range(0, m)])
            for i in range(0, n):
                for j in range(i + 1, n):
                    target += l[i][j] * (
                        (weightedL2(DT[i], DT[j], objective_weight)
                         - UMatrix[i][j] * weightedL2(DT[i], DT[j], INITIAL_WEIGHT)
                         ) ** 2)
            return sign * target

        def object_func_drive(x, sign=1.0):
            n = len(DT)
            m = len(DT[0])
            UMatrix = u["U"]
            targets = []
            objective_weight = np.array([x[i] for i in range(0, m)])
            for index in range(0, m):
                target = 0
                for i in range(0, n):
                    for j in range(i + 1, n):
                        part1 = l[i][j] * (
                            weightedL2(DT[i], DT[j], objective_weight)
                            - UMatrix[i][j] * weightedL2(DT[i], DT[j], INITIAL_WEIGHT)
                        )

                        part2 = (DT[i][index] - DT[j][index]) ** 2

                        target += sign * (part1 * part2)

                targets.append(2 * target)

            return np.array(targets)

        def constrain(x):
            m = len(DT[0])
            constrain = 0
            for i in range(0, m):
                constrain += x[i]
            constrain = constrain - 1
            return np.array([constrain])

        def constrain_jac(x):
            m = len(DT[0])
            driv = [1] * m
            return np.array(driv)

        cons = ({
                    'type': 'eq',
                    'fun': constrain,
                    'jac': constrain_jac
                },)

        bnds = tuple([(0, None)] * len(DT[0]))

        print("Ready to start the iteration process of getting the optimized weight")
        start_time = time.time()
        res = minimize(object_function, list(INITIAL_WEIGHT), jac=object_func_drive, bounds=bnds,
                       constraints=cons, method='SLSQP', options={'disp': True, 'maxiter': 5000})
        print("It costs", time.time()-start_time, "seconds to get optimized solution")
        weight_new = res.x
        print("New Weight", weight_new)
        INITIAL_WEIGHT = weight_new

        DT_2Dims_before = get2DimsDt(DT, INITIAL_WEIGHT)
        print("New DT_2dims_before", DT_2Dims_before)
        DT_2Dims_after = None
        return jsonify({
            "message": "Get the new weight"
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
