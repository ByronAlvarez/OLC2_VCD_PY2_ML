import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import io
import base64


def poly_reg(degrees, x_train, y_train, model_curve):
    features = PolynomialFeatures(degree=degrees)
    x_train_transformed = features.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_train_transformed, y_train)

    train_pred = model.predict(x_train_transformed)
    rmse_poly_train = mean_squared_error(y_train, train_pred, squared=False)

    return [rmse_poly_train]


def getPrediction(date, country, cases, countryName, csv, tiempoPred):
    col_list = [date, country, cases]

    dataset = pd.DataFrame(csv, columns=col_list)

    dataset[cases] = pd.to_numeric(dataset[cases])

    xx = dataset.loc[dataset[country] == countryName]
    xx[date] = xx[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)
    X = xx.iloc[:, pos1].values
    X = X.reshape(-1, 1)
    y = xx.iloc[:, pos2].values

    fig = plt.figure()

    plt.scatter(X, y, color="red")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Ploteo entre X y')

    flike = io.BytesIO()
    fig.savefig(flike)
    graph1 = base64.b64encode(flike.getvalue()).decode()

    model = LinearRegression()
    model.fit(X, y)

    model_curve = model.predict(X)
    fig2 = plt.figure()
    plt.plot(X[:, 0], y, 'ko')
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresión Lineal')

    flike = io.BytesIO()
    fig2.savefig(flike)
    graph2 = base64.b64encode(flike.getvalue()).decode()

    features = PolynomialFeatures(degree=4)

    x_train_transformed = features.fit_transform(X)

    model2 = LinearRegression()
    model2.fit(x_train_transformed, y)

    x_true_transformed = features.fit_transform(X)
    model_curve = model2.predict(x_true_transformed)

    fig3 = plt.figure()
    plt.plot(X[:, 0], y, 'bo')
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresión Polinomial entre grados 4')

    flike = io.BytesIO()
    fig3.savefig(flike)
    graph3 = base64.b64encode(flike.getvalue()).decode()

    errors = []
    for i in range(26):
        errors.append([i] + poly_reg(i, X, y, model_curve))

    df = pd.DataFrame(errors, columns=['grado', 'RMSE'])

    fig4 = plt.figure()

    plt.plot(df['grado'], df['RMSE'], 'o-r', label='Valores RMSE')
    plt.xlabel('Grado')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE para diferentes grados de la Regresión Polinomial')
    flike = io.BytesIO()
    fig4.savefig(flike)
    graph4 = base64.b64encode(flike.getvalue()).decode()

    bestdegree = np.argmin(errors, axis=0)[1]
    features2 = PolynomialFeatures(degree=bestdegree)

    x_train_transformed2 = features2.fit_transform(X)

    model3 = LinearRegression()
    model3.fit(x_train_transformed2, y)

    x_true_transformed2 = features2.fit_transform(X)
    model_curve3 = model3.predict(x_true_transformed2)
    mse = mean_squared_error(y_true=y, y_pred=model_curve3)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true=y, y_pred=model_curve3)
    coefs = model3.coef_

    coefs = model3.coef_
    eq = "y = " + str(model3.intercept_)
    for i in range(len(coefs)):
        if i != 0:
            eq += " +(" + str(coefs[i])+")x^"+str(i)

    fig5 = plt.figure()
    plt.plot(X[:, 0], y, 'bo')
    plt.plot(X[:, 0], model_curve3, 'r-', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresión Polinomial de mejor grado '+str(bestdegree))
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    # Predictions ------------------------------------------------------

    featuresP = PolynomialFeatures(degree=bestdegree)

    modelP = LinearRegression()

    x_pred = np.arange(0, len(y)+tiempoPred)
    x_pred = x_pred[:, np.newaxis]

    x_true_transformedP = featuresP.fit_transform(x_pred)
    model_curveP = modelP.predict(x_true_transformedP)

    fig6 = plt.figure()
    plt.plot(X[:, 0], y, 'bo')
    plt.plot(x_pred[:, 0], model_curveP, 'r-', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de prediccion en: ' + str(tiempoPred))
    flike = io.BytesIO()
    fig6.savefig(flike)
    graph6 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5, graph6], errors, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], eq
