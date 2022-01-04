import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from scipy.sparse.construct import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import io
import base64
import random


class Country:
    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y
        self.errors = []
        self.df = None
        self.bestdegree = 0
        self.metrics = []
        self.eq = ""
        self.model_curve = None


def poly_reg(degrees, x_train, y_train, model_curve):
    features = PolynomialFeatures(degree=degrees)
    x_train_transformed = features.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_train_transformed, y_train)

    train_pred = model.predict(x_train_transformed)
    rmse_poly_train = mean_squared_error(y_train, train_pred, squared=False)

    return [rmse_poly_train]


def getComparacion(date, country, cases, deaths, countryName, csv):
    col_list = [date, country, cases, deaths]
    colors = random.choices(
        list(mcolors.CSS4_COLORS.values()), k=140)

    dataset = pd.DataFrame(csv, columns=col_list)
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[deaths], inplace=True)
    dataset.dropna(subset=[cases], inplace=True)
    #dataset.fillna(0, inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])
    dataset[deaths] = pd.to_numeric(dataset[deaths])

    xx = dataset.loc[dataset[country] == countryName]
    auxLabel = pd.DatetimeIndex(xx[date])
    xx[date] = xx[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)
    pos3 = dataset.columns.get_loc(deaths)
    X = xx.iloc[:, pos1].values
    X = X.reshape(-1, 1)
    y = xx.iloc[:, pos2].values
    z = xx.iloc[:, pos3].values

    fig = plt.figure()
    plt.style.use('dark_background')

    plt.scatter(auxLabel, y, color=random.choice(
        colors), alpha=0.5, label=cases)
    plt.scatter(auxLabel, z, color=random.choice(
        colors), alpha=0.5, label=deaths)
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Ploteo entre X y')
    plt.xticks(rotation=45)
    plt.tight_layout()

    flike = io.BytesIO()
    fig.savefig(flike)
    graph1 = base64.b64encode(flike.getvalue()).decode()

    model = LinearRegression()
    model.fit(X, y)

    model_curve = model.predict(X)
    # Doouble--------------
    modelDD = LinearRegression()
    modelDD.fit(X, z)

    model_curveDD = modelDD.predict(X)

    mseL = mean_squared_error(y_true=y, y_pred=model_curve)
    rmseL = np.sqrt(mseL)
    r2L = r2_score(y_true=y, y_pred=model_curve)
    # -----------------------------
    mseLD = mean_squared_error(y_true=z, y_pred=model_curveDD)
    rmseLD = np.sqrt(mseLD)
    r2LD = r2_score(y_true=z, y_pred=model_curveDD)

    fig2 = plt.figure()
    plt.plot(X, y, 'o', color=random.choice(colors))
    plt.plot(X, z, 'o', color=random.choice(colors))
    plt.plot(X, model_curve, '-',
             color=random.choice(colors), linewidth=3, label=str(cases))
    plt.plot(X[:, 0], model_curveDD, '-',
             color=random.choice(colors), linewidth=3, label=str(deaths))
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
    plt.title('Modelo de Regresión Lineal')
    plt.xticks(rotation=45)
    plt.tight_layout()

    flike = io.BytesIO()
    fig2.savefig(flike)
    graph2 = base64.b64encode(flike.getvalue()).decode()

    features = PolynomialFeatures(degree=4)

    x_train_transformed = features.fit_transform(X)

    model2 = LinearRegression()
    model2.fit(x_train_transformed, y)

    x_true_transformed = features.fit_transform(X)
    model_curve = model2.predict(x_true_transformed)

    # Double ----------------------------
    model2DD = LinearRegression()
    model2DD.fit(x_train_transformed, z)
    model_curveDD = model2DD.predict(x_true_transformed)

    fig3 = plt.figure()
    plt.plot(auxLabel, y, 'o', color=random.choice(colors))
    plt.plot(auxLabel, z, 'o', color=random.choice(colors))
    plt.plot(auxLabel, model_curve, '-',
             color=random.choice(colors), linewidth=3, label=str(cases))
    plt.plot(auxLabel, model_curveDD, '-',
             color=random.choice(colors), linewidth=3, label=str(deaths))
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de Regresión Polinomial entre grados 4')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    flike = io.BytesIO()
    fig3.savefig(flike)
    graph3 = base64.b64encode(flike.getvalue()).decode()

    errors = []
    errors2 = []
    for i in range(26):
        errors.append([i] + poly_reg(i, X, y, model_curve))
        errors2.append([i] + poly_reg(i, X, z, model_curveDD))

    df = pd.DataFrame(errors, columns=['grado', 'RMSE'])
    dfDD = pd.DataFrame(errors2, columns=['grado', 'RMSE'])

    fig4 = plt.figure()

    plt.plot(df['grado'], df['RMSE'], 'o-',
             color=random.choice(colors), label='Valores RMSE')
    plt.plot(dfDD['grado'], dfDD['RMSE'], 'o-',
             color=random.choice(colors), label='Valores RMSE')
    plt.xlabel('Grado')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE para diferentes grados de la Regresión Polinomial')

    flike = io.BytesIO()
    fig4.savefig(flike)
    graph4 = base64.b64encode(flike.getvalue()).decode()

    bestdegree = np.argmin(errors, axis=0)[1]
    bestdegree2 = np.argmin(errors2, axis=0)[1]
    features2 = PolynomialFeatures(degree=bestdegree)

    x_train_transformed2 = features2.fit_transform(X)

    model3 = LinearRegression()
    model3.fit(x_train_transformed2, y)

    x_true_transformed2 = features2.fit_transform(X)
    model_curve3 = model3.predict(x_true_transformed2)

    # Double ------------------------
    model3DD = LinearRegression()
    model3DD.fit(x_train_transformed2, z)
    model_curve3DD = model3DD.predict(x_true_transformed2)
    # -------

    mse = mean_squared_error(y_true=y, y_pred=model_curve3)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true=y, y_pred=model_curve3)
    coefs = model3.coef_
    # --------
    mseD = mean_squared_error(y_true=z, y_pred=model_curve3DD)
    rmseD = np.sqrt(mseD)
    r2D = r2_score(y_true=z, y_pred=model_curve3DD)
    coefsD = model3DD.coef_
    # -----------------------------------------

    coefs = model3.coef_
    eq = "y = " + str(model3.intercept_)
    for i in range(len(coefs)):
        if i != 0:
            eq += " +(" + str(coefs[i])+")x^"+str(i)

    eqD = "y = " + str(model3DD.intercept_)
    for i in range(len(coefsD)):
        if i != 0:
            eqD += " +(" + str(coefsD[i])+")x^"+str(i)

    fig5 = plt.figure()
    plt.plot(auxLabel, y, 'o', color=random.choice(colors))
    plt.plot(auxLabel, z, 'o', color=random.choice(colors))
    plt.plot(auxLabel, model_curve3, '-',
             color=random.choice(colors), linewidth=3, label=str(cases))
    plt.plot(auxLabel, model_curve3DD, '-',
             color=random.choice(colors), linewidth=3, label=str(deaths))
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de Regresión Polinomial y Lineal de mejor grado ' +
              str(bestdegree) + "and" + str(bestdegree2))
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5], errors, errors2, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], [str(round(mseD, 5)), str(round(rmseD, 5)), str(round(r2D, 5))], eq, eqD, [str(round(mseL, 5)), str(round(rmseL, 5)), str(round(r2L, 5))], [str(round(mseLD, 5)), str(round(rmseLD, 5)), str(round(r2LD, 5))]


def getComparacion2Paises(date, countryC, cases, countriesName, csv):
    countries = []
    errorTables = []
    colors = random.choices(
        list(mcolors.CSS4_COLORS.values()), k=140)

    col_list = [date, cases, countryC]

    dataset = pd.DataFrame(csv, columns=col_list)
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[cases], inplace=True)
    #dataset.fillna(0, inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])

    for countryName in countriesName:
        xx = dataset.loc[dataset[countryC] == countryName]

        xx[date] = xx[date].astype('category').cat.codes
        pos1 = dataset.columns.get_loc(date)
        pos2 = dataset.columns.get_loc(cases)
        X = xx.iloc[:, pos1].values
        X = X.reshape(-1, 1)
        y = xx.iloc[:, pos2].values
        newC = Country(countryName, X, y)
        countries.append(newC)

# ----------------------------------

    features = PolynomialFeatures(degree=4)

    fig3 = plt.figure()
    plt.style.use('dark_background')
    for country in countries:

        x_train_transformed = features.fit_transform(country.X)

        model2 = LinearRegression()
        model2.fit(x_train_transformed, country.y)

        x_true_transformed = features.fit_transform(country.X)
        model_curve = model2.predict(x_true_transformed)
        country.model_curve = model_curve
        plt.plot(country.X[:, 0], model_curve, '-',
                 color=random.choice(colors), linewidth=3, label=str(country.name))

    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
    plt.title('Modelo de Regresión Polinomial entre grados 4')

    flike = io.BytesIO()
    fig3.savefig(flike)
    graph3 = base64.b64encode(flike.getvalue()).decode()

    for country in countries:
        errors = []
        for i in range(26):
            errors.append([i] + poly_reg(i, country.X,
                          country.y, country.model_curve))
        df = pd.DataFrame(errors, columns=['grado', 'RMSE'])
        country.errors = errors
        country.df = df

    fig4 = plt.figure()

    for country in countries:
        plt.plot(country.df['grado'], country.df['RMSE'],
                 'o-', color=random.choice(colors), label='Valores RMSE ' + str(country.name))

    plt.xlabel('Grado')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE para diferentes grados de la Regresión Polinomial')
    flike = io.BytesIO()
    fig4.savefig(flike)
    graph4 = base64.b64encode(flike.getvalue()).decode()

    fig5 = plt.figure()

    for country in countries:
        bestdegree = np.argmin(country.errors, axis=0)[1]
        features2 = PolynomialFeatures(degree=bestdegree)

        x_train_transformed2 = features2.fit_transform(country.X)

        model3 = LinearRegression()
        model3.fit(x_train_transformed2, country.y)

        x_true_transformed2 = features2.fit_transform(country.X)
        model_curve3 = model3.predict(x_true_transformed2)
        # ------------------------------------------
        mse = mean_squared_error(y_true=country.y, y_pred=model_curve3)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true=country.y, y_pred=model_curve3)
        coefs = model3.coef_
        country.metrics = [str(round(mse, 5)), str(
            round(rmse, 5)), str(round(r2, 5))]
        # -----------------------------------------
        eq = "y = " + str(model3.intercept_)
        for i in range(len(coefs)):
            if i != 0:
                eq += " +(" + str(coefs[i])+")x^"+str(i)
        country.eq = eq
        plt.plot(country.X[:, 0], model_curve3, '-',
                 color=random.choice(colors), linewidth=3, label=str(country.name))

    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
    plt.title('Modelo de Regresión Polinomial de mejor grado')
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    return [graph3, graph4, graph5], countries
