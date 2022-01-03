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


def poly_reg(degrees, x_train, y_train, model_curve):
    features = PolynomialFeatures(degree=degrees)
    x_train_transformed = features.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_train_transformed, y_train)

    train_pred = model.predict(x_train_transformed)
    rmse_poly_train = mean_squared_error(y_train, train_pred, squared=False)

    return [rmse_poly_train]


def muertes_regionn(country, regions, deaths, countryName, csv):
    col_list = [country, regions, deaths]
    colors = random.choices(
        list(mcolors.CSS4_COLORS.values()), k=140)

    dataset = pd.DataFrame(csv, columns=col_list)

    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[deaths], inplace=True)
    dataset[deaths] = pd.to_numeric(dataset[deaths])
    #dataset.fillna(0, inplace=True)
    xx = dataset.loc[dataset[country] == countryName]
    X_pre = xx[regions].values
    X_pre = list(dict.fromkeys(X_pre))

    print(X_pre)
    res = xx.groupby(by=[regions]).sum()

    #pos1 = res.columns.get_loc(regions)
    #pos2 = res.columns.get_loc(deaths)
    #X = res[regions].tolist()
    #X = X.reshape(-1, 1)
    y = res[deaths].tolist()

    fig3 = plt.figure()
    plt.style.use('dark_background')
    plt.bar(range(len(X_pre)), y, width=1, edgecolor="white", linewidth=0.7)

    plt.xlabel(regions)
    plt.ylabel(deaths)
    plt.title('Grafica de Muertes por Regiones')
    for i, v in enumerate(y):
        plt.text(i + .25, v + 3, str(v), color='white', fontweight='bold')
    plt.xticks(range(len(X_pre)), X_pre, rotation='vertical')
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.tight_layout()

    flike = io.BytesIO()
    fig3.savefig(flike)
    graph1 = base64.b64encode(flike.getvalue()).decode()
    return [graph1]


def porcentaje_muertess(country, cases, deaths, countryName, csv):
    col_list = [country, cases, deaths]

    dataset = pd.DataFrame(csv, columns=col_list)

    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[deaths], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])
    dataset[deaths] = pd.to_numeric(dataset[deaths])

    xx = dataset.loc[dataset[country] == countryName]

    totalCases = xx[cases].sum()
    totalDeaths = xx[deaths].sum()
    pD = (totalDeaths * 100)/totalCases
    pC = 100 - pD

    arrayL = [pD, pC]

    labels = ["Total Deaths", "Total Cases"]
    explode = (0.5, 0)

    fig3 = plt.figure()
    plt.pie(arrayL, labels=labels, explode=explode, autopct='%1.1f%%')
    plt.axis('equal')

    plt.legend(title="Deaths - Cases:")

    flike = io.BytesIO()
    fig3.savefig(flike)
    graph1 = base64.b64encode(flike.getvalue()).decode()

    arrayL2 = np.array([totalDeaths, totalCases-totalDeaths])
    labels2 = ["Total Deaths", "Total Cases"]
    explode2 = (0.5, 0)

    def absolute_value(val):
        a = np.round(val/100.*arrayL2.sum(), 0)
        return a

    fig1 = plt.figure()
    plt.pie(arrayL2, labels=labels2, explode=explode2, autopct=absolute_value)
    plt.axis('equal')

    plt.legend(title="Deaths - Cases:")

    flike = io.BytesIO()
    fig1.savefig(flike)
    graph2 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2]


def tasa_casos_deaths(date, country, cases, deaths, countryName, csv):
    col_list = [date, country, cases, deaths]

    dataset = pd.DataFrame(csv, columns=col_list)
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[deaths], inplace=True)
    dataset.dropna(subset=[cases], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])
    dataset[deaths] = pd.to_numeric(dataset[deaths])

    xx = dataset.loc[dataset[country] == countryName]
    # -------------
    xx[cases] = xx.groupby([date])[cases].transform('sum')
    xx[deaths] = xx.groupby([date])[deaths].transform('sum')

    new_df = xx.drop_duplicates(subset=[date])
    # ----------------------
    new_df[date] = new_df[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)
    pos3 = dataset.columns.get_loc(deaths)
    X = new_df.iloc[:, pos1].values
    X = X.reshape(-1, 1)
    y = new_df.iloc[:, pos2].values
    z = new_df.iloc[:, pos3].values

    fig = plt.figure()
    plt.style.use('dark_background')
    plt.scatter(X, y, color="red")
    plt.scatter(X, z, color="blue")
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Ploteo entre X y')

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
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], z, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], model_curve, '-', color="orangered", linewidth=3)
    plt.plot(X[:, 0], model_curveDD, '-', color="mediumblue", linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
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

    # Double ----------------------------
    model2DD = LinearRegression()
    model2DD.fit(x_train_transformed, z)
    model_curveDD = model2DD.predict(x_true_transformed)

    fig3 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], z, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], model_curve, '-', color="orangered", linewidth=3)
    plt.plot(X[:, 0], model_curveDD, '-', color="mediumblue", linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de Regresión Polinomial entre grados 4')

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

    plt.plot(df['grado'], df['RMSE'], 'o-r', label='Valores RMSE')
    plt.plot(dfDD['grado'], dfDD['RMSE'], 'o-b', label='Valores RMSE')
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
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], z, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], model_curve3, '-', color="orangered", linewidth=3)
    plt.plot(X[:, 0], model_curve3DD, '-', color="mediumblue", linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de Regresión Polinomial y Lineal de mejor grado ' +
              str(bestdegree) + "and" + str(bestdegree2))
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5], errors, errors2, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], [str(round(mseD, 5)), str(round(rmseD, 5)), str(round(r2D, 5))], eq, eqD


def tasa_casos_casosA_deaths(date, country, cases, casesA, deaths, countryName, csv):
    col_list = [date, country, cases, casesA, deaths]

    dataset = pd.DataFrame(csv, columns=col_list)
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[deaths], inplace=True)
    dataset.dropna(subset=[cases], inplace=True)
    dataset.dropna(subset=[casesA], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])
    dataset[deaths] = pd.to_numeric(dataset[deaths])
    dataset[casesA] = pd.to_numeric(dataset[casesA])

    xx = dataset.loc[dataset[country] == countryName]
    # -------------
    xx[cases] = xx.groupby([date])[cases].transform('sum')
    xx[deaths] = xx.groupby([date])[deaths].transform('sum')
    xx[casesA] = xx.groupby([date])[casesA].transform('sum')

    new_df = xx.drop_duplicates(subset=[date])
    # ----------------------
    new_df[date] = new_df[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)
    pos3 = dataset.columns.get_loc(deaths)
    pos4 = dataset.columns.get_loc(casesA)

    X = new_df.iloc[:, pos1].values
    X = X.reshape(-1, 1)
    y = new_df.iloc[:, pos2].values
    z = new_df.iloc[:, pos3].values
    w = new_df.iloc[:, pos4].values

    fig = plt.figure()
    plt.style.use('dark_background')
    plt.scatter(X, y, color="red", label=cases)
    plt.scatter(X, z, color="blue", label=deaths)
    plt.scatter(X, w, color="green", label=casesA)
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Ploteo entre X y')

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
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], z, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], model_curve, '-', color="orangered", linewidth=3)
    plt.plot(X[:, 0], model_curveDD, '-', color="mediumblue", linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
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

    # Double ----------------------------
    model2DD = LinearRegression()
    model2DD.fit(x_train_transformed, z)
    model_curveDD = model2DD.predict(x_true_transformed)
    # Double ----------------------------
    model2DW = LinearRegression()
    model2DW.fit(x_train_transformed, w)
    model_curveDW = model2DW.predict(x_true_transformed)

    fig3 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], z, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], w, 'o', color="palegreen")
    plt.plot(X[:, 0], model_curve, '-',
             color="orangered", linewidth=3, label=cases)
    plt.plot(X[:, 0], model_curveDD, '-',
             color="mediumblue", linewidth=3, label=deaths)
    plt.plot(X[:, 0], model_curveDW, '-',
             color="lime", linewidth=3, label=casesA)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
    plt.title('Modelo de Regresión Polinomial entre grados 4')

    flike = io.BytesIO()
    fig3.savefig(flike)
    graph3 = base64.b64encode(flike.getvalue()).decode()

    errors = []
    errors2 = []
    errors3 = []
    for i in range(26):
        errors.append([i] + poly_reg(i, X, y, model_curve))
        errors2.append([i] + poly_reg(i, X, z, model_curveDD))
        errors3.append([i] + poly_reg(i, X, w, model_curveDW))

    df = pd.DataFrame(errors, columns=['grado', 'RMSE'])
    dfDD = pd.DataFrame(errors2, columns=['grado', 'RMSE'])
    dfDW = pd.DataFrame(errors3, columns=['grado', 'RMSE'])

    fig4 = plt.figure()

    plt.plot(df['grado'], df['RMSE'], 'o-r', label='Valores RMSE')
    plt.plot(dfDD['grado'], dfDD['RMSE'], 'o-b', label='Valores RMSE')
    plt.plot(dfDW['grado'], dfDW['RMSE'], 'o-g', label='Valores RMSE')
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
    # Double ------------------------
    model3DW = LinearRegression()
    model3DW.fit(x_train_transformed2, w)
    model_curve3DW = model3DW.predict(x_true_transformed2)
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
    # --------
    mseW = mean_squared_error(y_true=w, y_pred=model_curve3DW)
    rmseW = np.sqrt(mseD)
    r2W = r2_score(y_true=w, y_pred=model_curve3DW)
    coefsW = model3DW.coef_
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

    eqW = "y = " + str(model3DW.intercept_)
    for i in range(len(coefsW)):
        if i != 0:
            eqW += " +(" + str(coefsW[i])+")x^"+str(i)

    fig5 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], z, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], w, 'o', color="palegreen")
    plt.plot(X[:, 0], model_curve3, '-',
             color="orangered", linewidth=3, label=cases)
    plt.plot(X[:, 0], model_curve3DD, '-',
             color="mediumblue", linewidth=3, label=deaths)
    plt.plot(X[:, 0], model_curve3DW, '-',
             color="lime", linewidth=3, label=casesA)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
    plt.title('Modelo de Regresión Polinomial y Lineal de mejor grado ' +
              str(bestdegree) + "and" + str(bestdegree2))
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph3, graph4, graph5]
