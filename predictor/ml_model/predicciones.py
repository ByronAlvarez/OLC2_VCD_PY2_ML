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
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[cases], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])

    xx = dataset.loc[dataset[country] == countryName]
    xx[date] = xx[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)
    X = xx.iloc[:, pos1].values
    X = X.reshape(-1, 1)
    y = xx.iloc[:, pos2].values

    fig = plt.figure()
    plt.style.use('dark_background')
    plt.scatter(X, y, color="red")
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

    mseL = mean_squared_error(y_true=y, y_pred=model_curve)
    rmseL = np.sqrt(mseL)
    r2L = r2_score(y_true=y, y_pred=model_curve)

    fig2 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], model_curve, '-', color="orangered", linewidth=3)
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

    fig3 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
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
    plt.grid()
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
    plt.plot(X[:, 0], y, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], model_curve3, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de Regresión Polinomial de mejor grado '+str(bestdegree))
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    # Predictions ------------------------------------------------------

    featuresP = PolynomialFeatures(degree=bestdegree)
    x_train_transformed = featuresP.fit_transform(X)
    modelP = LinearRegression()
    modelP.fit(x_train_transformed, y)

    x_pred = np.arange(0, len(y)+int(tiempoPred))
    x_pred = x_pred[:, np.newaxis]

    x_true_transformedP = featuresP.fit_transform(x_pred)
    model_curveP = modelP.predict(x_true_transformedP)

    valordePredPolinomial = model_curveP[[(len(y)+int(tiempoPred))-1]]

    fig6 = plt.figure()
    #plt.plot(X[:, 0], y, 'bo')
    plt.plot(x_pred[:, 0], model_curveP, 'r-', linewidth=3)
    plt.grid()
    plt.xlim(0,  len(y)+int(tiempoPred))
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de prediccion Polinomial en: ' +
              str(tiempoPred) + "dias/semanas/meses/años")
    flike = io.BytesIO()
    fig6.savefig(flike)
    graph6 = base64.b64encode(flike.getvalue()).decode()
    #  PRediccion Lineal----------------------------------------------

    model_curvePL = model.predict(x_pred)
    valordePredLineal = model_curvePL[[(len(y)+int(tiempoPred))-1]]

    eqL = "y = " + str(model.intercept_) + " + " + str(model.coef_[0]) + "x"

    fig7 = plt.figure()
    #plt.plot(X[:, 0], y, 'bo')
    plt.plot(x_pred[:, 0], model_curvePL, 'r-', linewidth=3)
    plt.grid()
    plt.xlim(0,  len(y)+int(tiempoPred))
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de prediccion Lineal en: ' +
              str(tiempoPred) + "dias/semanas/meses/años")
    flike = io.BytesIO()
    fig7.savefig(flike)
    graph7 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5, graph6, graph7], errors, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], [eq, eqL], [valordePredPolinomial, valordePredLineal], [str(round(mseL, 5)), str(round(rmseL, 5)), str(round(r2L, 5))]


def getDoublePrediction(date, country, cases, deaths, countryName, csv, tiempoPred):
    col_list = [date, country, cases, deaths]

    dataset = pd.DataFrame(csv, columns=col_list)
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[deaths], inplace=True)
    dataset.dropna(subset=[cases], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])
    dataset[deaths] = pd.to_numeric(dataset[deaths])

    xx = dataset.loc[dataset[country] == countryName]
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
    plt.scatter(X, y, color="red", label=str(cases))
    plt.scatter(X, z, color="blue", label=str(deaths))
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
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3, label=str(cases))
    plt.plot(X[:, 0], model_curveDD, 'b-', linewidth=3, label=str(deaths))
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
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
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3, label=str(cases))
    plt.plot(X[:, 0], model_curveDD, 'b-', linewidth=3, label=str(deaths))
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
    plt.plot(X[:, 0], model_curve3, 'r-', linewidth=3, label=str(cases))
    plt.plot(X[:, 0], model_curve3DD, 'b-', linewidth=3, label=str(deaths))
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
    plt.title('Modelo de Regresión Polinomial y Lineal de mejor grado ' +
              str(bestdegree) + "and" + str(bestdegree2))
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    # Predictions ------------------------------------------------------

    featuresP = PolynomialFeatures(degree=bestdegree)
    x_train_transformed = featuresP.fit_transform(X)
    modelP = LinearRegression()
    modelP.fit(x_train_transformed, y)

    x_pred = np.arange(0, len(y)+int(tiempoPred))
    x_pred = x_pred[:, np.newaxis]

    x_true_transformedP = featuresP.fit_transform(x_pred)
    model_curveP = modelP.predict(x_true_transformedP)
    # Double-----------------------
    modelPDD = LinearRegression()
    modelPDD.fit(x_train_transformed, z)
    model_curvePDD = modelPDD.predict(x_true_transformedP)

    valordePredPolinomial = model_curveP[[(len(y)+int(tiempoPred))-1]]
    valordePredPolinomialD = model_curvePDD[[(len(z)+int(tiempoPred))-1]]

    fig6 = plt.figure()
    #plt.plot(X[:, 0], y, 'bo')
    #plt.plot(X[:, 0], z, 'ko')
    plt.plot(x_pred[:, 0], model_curveP, 'r-', linewidth=3, label=str(cases))
    plt.plot(x_pred[:, 0], model_curvePDD, 'b-',
             linewidth=3, label=str(deaths))
    plt.grid()
    plt.xlim(0,  len(y)+int(tiempoPred))
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
    plt.title('Modelo de prediccion Polinomial en: ' +
              str(tiempoPred) + "dias/semanas/meses/años")
    flike = io.BytesIO()
    fig6.savefig(flike)
    graph6 = base64.b64encode(flike.getvalue()).decode()
    #  PRediccion Lineal----------------------------------------------

    model_curvePL = model.predict(x_pred)

    model_curvePLDD = modelDD.predict(x_pred)

    valordePredLineal = model_curvePL[[(len(y)+int(tiempoPred))-1]]
    valordePredLinealD = model_curvePLDD[[(len(z)+int(tiempoPred))-1]]

    eqL = "y = " + str(model.intercept_) + " + " + str(model.coef_[0]) + "x"
    eqLD = "y = " + str(modelDD.intercept_) + " + " + \
        str(modelDD.coef_[0]) + "x"

    fig7 = plt.figure()
    #plt.plot(X[:, 0], y, 'bo')
    #plt.plot(X[:, 0], z, 'ko')
    plt.plot(x_pred[:, 0], model_curvePL, 'r-', linewidth=3, label=str(cases))
    plt.plot(x_pred[:, 0], model_curvePLDD, 'b-',
             linewidth=3, label=str(deaths))
    plt.grid()
    plt.xlim(0,  len(y)+int(tiempoPred))
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.legend()
    plt.title('Modelo de prediccion Lineal en: ' +
              str(tiempoPred) + "dias/semanas/meses/años")
    flike = io.BytesIO()
    fig7.savefig(flike)
    graph7 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5, graph6, graph7], errors, errors2, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], [str(round(mseD, 5)), str(round(rmseD, 5)), str(round(r2D, 5))], [eq, eqL], [eqD, eqLD], [valordePredPolinomial, valordePredLineal], [valordePredPolinomialD, valordePredLinealD], [str(round(mseL, 5)), str(round(rmseL, 5)), str(round(r2L, 5))], [str(round(mseLD, 5)), str(round(rmseLD, 5)), str(round(r2LD, 5))]


def getPredictionDepa(date, country, department, cases, countryName, depaName, csv, tiempoPred):
    col_list = [date, country, department, cases]

    dataset = pd.DataFrame(csv, columns=col_list)
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[cases], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])

    xx = dataset.loc[dataset[country] == countryName]
    xx[date] = xx[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)

    xx = xx.loc[xx[country] == countryName]
    xx = xx.loc[xx[department] == depaName]

    X = xx.iloc[:, pos1].values
    X = X.reshape(-1, 1)
    y = xx.iloc[:, pos2].values

    fig = plt.figure()
    plt.style.use('dark_background')
    plt.scatter(X, y, color="red")
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

    mseL = mean_squared_error(y_true=y, y_pred=model_curve)
    rmseL = np.sqrt(mseL)
    r2L = r2_score(y_true=y, y_pred=model_curve)

    fig2 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3)
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

    fig3 = plt.figure()
    plt.plot(X[:, 0], y, 'bo')
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
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
    plt.grid()
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
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de Regresión Polinomial de mejor grado '+str(bestdegree))
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    # Predictions ------------------------------------------------------

    featuresP = PolynomialFeatures(degree=bestdegree)
    x_train_transformed = featuresP.fit_transform(X)
    modelP = LinearRegression()
    modelP.fit(x_train_transformed, y)

    x_pred = np.arange(0, len(y)+int(tiempoPred))
    x_pred = x_pred[:, np.newaxis]

    x_true_transformedP = featuresP.fit_transform(x_pred)
    model_curveP = modelP.predict(x_true_transformedP)

    valordePredPolinomial = model_curveP[[(len(y)+int(tiempoPred))-1]]

    fig6 = plt.figure()
    #plt.plot(X[:, 0], y, 'bo')
    plt.plot(x_pred[:, 0], model_curveP, 'r-', linewidth=3)
    plt.grid()
    plt.xlim(0,  len(y)+int(tiempoPred))
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de prediccion Polinomial en: ' +
              str(tiempoPred) + " dias/semanas/meses/años")
    flike = io.BytesIO()
    fig6.savefig(flike)
    graph6 = base64.b64encode(flike.getvalue()).decode()
    #  PRediccion Lineal----------------------------------------------

    model_curvePL = model.predict(x_pred)
    valordePredLineal = model_curvePL[[(len(y)+int(tiempoPred))-1]]

    eqL = "y = " + str(model.intercept_) + " + " + str(model.coef_[0]) + "x"

    fig7 = plt.figure()
    #plt.plot(X[:, 0], y, 'bo')
    plt.plot(x_pred[:, 0], model_curvePL, 'r-', linewidth=3)
    plt.grid()
    plt.xlim(0,  len(y)+int(tiempoPred))
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de prediccion Lineal en: ' +
              str(tiempoPred) + " dias/semanas/meses/años")
    flike = io.BytesIO()
    fig7.savefig(flike)
    graph7 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5, graph6, graph7], errors, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], [eq, eqL],  [valordePredPolinomial, valordePredLineal], [str(round(mseL, 5)), str(round(rmseL, 5)), str(round(r2L, 5))]


def getPredictionLastDay(date, country, cases, countryName, csv):
    col_list = [date, country, cases]

    dataset = pd.DataFrame(csv, columns=col_list)
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[cases], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])

    xx = dataset.loc[dataset[country] == countryName]

    años = pd.DatetimeIndex(xx[date]).year

    minaño = pd.DatetimeIndex(xx[date]).is_year_end
    res = [i for i, val in enumerate(minaño) if val and años[i] == min(años)]

    xx[date] = xx[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)
    X = xx.iloc[:, pos1].values
    X = X.reshape(-1, 1)
    y = xx.iloc[:, pos2].values

    fig = plt.figure()
    plt.style.use('dark_background')
    plt.scatter(X, y, color="red")
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

    mseL = mean_squared_error(y_true=y, y_pred=model_curve)
    rmseL = np.sqrt(mseL)
    r2L = r2_score(y_true=y, y_pred=model_curve)

    fig2 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="mistyrose")
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3)
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

    fig3 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
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
    plt.grid()
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
    plt.plot(X[:, 0], y, 'o', color="lightsteelblue")
    plt.plot(X[:, 0], model_curve3, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de Regresión Polinomial de mejor grado '+str(bestdegree))
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    # Predictions ------------------------------------------------------

    featuresP = PolynomialFeatures(degree=bestdegree)
    x_train_transformed = featuresP.fit_transform(X)
    modelP = LinearRegression()
    modelP.fit(x_train_transformed, y)

    x_true_transformedP = featuresP.fit_transform(X)
    model_curveP = modelP.predict(x_true_transformedP)

    valordePredPolinomial = model_curveP[[res]]

    fig6 = plt.figure()
    #plt.plot(X[:, 0], y, 'bo')
    plt.plot(X[:, 0], model_curveP, 'r-', linewidth=3)
    plt.plot([res], valordePredPolinomial, marker="o", markersize=15,
             markeredgecolor="red", markerfacecolor="lime")
    plt.grid()

    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de prediccion Polinomial en: ' +
              str(res) + "dias/semanas/meses/años")
    flike = io.BytesIO()
    fig6.savefig(flike)
    graph6 = base64.b64encode(flike.getvalue()).decode()
    #  PRediccion Lineal----------------------------------------------

    model_curvePL = model.predict(X)
    valordePredLineal = model_curvePL[[res]]

    eqL = "y = " + str(model.intercept_) + " + " + str(model.coef_[0]) + "x"

    fig7 = plt.figure()
    #plt.plot(X[:, 0], y, 'bo')
    plt.plot(X[:, 0], model_curvePL, 'r-', linewidth=3)
    plt.plot([res], valordePredLineal, marker="o", markersize=15,
             markeredgecolor="red", markerfacecolor="lime")
    plt.grid()

    plt.xlabel(date)
    plt.ylabel(cases)
    plt.title('Modelo de prediccion Lineal en: ' +
              str(res) + "dias/semanas/meses/años")
    flike = io.BytesIO()
    fig7.savefig(flike)
    graph7 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5, graph6, graph7], errors, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], [eq, eqL], [valordePredPolinomial, valordePredLineal], [str(round(mseL, 5)), str(round(rmseL, 5)), str(round(r2L, 5))]
