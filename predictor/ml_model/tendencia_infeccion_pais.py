
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
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


def getGraph(date, country, cases, countryName, csv):
    pos1 = 0
    pos2 = 0
    col_list = [date, country, cases]

    dataset = pd.DataFrame(csv, columns=col_list)

    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[cases], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])

    #dataset = pd.read_csv(csv, usecols=col_list)

    xx = dataset.loc[dataset[country] == countryName]

    # ----------

    # ------------
    #xx[date] = pd.DatetimeIndex(xx[date])
    auxLabel = pd.DatetimeIndex(xx[date])
    xx[date] = xx[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)

    X = xx.iloc[:, pos1].values
    #X = np.asarray(xx[date]).reshape(-1, 1)
    X = X.reshape(-1, 1)

    #yy = dataset.loc[dataset[country] == countryName]
    y = xx.iloc[:, pos2].values

    fig = plt.figure()
    plt.style.use('dark_background')
    plt.scatter(auxLabel, y, color="red")
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
    fig2 = plt.figure()
    plt.plot(auxLabel, y, 'o', color="lightcoral")
    plt.plot(auxLabel, model_curve, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresi??n Lineal')
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

    fig3 = plt.figure()
    plt.plot(auxLabel, y, 'bo')
    plt.plot(auxLabel, model_curve, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresi??n Polinomial entre grados 4')
    plt.xticks(rotation=45)
    plt.tight_layout()

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
    plt.title('RMSE para diferentes grados de la Regresi??n Polinomial')
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
    plt.plot(auxLabel, y, 'bo')
    plt.plot(auxLabel, model_curve3, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresi??n Polinomial de mejor grado '+str(bestdegree))
    plt.xticks(rotation=45)
    plt.tight_layout()
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5], errors, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], eq


def getGraph2(date, country, department, cases, countryName, depaName, csv):
    pos1 = 0
    pos2 = 0
    col_list = [date, country, department, cases]

    dataset = pd.DataFrame(csv, columns=col_list)
    dataset = dataset.replace(r'^\s*$', np.NaN, regex=True)
    dataset.dropna(subset=[cases], inplace=True)

    dataset[cases] = pd.to_numeric(dataset[cases])

    #dataset = pd.read_csv(csv, usecols=col_list)

    xx = dataset.loc[dataset[country] == countryName]

    #xx[date] = pd.DatetimeIndex(xx[date])
    auxLabel = pd.DatetimeIndex(xx[date])
    xx[date] = xx[date].astype('category').cat.codes

    pos1 = dataset.columns.get_loc(date)
    pos2 = dataset.columns.get_loc(cases)

    xx = xx.loc[xx[country] == countryName]
    xx = xx.loc[xx[department] == depaName]

    X = xx.iloc[:, pos1].values
    X = X.reshape(-1, 1)

    #yy = dataset.loc[dataset[country] == countryName]
    y = xx.iloc[:, pos2].values

    fig = plt.figure()
    plt.style.use('dark_background')
    plt.scatter(auxLabel, y, color="red")
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
    fig2 = plt.figure()
    plt.plot(auxLabel, y, 'bo')
    plt.plot(auxLabel, model_curve, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresi??n Lineal')
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

    fig3 = plt.figure()
    plt.plot(auxLabel, y, 'bo')
    plt.plot(auxLabel, model_curve, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresi??n Polinomial entre grados 4')
    plt.xticks(rotation=45)
    plt.tight_layout()

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
    plt.title('RMSE para diferentes grados de la Regresi??n Polinomial')
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
    plt.plot(auxLabel, y, 'bo')
    plt.plot(auxLabel, model_curve3, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresi??n Polinomial de mejor grado '+str(bestdegree))
    plt.xticks(rotation=45)
    plt.tight_layout()
    flike = io.BytesIO()
    fig5.savefig(flike)
    graph5 = base64.b64encode(flike.getvalue()).decode()

    return [graph1, graph2, graph3, graph4, graph5], errors, [str(round(mse, 5)), str(round(rmse, 5)), str(round(r2, 5))], eq
