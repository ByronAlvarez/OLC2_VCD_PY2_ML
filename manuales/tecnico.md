<br><br><br><br>

<h1 align="center" style="font-size: 40px; font-weight: bold;">Proyecto 2</h1>
<h3 align="center" style="font-size: 20px; font-weight: bold;">Manual Técnico</h3>

## <br><br><br>

<h1>Tabla de Contenido</h1>

- [**1. Introduccion**](#1-introduccion)
- [**2. Requisitos del Sistema**](#2-requisitos-del-sistema)
- [**3. Descripción del Código**](#3-descripcion-del-codigo)
- [**4. Herramientas**](#4-herramientas)
- [**5. Django**](#5-django)
- [**6. Heroku**](#5-heroku)

## <br><br>

# **1. Introduccion**

Uno de los recursos mas valiosos a lo largo de todo el munod hoy en dia es la informacion, los datos. En una epoca donde todo funciona y se obtiene a traves de datos es logico pensar que es un campo de estudio
Obviamente para un ingeniero en sistemas no es la excepcion saber utilizar herramientas que producen un gran numero de resultados sino tambien saber analizarlos e interpretalos para obtener su mayor provecho en todos los campos que abarca esta ciencia.
<br><br>

---

# **2. Requisitos del Sistema**

| Componente | Mínimo                                                                 | Recomendado                                                    |
| ---------- | ---------------------------------------------------------------------- | -------------------------------------------------------------- |
| Procesador | Procesador de x86 o x64 bits de doble núcleo de 2,33 gigahercios (GHz) | Procesador de 64 bits de doble núcleo de 3,3 gigahercios (GHz) |
| Memoria    | 8 GB de RAM                                                            | 16 GB de RAM o más                                             |
| Disco Duro | 3 GB de espacio disponible en el disco duro                            | 3 GB de espacio disponible en el disco duro                    |
| Pantalla   | 1024 x 768                                                             | 1024 x 768                                                     |

<br><br>

---

# **3. Descripción del Código**

En esta sección se detalla y explican algunos de los métodos relevantes de ejecución del programa que permiten su funcionamientos y una salida de información adecuada a lo que se requiere.

<br><br>

## DataFrame

Este objeto que proviene de la libreria pandas permite trabajar datos y agruparlos de una mejor manera, ya que tiene muchos metodos propios que facilitan el trabajo. En repetidos metodos se muestran las siguientes lineas de codigo donde se busca manipular la informacion para distribuirla y usarla a nuestra conveniencia.

```python
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
```

## Linear Regression

El análisis de regresión engloba a un conjunto de métodos estadísticos que usamos cuando tanto la variable de respuesta como la la(s) variable(s) predictiva(s) son contínuas y queremos predecir valores de la primera en función de valores observados de las segundas. Por lo que con el siguiente codigo se consigue modelar esta regresion para utilizar los datos

```python
    model = LinearRegression()
    model.fit(X, y)

    model_curve = model.predict(X)
```

## Polynomial Regression

La dependencia entre la variable de respuesta y la regresora frecuentemente no es lineal. No obstante, se requiere que ajustemos un modelo lineal simple (el más simple, menos parametrizado) como modelo nulo, a no ser que un modelo no lineal se demuestre significativamente superior al nulo. Por lo que se realizar la modelacion polinomial ajustando con una regresion lineal de la siguiente manera y que se utiliza en distintas partes de codigo.

```python
    features = PolynomialFeatures(degree=4)

    x_train_transformed = features.fit_transform(X)

    model2 = LinearRegression()
    model2.fit(x_train_transformed, y)

    x_true_transformed = features.fit_transform(X)
    model_curve = model2.predict(x_true_transformed)
```

## Errores

Es importate calcular distintos datos y metricas con respecto a los modelos utilizados por lo que las mimsas librerias proporcionan estos metodos a traves del siguiente codigo.

```python
mse = mean_squared_error((y_true = y), (y_pred = model_curve3));
rmse = np.sqrt(mse);
r2 = r2_score((y_true = y), (y_pred = model_curve3));
coefs = model3.coef_;
```

## Graficas

A traves de matplolib se generan las graficas y medianto porciones de codigo parecidas a estas es como se generan las graficas que muetran los resultados obtenidos al manejar los datos.

```python
    fig = plt.figure()
    plt.style.use('dark_background')
    plt.scatter(X, y, color="red")
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Ploteo entre X y')

    fig2 = plt.figure()
    plt.plot(X[:, 0], y, 'o', color="lightcoral")
    plt.plot(X[:, 0], model_curve, 'r-', linewidth=3)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de Regresión Lineal')
```

El resto de clases son la implementacion del lado frontend donde son conceptos mas especificos de Django y su manejo de modulos.

---

# **4. Herramientas**

Scikit-Learn es una de estas librerías gratuitas para Python. Cuenta con algoritmos de clasificación, regresión, clustering y reducción de dimensionalidad. Además, presenta la compatibilidad con otras librerías de Python como NumPy, SciPy y matplotlib.

<p align="center">
  <img width="460" height="250" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png">
</p>

- **[Scikit Learn Official Page](https://scikit-learn.org/stable/)**
- **[Scikit Learn Ordinary Least Squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)**
- **[Scikit Learn Polynomial Regression](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)**

Matplotlib es una biblioteca para la generación de gráficos a partir de datos contenidos en listas o arrays en el lenguaje de programación Python y su extensión matemática NumPy. Proporciona una API, pylab, diseñada para recordar a la de MATLAB.

<p align="center">
  <img width="460" height="230" src="https://frankgalandev.com/wp-content/uploads/2021/12/logo_matplotlib.png">
</p>

- **[Matplotlib Official Page](https://matplotlib.org/)**

Pandas es una muy popular librería de código abierto dentro de los desarrolladores de Python, y sobre todo dentro del ámbito de Data Science y Machine Learning, ya que ofrece unas estructuras muy poderosas y flexibles que facilitan la manipulación y tratamiento de datos.

<p align="center">
  <img width="500" height="200" src="https://www.adictosaltrabajo.com/wp-content/uploads/2020/12/1200px-Pandas_logo.svg_.png">
</p>

- **[Pandas Official Page](https://pandas.pydata.org/)**

# **5. Django**

<p align="center">
  <img width="460" height="230" src="https://dc722jrlp2zu8.cloudfront.net/media/featured_images/django-logo-negative.png">
</p>

- **[Django Official Page](https://www.djangoproject.com/)**

Se utilizao Django tanto para el frontend como para el backend con la razon principal de utilizar la facilidad de python para manejar datos y asi mover la informacion sin muchos problemas. Por parte del lado del frontend se utilizaron los templates propios de Django con el auxilio de Bootstrap para añadir estilos a la pagina.

# **6. Heroku**

<p align="center">
  <img width="500" height="150" src="https://estebanromero.com/wp-content/uploads/2018/02/heroku1.png">
</p>

- **[Heroku Official Page](https://www.heroku.com/)**

Heroku es una plataforma de servicios en la nube (conocidos como PaaS o Platform as a Service) que permite manejar los servidores y sus configuraciones, escalamiento y la administración. Su popularidad ha crecido en los últimos años debido a su facilidad de uso y versatilidad para distintos proyectos

Este proyecto tiene su deploy en este servicio de la nube por su alto rango de lenguajes de despligue y su estabilidad en los recursos que ofrece.

## <br><br>
