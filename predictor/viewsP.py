from os import name
from django.shortcuts import render, HttpResponse
from numpy.lib.function_base import gradient
from reportlab.lib import pagesizes
from .forms import ModelForm, ParamSelector
from django.http import FileResponse
import io
import base64
import urllib
import csv
import json
import pandas
from .ml_model.predicciones import getPrediction
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import BaseDocTemplate, Frame, Paragraph, PageBreak, PageTemplate, NextPageTemplate, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus.tables import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from . import views as encabezados
enc = encabezados.get_enc()
jsonArray = []
tempPais = ""
tempDepa = ""
paisNN = ""
auxG = None
metricsG = None
coefsG = None
errorsT = []


def prediccion(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")
        valorPrediccion = request.POST.get("valorPrediccion")
        if columnaPais:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados and valorPrediccion:

            graphs, errors, metrics, coefs = getPrediction(columnaFecha, tempPais, columnaInfectados,
                                                           paisEspecifico, jsonArray, valorPrediccion)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])
            global metricsG
            metricsG = metrics
            global coefsG
            coefsG = coefs

        return render(request, 'predicciones/pred_infectados_pais.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs})

    elif request.GET.get('Down') == 'Down':
        return some_view2(request, auxG, errorsT, metricsG, coefsG)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'predicciones/pred_infectados_pais.html', {'enc': enc,  'paises': paises})


def some_view2(request, graphs, errorsT, metrics, coefs):

    buffer = io.BytesIO()
    doc = BaseDocTemplate(buffer, showBoundary=0, pagesizes=A4)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Titulos',
               fontName='Times-Roman', fontSize=20))
    styles.add(ParagraphStyle(name='Cuerpo',
               fontName='Times-Roman', fontSize=12, alignment=TA_JUSTIFY, leading=14, spaceAfter=10))
    styles.add(ParagraphStyle(name='Subtitulo',
               fontName='Times-Roman', fontSize=14, leading=14, spaceAfter=20))
    styles.add(ParagraphStyle(name='Equations',
               fontName='Times-Roman', fontSize=14, alignment=TA_CENTER, leading=14, spaceAfter=15, spaceBefore=15))
    styles.add(ParagraphStyle(name='Metrics',
               fontName='Times-Roman', fontSize=10, alignment=TA_CENTER, leading=10, spaceAfter=10, spaceBefore=10))

    frame1 = Frame(doc.leftMargin, doc.bottomMargin,
                   doc.width/2-4, doc.height, id='col1')

    frame2 = Frame(doc.leftMargin+doc.width/2+4, doc.bottomMargin,
                   doc.width/2-4, doc.height, id='col2')

    Elements = []

    Elements.append(NextPageTemplate('TwoCol'))
    Elements.append(
        Paragraph("Modelo de regresión lineal simple", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresión lineal son ampliamente usados en la ingeniería ya que sirven para analizar el
    comportamiento de las variables de entrada (o regresora) y salida (o respuesta) estableciendo predicciones y
    estimaciones. En este trabajo la variable regresora corresponde a la distorsión armónica individual de corriente y
    la variable de respuesta corresponde a la distorsión armónica individual de tensión.
    La ecuación, muestra la representación de un modelo de regresión lineal simple, donde Y es la respuesta, X es la
    variable regresora, 0 y 1 son los parámetros del modelo o coeficientes de regresión y es el error del modelo""", styles['Cuerpo']))
    Elements.append(
        Paragraph("El coeficiente de determinación R2 se expresa como un porcentaje que indica la variación de los valores de la variable independiente que se puede explicar con la ecuación de regresión", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> X <greek>e</greek>""", styles['Equations']))
    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Modelo de regresión polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresión polinomial se usan cuando la variable de respuesta muestra un comportamiento curvilíneo
    o no lineal. La ecuación, describe el modelo de regresión polinomial de orden k en una variable regresora y la
    ecuación, muestra el modelo ajustado de regresión polinomial de orden k. Los estimadores de los parámetros del
    modelo se obtienen por el método de los mínimos cuadrados usando la ecuación, donde y, X, X’ son vectores. En
    este trabajo se aplican los modelos de regresión polinomial de orden 2 y orden 3 en una variable regresora""", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> + <greek>b</greek><sub>2</sub> X<super>2</super> + ... + 
        <greek>b</greek><sub>k</sub> X<super>k</super>""", styles['Equations']))
    # Coeficiente de deteriminacion R2
    Elements.append(
        Paragraph("Coeficiente de determinación R<super>2</super>", styles['Subtitulo']))
    Elements.append(Paragraph("""El coeficiente de determinación R2 mide la proporción de la variación de la respuesta Y que es explicada por el
    modelo de regresión. El coeficiente R2 se calcula usando la ecuación, donde SSR es la medida de variabilidad
    del modelo de regresión y SST corresponde a la medida de variabilidad de Y sin considerar el efecto de la variable
    regresora X.""", styles['Cuerpo']))
    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distintas dispersiones y ploteos 
                              de cada caso en especifico, siendo evidente la necesidad de aplicar otro modelo diferente a una regresion lienal""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresión al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utilizó el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gráficas de dispersión de
    las variables de respuesta y regresoras y los resultados analíticos de los modelos. Las figuras, muestra el
    comportamiento gráfico de los modelos de regresión lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

    Elements.append(Image("data:image/png;base64," +
                    graphs[1], width=200, height=150))

    Elements.append(Image("data:image/png;base64," +
                    graphs[2], width=200, height=150))
    # --------------------------------------------------------------------------
    Elements.append(Paragraph("""Por supuesto el supuesto al utilizar un polinomio de grado 4 no es el adecuado ya que la informacion
    y datos son totalmente diferentes para os distintos casos por lo que se realizado un testeo y evaluacion para encontrat el grado
    optimo dados unos datos especificos, a continuacion se muestra una tabla con los resultados al evaluar distintos grados
    y la grafica para evidenciar el mejor grado que muestra un error menor""", styles['Cuerpo']))
    t = Table(errorsT)
    t.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 6),
               ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'), ('INNERGRID',
                                                               (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    Elements.append(t)
    Elements.append(Image("data:image/png;base64," +
                    graphs[3], width=200, height=150))
    # ------------------------------------------------------------------------------------------
    Elements.append(Paragraph("""Ya obtenido el grado que mejor se ajusta para los datos obtenidos se muestra la grafica que mejor se adopta
        y utiliza este grado optimo, ademas con la cual se pueden obtener las metricas que mas se acercan y reflejan
        resutlados adecuados.""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[4], width=200, height=150))

    # ------------------------------------------------------------------------------------------
    Elements.append(Paragraph("Metricas", styles['Subtitulo']))
    Elements.append(Paragraph("""Durente el calculo de todas las graficas el programa se generan distintas metricas y
        valores puntuales que ayudan a la generacion de las gracias como lo son coeficientes y errores que se expondran a continuacion.""", styles['Cuerpo']))
    Elements.append(Paragraph(coefs, styles['Metrics']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresión lineal simple y regresión
    polinomial de orden 4 y regresión múltiple para describir la relación entre la distorsión individual de
    datos de covid y la distorsión individual del paso del tiepo, siendo el modelo de regresión lineal
    múltiple el que mejor ajustó los datos de las mediciones del proceso, con mejor coeficiente de determinación R2
    ("""+str(metrics[2])+""").""", styles['Cuerpo']))

    Elements.append(Paragraph("""Los pronósticos realizados con el modelo de regresión lineal múltiple, permiten estimar la variacion
    individual de datos de variables dependientes e independeintes y direccionar medidas correctivas para el control del contenido
    y ajuste del proceso de analisis.""", styles['Cuerpo']))

    Elements.append(PageBreak())

    doc.addPageTemplates([
        PageTemplate(id='TwoCol', frames=[
            frame1, frame2]),
    ])

    doc.build(Elements)
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename='reportes.pdf')
