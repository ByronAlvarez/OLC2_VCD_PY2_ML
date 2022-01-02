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
from .ml_model.tendencia_infeccion_pais import getGraph, getGraph2
from .ml_model.predicciones import getPrediction, getPredictionLastDay, getPredictionDepa, getDoublePrediction
from .ml_model.analisis import getComparacion, getComparacion2Paises
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import BaseDocTemplate, Frame, Paragraph, PageBreak, PageTemplate, NextPageTemplate, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus.tables import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
enc = []
paisess = []
departs = []
jsonArray = []
csvActual = None
parameters = []
listapaises = []

tempPais = ""
tempDepa = ""
paisNN = ""
auxG = None
metricsG = None
metrics2G = None
coefsG = None
coefs2G = None
valorPredG = None
valorPred2G = None
metricsGL = None
metrics2GL = None
countriesG = []
errorsT = []
errorsT2 = []


def get_enc():
    return enc


def home(request):

    return render(request, 'base.html')


def upload(request):
    if request.method == 'POST':

        global jsonArray
        csvReader = request.FILES['formFile']
        extension = csvReader.name.split(".")[1].lower()
        if extension == "csv":
            global csvActual
            csvActual = csvReader
            decoded_file = csvReader.read().decode('utf-8').splitlines()
            reader = csv.DictReader(decoded_file)
            jsonArray = []
            for rows in reader:
                for key in rows:
                    if key is None or rows[key] is None:
                        del rows[key]

                jsonArray.append(rows)
        elif extension == "json":
            jsonArray = json.load(csvReader)
        elif extension == "xlsx" or extension == "xls":

            aux = pandas.read_excel(csvReader)
            jsonArray = aux.to_dict('records')

        global enc
        enc = jsonArray[0].keys()

        return render(request, 'upload.html', {'enc': enc, 'data': jsonArray})
    else:
        return render(request, 'upload.html', {'enc': enc, 'data': jsonArray})


def comp_vac(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        pais2Especifico = request.POST.get("pais2Especifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")
        if columnaPais:
            graphs = []
            countries = []
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and pais2Especifico and columnaFecha and columnaInfectados:
            global listapaises
            listapaises.append(paisEspecifico)
            listapaises.append(pais2Especifico)
            graphs, countries = getComparacion2Paises(
                columnaFecha, tempPais, columnaInfectados, listapaises, jsonArray)
            listapaises = []
            global auxG
            auxG = graphs
            global countriesG
            countriesG = countries

        return render(request, 'analisis/comp_vac.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'countries': countries, 'listapaises': listapaises})

    elif request.GET.get('Down') == 'Down':

        return reportComp2omas(request, auxG, countriesG)
        # return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'analisis/comp_vac.html', {'enc': enc,  'paises': paises})


def comp_2omas(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")
        if columnaPais:
            graphs = []
            countries = []
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico:
            graphs = []
            countries = []
            global listapaises
            listapaises.append(paisEspecifico)

        elif columnaFecha and columnaInfectados:

            graphs, countries = getComparacion2Paises(
                columnaFecha, tempPais, columnaInfectados, listapaises, jsonArray)
            listapaises = []
            global auxG
            auxG = graphs
            global countriesG
            countriesG = countries

        return render(request, 'analisis/comp_2omas.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'countries': countries, 'listapaises': listapaises})

    elif request.GET.get('Down') == 'Down':

        return reportComp2omas(request, auxG, countriesG)
        # return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'analisis/comp_2omas.html', {'enc': enc,  'paises': paises})


def comp_casos_pruebas(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")
        columnaDeath = request.POST.get("columnaDeath")
        if columnaPais:
            graphs = []
            errors = []
            errors2 = []
            coefs = ""
            coefs2 = ""
            metrics = ""
            metrics2 = ""
            metricsL = ""
            valorPred = ""
            valorPredD = ""
            metricsLD = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados and columnaDeath:

            graphs, errors, errors2, metrics, metrics2, coefs, coefs2, metricsL, metricsLD = getComparacion(columnaFecha, tempPais, columnaInfectados, columnaDeath,
                                                                                                            paisEspecifico, jsonArray)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            global errorsT2
            errorsT2 = errors2
            errorsT2.insert(0, ["Grado", "RMSE"])
            errorsT.insert(0, ["Grado", "RMSE"])
            global metricsG
            metricsG = metrics
            global metrics2G
            metrics2G = metrics2
            global metricsGL
            metricsGL = metricsL
            global metrics2GL
            metrics2GL = metricsLD
            global coefsG
            coefsG = coefs
            global coefs2G
            coefs2G = coefs2
        return render(request, 'analisis/comp_casos_pruebas.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'errors2': errors2, 'metrics': metrics, 'coefs': coefs, 'coefs2': coefs2, 'metricsL': metricsL, 'metrics2': metrics2, 'metricsLD': metricsLD})

    elif request.GET.get('Down') == 'Down':

        return reportComp(request, auxG, errorsT, errorsT2, metricsG, metrics2G, coefsG, coefs2G, metricsGL, metrics2GL)
        # return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'analisis/comp_casos_pruebas.html', {'enc': enc,  'paises': paises})


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
            metricsL = ""
            valorPred = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados and valorPrediccion:

            graphs, errors, metrics, coefs, valorPred, metricsL = getPrediction(columnaFecha, tempPais, columnaInfectados,
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
            global valorPredG
            valorPredG = valorPred
            global metricsGL
            metricsGL = metricsL

        return render(request, 'predicciones/pred_infectados_pais.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs, 'valorPred': valorPred, 'metricsL': metricsL})

    elif request.GET.get('Down') == 'Down':
        return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'predicciones/pred_infectados_pais.html', {'enc': enc,  'paises': paises})


def pred_mortalidad_depa(request):
    # if this is a POST request we need to process the form data
    paises = []
    depas = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        columnaDepa = request.POST.get("columnaDepa")
        paisEspecifico = request.POST.get("paisEspecifico")
        depaEspecifico = request.POST.get("depaEspecifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")
        valorPrediccion = request.POST.get("valorPrediccion")
        if columnaPais and columnaDepa:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            valorPred = ""
            metricsL = ""
            global tempPais
            tempPais = columnaPais
            global tempDepa
            tempDepa = columnaDepa
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])
            global paisess
            paisess = paises
        elif paisEspecifico:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            valorPred = ""
            metricsL = ""
            global paisNN
            paisNN = paisEspecifico
            for rows in jsonArray:
                if rows[tempPais] == paisEspecifico:
                    if not(rows[tempDepa] in depas):
                        depas.append(rows[tempDepa])

        elif depaEspecifico and columnaFecha and columnaInfectados and valorPrediccion:

            graphs, errors, metrics, coefs, valorPred, metricsL = getPredictionDepa(columnaFecha, tempPais, tempDepa, columnaInfectados,
                                                                                    paisNN, depaEspecifico, jsonArray, valorPrediccion)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])

            global metricsG
            metricsG = metrics
            global metricsGL
            metricsGL = metricsL
            global coefsG
            coefsG = coefs
            global valorPredG
            valorPredG = valorPred

        return render(request, 'predicciones/pred_mortalidad_depa.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs, 'valorPred': valorPred, 'metricsL': metricsL, 'depas': depas})

    elif request.GET.get('Down') == 'Down':
        return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'predicciones/pred_mortalidad_depa.html', {'enc': enc,  'paises': paises})


def pred_mortalidad_pais(request):
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
            metricsL = ""
            valorPred = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados and valorPrediccion:

            graphs, errors, metrics, coefs, valorPred, metricsL = getPrediction(columnaFecha, tempPais, columnaInfectados,
                                                                                paisEspecifico, jsonArray, valorPrediccion)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])

            global metricsG
            metricsG = metrics
            global metricsGL
            metricsGL = metricsL
            global coefsG
            coefsG = coefs
            global valorPredG
            valorPredG = valorPred

        return render(request, 'predicciones/pred_mortalidad_pais.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs, 'valorPred': valorPred, 'metricsL': metricsL})

    elif request.GET.get('Down') == 'Down':
        return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'predicciones/pred_mortalidad_pais.html', {'enc': enc,  'paises': paises})


def pred_casos_pais(request):
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
            metricsL = ""
            valorPred = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados and valorPrediccion:

            graphs, errors, metrics, coefs, valorPred, metricsL = getPrediction(columnaFecha, tempPais, columnaInfectados,
                                                                                paisEspecifico, jsonArray, valorPrediccion)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])

            global metricsG
            metricsG = metrics
            global metricsGL
            metricsGL = metricsL
            global coefsG
            coefsG = coefs
            global valorPredG
            valorPredG = valorPred

        return render(request, 'predicciones/pred_casos_pais.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs, 'valorPred': valorPred, 'metricsL': metricsL})

    elif request.GET.get('Down') == 'Down':
        return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'predicciones/pred_casos_pais.html', {'enc': enc,  'paises': paises})


def pred_casos_dia(request):
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
            metricsL = ""
            valorPred = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados and valorPrediccion:

            graphs, errors, metrics, coefs, valorPred, metricsL = getPrediction(columnaFecha, tempPais, columnaInfectados,
                                                                                paisEspecifico, jsonArray, valorPrediccion)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])

            global metricsG
            metricsG = metrics
            global metricsGL
            metricsGL = metricsL
            global coefsG
            coefsG = coefs
            global valorPredG
            valorPredG = valorPred

        return render(request, 'predicciones/pred_casos_dia.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs, 'valorPred': valorPred, 'metricsL': metricsL})

    elif request.GET.get('Down') == 'Down':
        return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'predicciones/pred_casos_dia.html', {'enc': enc,  'paises': paises})


def pred_casos_muertes(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")
        columnaDeath = request.POST.get("columnaDeath")
        valorPrediccion = request.POST.get("valorPrediccion")
        if columnaPais:
            graphs = []
            errors = []
            errors2 = []
            coefs = ""
            coefs2 = ""
            metrics = ""
            metrics2 = ""
            metricsL = ""
            valorPred = ""
            valorPredD = ""
            metricsLD = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados and columnaDeath and valorPrediccion:

            graphs, errors, errors2, metrics, metrics2, coefs, coefs2, valorPred, valorPredD, metricsL, metricsLD = getDoublePrediction(columnaFecha, tempPais, columnaInfectados, columnaDeath,
                                                                                                                                        paisEspecifico, jsonArray, valorPrediccion)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            global errorsT2
            errorsT2 = errors2
            errorsT2.insert(0, ["Grado", "RMSE"])
            errorsT.insert(0, ["Grado", "RMSE"])
            global metricsG
            metricsG = metrics
            global metrics2G
            metrics2G = metrics2
            global metricsGL
            metricsGL = metricsL
            global metrics2GL
            metrics2GL = metricsLD
            global coefsG
            coefsG = coefs
            global coefs2G
            coefs2G = coefs2
            global valorPredG
            valorPredG = valorPred
            global valorPred2G
            valorPred2G = valorPredD

        return render(request, 'predicciones/pred_casos_muertes.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'errors2': errors2, 'metrics': metrics, 'coefs': coefs, 'coefs2': coefs2, 'valorPred': valorPred, 'valorPredD': valorPredD, 'metricsL': metricsL, 'metrics2': metrics2, 'metricsLD': metricsLD})

    elif request.GET.get('Down') == 'Down':

        return reportPredDouble(request, auxG, errorsT, errorsT2, metricsG, metrics2G, coefsG, coefs2G, valorPredG, valorPred2G, metricsGL, metrics2GL)
        # return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'predicciones/pred_casos_muertes.html', {'enc': enc,  'paises': paises})


def pred_ultimo_primero(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")
        if columnaPais:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            metricsL = ""
            valorPred = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados:

            graphs, errors, metrics, coefs, valorPred, metricsL = getPredictionLastDay(columnaFecha, tempPais, columnaInfectados,
                                                                                       paisEspecifico, jsonArray)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])

            global metricsG
            metricsG = metrics
            global metricsGL
            metricsGL = metricsL
            global coefsG
            coefsG = coefs
            global valorPredG
            valorPredG = valorPred

        return render(request, 'predicciones/pred_ultimo_primero.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs, 'valorPred': valorPred, 'metricsL': metricsL})

    elif request.GET.get('Down') == 'Down':
        return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'predicciones/pred_ultimo_primero.html', {'enc': enc,  'paises': paises})


def tendencia_casos_depa(request):
    # if this is a POST request we need to process the form data
    paises = []
    depas = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        pais = request.POST.get("paisesS")
        depa = request.POST.get("depaS")
        paisN = request.POST.get("paisN")
        depaa = request.POST.get("depaa")
        param2 = request.POST.get("param2")
        param3 = request.POST.get("param3")
        if pais and depa:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            global tempPais
            tempPais = pais
            global tempDepa
            tempDepa = depa
            for rows in jsonArray:
                for key in rows:
                    if key == pais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])
            global paisess
            paisess = paises
        elif paisN:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            global paisNN
            paisNN = paisN
            for rows in jsonArray:
                if rows[tempPais] == paisN:
                    if not(rows[tempDepa] in depas):
                        depas.append(rows[tempDepa])

        elif depaa and param2 and param3:
            global parameters
            parameters = []
            parameters.append(param2)
            parameters.append(param3)
            graphs, errors, metrics, coefs = getGraph2(param2, tempPais, tempDepa, param3,
                                                       paisNN, depaa, jsonArray)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])
            global metricsG
            metricsG = metrics
            global coefsG
            coefsG = coefs

        return render(request, 'tendencias/tendencia_casos_depa.html', {'enc': enc, 'parameters': parameters, 'paises': paisess, 'depas': depas, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs})

    elif request.GET.get('Down') == 'Down':
        return some_view2(request, auxG, errorsT, metricsG, coefsG)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'tendencias/tendencia_casos_depa.html', {'enc': enc, 'parameters': parameters, 'paises': paisess})


def tendencia_vacuancion_pais(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        pais = request.POST.get("paisesS")
        param1 = request.POST.get("param1")
        param2 = request.POST.get("param2")
        param3 = request.POST.get("param3")
        if pais:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            global tempPais
            tempPais = pais
            for rows in jsonArray:
                for key in rows:
                    if key == pais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif param1 and param2 and param3:
            global parameters
            parameters = []
            parameters.append(param2)
            parameters.append(param3)
            graphs, errors, metrics, coefs = getGraph(param2, tempPais, param3,
                                                      param1, jsonArray)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])
            global metricsG
            metricsG = metrics
            global coefsG
            coefsG = coefs

        return render(request, 'tendencias/tendencia_vacuancion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs})

    elif request.GET.get('Down') == 'Down':
        return some_view2(request, auxG, errorsT, metricsG, coefsG)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'tendencias/tendencia_vacuancion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})


def tendencia_infecxdia_pais(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        pais = request.POST.get("paisesS")
        param1 = request.POST.get("param1")
        param2 = request.POST.get("param2")
        param3 = request.POST.get("param3")
        if pais:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            global tempPais
            tempPais = pais
            for rows in jsonArray:
                for key in rows:
                    if key == pais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif param1 and param2 and param3:
            global parameters
            parameters = []
            parameters.append(param2)
            parameters.append(param3)
            graphs, errors, metrics, coefs = getGraph(param2, tempPais, param3,
                                                      param1, jsonArray)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])
            global metricsG
            metricsG = metrics
            global coefsG
            coefsG = coefs

        return render(request, 'tendencias/tendencia_infecxdia_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs})

    elif request.GET.get('Down') == 'Down':
        return some_view2(request, auxG, errorsT, metricsG, coefsG)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'tendencias/tendencia_infecxdia_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})


def tendencia_infeccion_pais(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        pais = request.POST.get("paisesS")
        param1 = request.POST.get("param1")
        param2 = request.POST.get("param2")
        param3 = request.POST.get("param3")
        if pais:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            global tempPais
            tempPais = pais
            for rows in jsonArray:
                for key in rows:
                    if key == pais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif param1 and param2 and param3:
            global parameters
            parameters = []
            parameters.append(param2)
            parameters.append(param3)
            graphs, errors, metrics, coefs = getGraph(param2, tempPais, param3,
                                                      param1, jsonArray)
            global auxG
            auxG = graphs
            global errorsT
            errorsT = errors
            errorsT.insert(0, ["Grado", "RMSE"])
            global metricsG
            metricsG = metrics
            global coefsG
            coefsG = coefs

        return render(request, 'tendencias/tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs})

    elif request.GET.get('Down') == 'Down':
        return some_view2(request, auxG, errorsT, metricsG, coefsG)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'tendencias/tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})


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


def reportPredicciones(request, graphs, errorsT, metrics, coefs, valPred, metricsL):

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
    # ----------------------------------------------------------
    t = Table(errorsT)
    t.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 6),
               ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'), ('INNERGRID',
                                                               (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ]))

    Elements.append(t)

    Elements.append(Image("data:image/png;base64," +
                    graphs[3], width=200, height=150))
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
    Elements.append(
        Paragraph("Ecuacion Polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(coefs[0], styles['Metrics']))
    Elements.append(
        Paragraph("Ecuacion Lineal", styles['Subtitulo']))
    Elements.append(Paragraph(coefs[1], styles['Metrics']))
    # ----------------------------------------------------------------------
    Elements.append(
        Paragraph("Metricas del Modelo Polinomial", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Lineal", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metricsL[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metricsL[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metricsL[2]), styles['Metrics']))

    Elements.append(Paragraph("""Los presentes modelos apoyan para generar predicciones estimadas a un plazos de tiempo estipulado
    buscando la mejor precision posible dependiendo del modelo seleccionado que en este caso se evaluado con un modelo lienal
    y polinomial, de los cuales se presentan ambas graficas predictivas respectivamente""", styles['Cuerpo']))

    Elements.append(Paragraph(
        """El valor de la prediccion generado por el modelo polinomial es de""" + str(valPred[0]), styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[5], width=200, height=150))
    Elements.append(Paragraph(
        """El valor de la prediccion generado por el modelo lineal es de""" + str(valPred[1]), styles['Cuerpo']))

    Elements.append(Image("data:image/png;base64," +
                    graphs[6], width=200, height=150))

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


def reportPredDouble(request, graphs, errorsT, errorsT2, metrics, metrics2, coefs, coefs2, valPred, valPred2, metricsL, metricsL2):

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
    # ----------------------------------------------------------

    t = Table(errorsT)
    t.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 6),
               ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'), ('INNERGRID',
                                                               (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    Elements.append(
        Paragraph("Tabla RSME del modelo Polinomial de Casos", styles['Subtitulo']))

    Elements.append(t)

    t2 = Table(errorsT2)
    t2.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 6),
                            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'), ('INNERGRID',
                                                                            (0, 0), (-1, -1), 0.25, colors.black),
                            ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                            ]))

    Elements.append(
        Paragraph("Tabla RSME del modelo Polinomial de Deaths", styles['Subtitulo']))
    Elements.append(t2)

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
    Elements.append(
        Paragraph("Ecuacion Polinomial de Casos", styles['Subtitulo']))
    Elements.append(Paragraph(coefs[0], styles['Metrics']))
    Elements.append(
        Paragraph("Ecuacion Lineal de Casos", styles['Subtitulo']))
    Elements.append(Paragraph(coefs[1], styles['Metrics']))

    Elements.append(
        Paragraph("Ecuacion Polinomial de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph(coefs2[0], styles['Metrics']))
    Elements.append(
        Paragraph("Ecuacion Lineal de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph(coefs2[1], styles['Metrics']))
    # ----------------------------------------------------------------------
    Elements.append(
        Paragraph("Metricas del Modelo Polinomial de Casos", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Polinomial de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metrics2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metrics2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metrics2[2]), styles['Metrics']))

    # ----------------------------------------------------------------------------
    Elements.append(
        Paragraph("Metricas del Modelo Lineal de Casos", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metricsL[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metricsL[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metricsL[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Lineal de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metricsL2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metricsL2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metricsL2[2]), styles['Metrics']))

    Elements.append(Paragraph("""Los presentes modelos apoyan para generar predicciones estimadas a un plazos de tiempo estipulado
    buscando la mejor precision posible dependiendo del modelo seleccionado que en este caso se evaluado con un modelo lienal
    y polinomial, de los cuales se presentan ambas graficas predictivas respectivamente""", styles['Cuerpo']))

    Elements.append(Paragraph(
        """El valor de la prediccion generado por el modelo polinomial de casos es de""" + str(valPred[0]), styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[5], width=200, height=150))
    Elements.append(Paragraph(
        """El valor de la prediccion generado por el modelo lineal de casos es de""" + str(valPred[1]), styles['Cuerpo']))

    Elements.append(Paragraph(
        """El valor de la prediccion generado por el modelo polinomial de muertes es de""" + str(valPred2[0]), styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[5], width=200, height=150))
    Elements.append(Paragraph(
        """El valor de la prediccion generado por el modelo lineal de muertes es de""" + str(valPred2[1]), styles['Cuerpo']))

    Elements.append(Image("data:image/png;base64," +
                    graphs[6], width=200, height=150))

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


def reportComp(request, graphs, errorsT, errorsT2, metrics, metrics2, coefs, coefs2, metricsL, metricsL2):

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
    # ----------------------------------------------------------

    t = Table(errorsT)
    t.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 6),
               ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'), ('INNERGRID',
                                                               (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    Elements.append(
        Paragraph("Tabla RSME del modelo Polinomial de Casos", styles['Subtitulo']))

    Elements.append(t)

    t2 = Table(errorsT2)
    t2.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 6),
                            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'), ('INNERGRID',
                                                                            (0, 0), (-1, -1), 0.25, colors.black),
                            ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                            ]))

    Elements.append(
        Paragraph("Tabla RSME del modelo Polinomial de Deaths", styles['Subtitulo']))
    Elements.append(t2)

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
    Elements.append(
        Paragraph("Ecuacion Polinomial de Casos", styles['Subtitulo']))
    Elements.append(Paragraph(coefs[0], styles['Metrics']))
    Elements.append(
        Paragraph("Ecuacion Lineal de Casos", styles['Subtitulo']))
    Elements.append(Paragraph(coefs[1], styles['Metrics']))

    Elements.append(
        Paragraph("Ecuacion Polinomial de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph(coefs2[0], styles['Metrics']))
    Elements.append(
        Paragraph("Ecuacion Lineal de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph(coefs2[1], styles['Metrics']))
    # ----------------------------------------------------------------------
    Elements.append(
        Paragraph("Metricas del Modelo Polinomial de Casos", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Polinomial de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metrics2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metrics2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metrics2[2]), styles['Metrics']))

    # ----------------------------------------------------------------------------
    Elements.append(
        Paragraph("Metricas del Modelo Lineal de Casos", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metricsL[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metricsL[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metricsL[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Lineal de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                    str(metricsL2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Raíz del Error Cuadrático Medio (RMSE) = " + str(metricsL2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                    str(metricsL2[2]), styles['Metrics']))

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


def reportComp2omas(request, graphs, countries):

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

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresión al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utilizó el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gráficas de dispersión de
    las variables de respuesta y regresoras y los resultados analíticos de los modelos. Las figuras, muestra el
    comportamiento gráfico de los modelos de regresión lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # --------------------------------------------------------------------------
    Elements.append(Paragraph("""Por supuesto el supuesto al utilizar un polinomio de grado 4 no es el adecuado ya que la informacion
    y datos son totalmente diferentes para os distintos casos por lo que se realizado un testeo y evaluacion para encontrat el grado
    optimo dados unos datos especificos, a continuacion se muestra una tabla con los resultados al evaluar distintos grados
    y la grafica para evidenciar el mejor grado que muestra un error menor""", styles['Cuerpo']))
    # ----------------------------------------------------------
    for country in countries:
        t = Table(country.errors)
        t.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 6),
                               ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'), ('INNERGRID',
                                                                               (0, 0), (-1, -1), 0.25, colors.black),
                               ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                               ]))
        Elements.append(
            Paragraph("Tabla RSME del modelo Polinomial de " + str(country.name), styles['Subtitulo']))

        Elements.append(t)

    Elements.append(Image("data:image/png;base64," +
                    graphs[1], width=200, height=150))

    # ------------------------------------------------------------------------------------------
    Elements.append(Paragraph("""Ya obtenido el grado que mejor se ajusta para los datos obtenidos se muestra la grafica que mejor se adopta
        y utiliza este grado optimo, ademas con la cual se pueden obtener las metricas que mas se acercan y reflejan
        resutlados adecuados.""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[2], width=200, height=150))

    # ------------------------------------------------------------------------------------------
    for country in countries:
        Elements.append(Paragraph("Metricas", styles['Subtitulo']))
        Elements.append(Paragraph("""Durente el calculo de todas las graficas el programa se generan distintas metricas y
            valores puntuales que ayudan a la generacion de las gracias como lo son coeficientes y errores que se expondran a continuacion.""", styles['Cuerpo']))
        Elements.append(
            Paragraph("Ecuacion Polinomial de Casos", styles['Subtitulo']))
        Elements.append(Paragraph(country.eq, styles['Metrics']))

        # ----------------------------------------------------------------------
        Elements.append(
            Paragraph("Metricas del Modelo Polinomial de " + str(country.name), styles['Subtitulo']))
        Elements.append(Paragraph("Error Cuadrático Medio (MSE) = " +
                        str(country.metrics[0]), styles['Metrics']))
        Elements.append(Paragraph(
            "Raíz del Error Cuadrático Medio (RMSE) = " + str(country.metrics[1]), styles['Metrics']))
        Elements.append(Paragraph("Coeficiente de Determinación R2 =  " +
                        str(country.metrics[2]), styles['Metrics']))

    # ----------------------------------------------------------------------------

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresión lineal simple y regresión
    polinomial de orden 4 y regresión múltiple para describir la relación entre la distorsión individual de
    datos de covid y la distorsión individual del paso del tiepo, siendo el modelo de regresión lineal
    múltiple el que mejor ajustó los datos de las mediciones del proceso, con mejor coeficiente de determinación R2""", styles['Cuerpo']))

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
