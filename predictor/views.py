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
from .ml_model.index_rates import muertes_regionn, porcentaje_muertess, tasa_casos_deaths, tasa_casos_casosA_deaths, porcentaje_edades, porcentaje_men, mortalidad, cases_deaths_rate
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import BaseDocTemplate, Frame, Paragraph, PageBreak, PageTemplate, NextPageTemplate, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus.tables import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from django.contrib import messages
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
        global enc
        try:
            csvReader = request.FILES['formFile']
            extension = csvReader.name.split(".")[1].lower()
            messages.success(request, "Se cargo el archivo correctamente")
        except:
            messages.error(request, "Error al cargar archivo")
            jsonArray = []
            enc = []
            return render(request, 'upload.html', {'enc': enc, 'data': jsonArray})

        if extension == "csv":
            global csvActual
            csvActual = csvReader
            decoded_file = csvReader.read().decode('utf-8').splitlines()
            reader = csv.DictReader(decoded_file)
            jsonArray = []
            for rows in reader:
                # for key in rows:
                for key in list(rows.keys()):
                    # if key is None or rows[key] is None:
                    if key is None:
                        del rows[key]

                jsonArray.append(rows)
        elif extension == "json":
            jsonArray = json.load(csvReader)
        elif extension == "xlsx" or extension == "xls":

            aux = pandas.read_excel(csvReader)
            jsonArray = aux.to_dict('records')

        enc = jsonArray[0].keys()

        return render(request, 'upload.html', {'enc': enc, 'data': jsonArray})
    else:
        return render(request, 'upload.html', {'enc': enc, 'data': jsonArray})


def clas_muni(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaReg = request.POST.get("columnaReg")
        columnaInfectados = request.POST.get("columnaInfectados")

        if columnaPais:
            graphs = []
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaReg and columnaInfectados:
            global auxG
            try:
                graphs = muertes_regionn(
                    tempPais, columnaReg, columnaInfectados, paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Error al cargar archivo")
                return render(request, 'analisis/clas_muni.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaReg or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'analisis/clas_muni.html', {'enc': enc,  'paises': paises})

        return render(request, 'analisis/clas_muni.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':
        try:
            return reportBarras(request, auxG)
        except:
            messages.error(request, "Error al cargar archivo")
            return render(request, 'analisis/clas_muni.html', {'enc': enc,  'paises': paises})
    else:

        return render(request, 'analisis/clas_muni.html', {'enc': enc,  'paises': paises})


def factores_muertes(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaReg = request.POST.get("columnaReg")
        columnaInfectados = request.POST.get("columnaInfectados")

        if columnaPais:
            graphs = []
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaReg and columnaInfectados:

            global auxG
            try:
                graphs = muertes_regionn(
                    tempPais, columnaReg, columnaInfectados, paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Error al analizar la informacion")
                return render(request, 'analisis/factores_muertes.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaReg or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'analisis/factores_muertes.html', {'enc': enc,  'paises': paises})

        return render(request, 'analisis/factores_muertes.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':

        return reportBarras(request, auxG)
    else:

        return render(request, 'analisis/factores_muertes.html', {'enc': enc,  'paises': paises})


def porcentaje_hombres(request):
    # if this is a POST request we need to process the form data
    paises = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaDeath = request.POST.get("columnaDeath")
        columnaInfectados = request.POST.get("columnaInfectados")

        if columnaPais:
            graphs = []
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaDeath and columnaInfectados:
            global auxG
            try:
                graphs = porcentaje_men(
                    tempPais, columnaInfectados, columnaDeath, paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(
                    request, "Se analizaron correctamente los datos")
            except:
                messages.error(request, "Error al analizar la informacion")
                return render(request, 'indices_tasas/porcentaje_hombres.html', {'enc': enc,  'paises': paises})
        elif not columnaPais or not paisEspecifico or not columnaDeath or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'indices_tasas/porcentaje_hombres.html', {'enc': enc,  'paises': paises})

        return render(request, 'indices_tasas/porcentaje_hombres.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':

        return reportPie(request, auxG)
    else:

        return render(request, 'indices_tasas/porcentaje_hombres.html', {'enc': enc,  'paises': paises})


def muertes_edad(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaDeath = request.POST.get("columnaDeath")
        columnaInfectados = request.POST.get("columnaInfectados")

        if columnaPais:
            graphs = []
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaDeath and columnaInfectados:
            global auxG
            try:
                graphs = porcentaje_edades(
                    tempPais, columnaInfectados, columnaDeath, paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(
                    request, "Se analizo la informacion correctamente")
            except:
                messages.error(request, "Alguno de los campos posee un error")
                return render(request, 'indices_tasas/muertes_edad.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaDeath or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'indices_tasas/muertes_edad.html', {'enc': enc,  'paises': paises})

        return render(request, 'indices_tasas/muertes_edad.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':

        return reportPie(request, auxG)
    else:

        return render(request, 'indices_tasas/muertes_edad.html', {'enc': enc,  'paises': paises})


def tasa_casos_casosdia_muertos(request):
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
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados and columnaDeath:
            global auxG
            try:
                graphs = tasa_casos_casosA_deaths(
                    columnaFecha, tempPais, columnaInfectados, columnaDeath, paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(request, "La data se analizo correctamente")
            except:
                messages.error(request, "Alguno de los campos posee un error")
                return render(request, 'indices_tasas/tasa_casos_casosdia_muertos.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados or not columnaDeath:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'indices_tasas/tasa_casos_casosdia_muertos.html', {'enc': enc,  'paises': paises})

        return render(request, 'indices_tasas/tasa_casos_casosdia_muertos.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':

        return reportPie(request, auxG)
    else:

        return render(request, 'indices_tasas/tasa_casos_casosdia_muertos.html', {'enc': enc,  'paises': paises})


def tasa_casos_muertes(request):
    # if this is a POST request we need to process the form data
    paises = []

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
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaInfectados and columnaDeath and columnaFecha:
            global auxG
            global errorsT
            global errorsT2
            global metricsG
            global metrics2G
            global coefsG
            global coefs2G
            try:
                graphs = cases_deaths_rate(
                    tempPais, columnaFecha, columnaInfectados, columnaDeath, paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Alguno de los campos posee un error")
                return render(request, 'indices_tasas/tasa_casos_muertes.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaInfectados or not columnaDeath and not columnaFecha:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'indices_tasas/tasa_casos_muertes.html', {'enc': enc,  'paises': paises})

        return render(request, 'indices_tasas/tasa_casos_muertes.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':

        return reportPie(request, auxG)
        # return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'indices_tasas/tasa_casos_muertes.html', {'enc': enc,  'paises': paises})


def porcentaje_muertes(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaDeath = request.POST.get("columnaDeath")
        columnaInfectados = request.POST.get("columnaInfectados")

        if columnaPais:
            graphs = []
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaDeath and columnaInfectados:
            global auxG
            try:
                graphs = porcentaje_muertess(
                    tempPais, columnaInfectados, columnaDeath, paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(request, "Data analizada correctamente")
            except:
                messages.error(request, "Alguno de los campos posee un error")
                return render(request, 'indices_tasas/porcentaje_muertes.html', {'enc': enc,  'paises': paises})
        elif not columnaPais or not paisEspecifico or not columnaDeath or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'indices_tasas/porcentaje_muertes.html', {'enc': enc,  'paises': paises})

        return render(request, 'indices_tasas/porcentaje_muertes.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':

        return reportPie(request, auxG)
    else:

        return render(request, 'indices_tasas/porcentaje_muertes.html', {'enc': enc,  'paises': paises})


def muertes_region(request):
    # if this is a POST request we need to process the form data
    paises = []
    params = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaReg = request.POST.get("columnaReg")
        columnaInfectados = request.POST.get("columnaInfectados")

        if columnaPais:
            graphs = []
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaReg and columnaInfectados:
            global auxG
            try:
                graphs = muertes_regionn(
                    tempPais, columnaReg, columnaInfectados, paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(
                    request, "Se analizaron los datos correctamente")
            except:
                messages.error(request, "Alguno de los campos posee un error")
                return render(request, 'indices_tasas/muertes_region.html', {'enc': enc,  'paises': paises})
        elif not columnaPais or not paisEspecifico or not columnaReg or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'indices_tasas/muertes_region.html', {'enc': enc,  'paises': paises})

        return render(request, 'indices_tasas/muertes_region.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':

        return reportBarras(request, auxG)
    else:

        return render(request, 'indices_tasas/muertes_region.html', {'enc': enc,  'paises': paises})


def indice_pand(request):
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
            metrics = ""
            coefs = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados:
            global auxG
            global errorsT
            global metricsG
            global coefsG
            try:
                graphs, errors, metrics, coefs = getGraph(columnaFecha, tempPais, columnaInfectados,
                                                          paisEspecifico, jsonArray)
                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                coefsG = coefs
                messages.success(request, "Data analizada correctamente")
            except:
                messages.error(request, "Error al analizar la data")
                return render(request, 'indices_tasas/indice_pand.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'indices_tasas/indice_pand.html', {'enc': enc,  'paises': paises})

        return render(request, 'indices_tasas/indice_pand.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs, })

    elif request.GET.get('Down') == 'Down':

        return some_view2(request, auxG, errorsT, metricsG, coefsG)
    else:

        return render(request, 'indices_tasas/indice_pand.html', {'enc': enc,  'paises': paises})


def tasa_mortalidad(request):
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
            metrics = ""
            coefs = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados:
            global auxG
            try:

                global errorsT
                global metricsG
                global coefsG
                graphs = mortalidad(tempPais, columnaInfectados, columnaFecha,
                                    paisEspecifico, jsonArray)
                auxG = graphs
                messages.success(
                    request, "La inofrmacion se analizo correctamente")
            except:
                messages.error(request, "Error al analizar la informacion")
                return render(request, 'indices_tasas/tasa_mortalidad.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'indices_tasas/tasa_mortalidad.html', {'enc': enc,  'paises': paises})

        return render(request, 'indices_tasas/tasa_mortalidad.html', {'enc': enc, 'paises': paises, 'graphs': graphs})

    elif request.GET.get('Down') == 'Down':

        return reportPie(request, auxG)
        # return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'indices_tasas/tasa_mortalidad.html', {'enc': enc,  'paises': paises})


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
            global auxG
            global countriesG
            try:
                graphs, countries = getComparacion2Paises(
                    columnaFecha, tempPais, columnaInfectados, listapaises, jsonArray)
                listapaises = []
                auxG = graphs
                countriesG = countries
                messages.success(request, "Se analizo correctamente")
            except:
                messages.error(request, "Error al analizar la data")
                return render(request, 'analisis/comp_vac.html', {'enc': enc,  'paises': paises})
        elif not columnaPais or not paisEspecifico or not pais2Especifico or not columnaFecha or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'analisis/comp_vac.html', {'enc': enc,  'paises': paises})

        return render(request, 'analisis/comp_vac.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'countries': countries, 'listapaises': listapaises})

    elif request.GET.get('Down') == 'Down':

        return reportComp2omas(request, auxG, countriesG)
        # return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'analisis/comp_vac.html', {'enc': enc,  'paises': paises})


def ana_deaths_pais(request):
    # if this is a POST request we need to process the form data
    paises = []

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        columnaPais = request.POST.get("columnaPais")
        paisEspecifico = request.POST.get("paisEspecifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")

        if columnaPais:
            graphs = []
            errors = []
            metrics = ""
            coefs = ""
            global tempPais
            tempPais = columnaPais
            for rows in jsonArray:
                for key in rows:
                    if key == columnaPais:
                        if not(rows[key] in paises):
                            paises.append(rows[key])

        elif paisEspecifico and columnaFecha and columnaInfectados:
            global auxG
            global errorsT
            global metricsG
            global coefsG
            try:

                graphs, errors, metrics, coefs = getGraph(columnaFecha, tempPais, columnaInfectados,
                                                          paisEspecifico, jsonArray)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                coefsG = coefs
                messages.success(
                    request, "Se analizo la informacion correctamente")
            except:
                messages.error(request, "Error al analizar la informacion")
                return render(request, 'analisis/ana_deaths_pais.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'analisis/ana_deaths_pais.html', {'enc': enc,  'paises': paises})

        return render(request, 'analisis/ana_deaths_pais.html', {'enc': enc, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics, 'coefs': coefs, })

    elif request.GET.get('Down') == 'Down':

        return some_view2(request, auxG, errorsT, metricsG, coefsG)
        # return reportPredicciones(request, auxG, errorsT, metricsG, coefsG, valorPredG, metricsGL)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'analisis/ana_deaths_pais.html', {'enc': enc,  'paises': paises})


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
            global auxG
            global countriesG
            try:
                graphs, countries = getComparacion2Paises(
                    columnaFecha, tempPais, columnaInfectados, listapaises, jsonArray)
                listapaises = []
                auxG = graphs
                countriesG = countries
                messages.success(request, "Se analizo la data con exito")
            except:
                messages.error(request, "Error al analizar la data")
                return render(request, 'analisis/comp_2omas.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'analisis/comp_2omas.html', {'enc': enc,  'paises': paises})

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
            global auxG
            global errorsT
            global errorsT2
            global metricsG
            global metrics2G
            global metricsGL
            global metrics2GL
            global coefsG
            global coefs2G
            try:
                graphs, errors, errors2, metrics, metrics2, coefs, coefs2, metricsL, metricsLD = getComparacion(columnaFecha, tempPais, columnaInfectados, columnaDeath,
                                                                                                                paisEspecifico, jsonArray)

                auxG = graphs
                errorsT = errors
                errorsT2 = errors2
                errorsT2.insert(0, ["Grado", "RMSE"])
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics2
                metrics2G = metrics2
                metricsGL = metricsL
                metrics2GL = metricsLD
                coefsG = coefs
                coefs2G = coefs2
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Error al analizar la informacion")
                return render(request, 'analisis/comp_casos_pruebas.html', {'enc': enc,  'paises': paises})
        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados or not columnaDeath:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'analisis/comp_casos_pruebas.html', {'enc': enc,  'paises': paises})

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
            global auxG
            global errorsT
            global metricsG
            global coefsG
            global valorPredG
            global metricsGL
            try:

                graphs, errors, metrics, coefs, valorPred, metricsL = getPrediction(columnaFecha, tempPais, columnaInfectados,
                                                                                    paisEspecifico, jsonArray, valorPrediccion)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                coefsG = coefs
                valorPredG = valorPred
                metricsGL = metricsL
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Error al analizar la inforamcion")
                return render(request, 'predicciones/pred_infectados_pais.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados or not valorPrediccion:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'predicciones/pred_infectados_pais.html', {'enc': enc,  'paises': paises})

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

        columnaDepa = request.POST.get("columnaDepa")
        paisEspecifico = request.POST.get("paisEspecifico")
        depaEspecifico = request.POST.get("depaEspecifico")
        columnaFecha = request.POST.get("columnaFecha")
        columnaInfectados = request.POST.get("columnaInfectados")
        valorPrediccion = request.POST.get("valorPrediccion")

        if columnaDepa:
            graphs = []
            errors = []
            coefs = ""
            metrics = ""
            valorPred = ""
            metricsL = ""
            global tempDepa
            tempDepa = columnaDepa
            for rows in jsonArray:
                if not(rows[columnaDepa] in depas):
                    depas.append(rows[tempDepa])

        elif depaEspecifico and columnaFecha and columnaInfectados and valorPrediccion:
            global auxG
            global errorsT
            global metricsG
            global metricsGL
            global coefsG
            global valorPredG
            try:
                graphs, errors, metrics, coefs, valorPred, metricsL = getPredictionDepa(columnaFecha, tempDepa, columnaInfectados,
                                                                                        depaEspecifico, jsonArray, valorPrediccion)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                metricsGL = metricsL
                coefsG = coefs
                valorPredG = valorPred
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Error al analizar la informacion")
                return render(request, 'predicciones/pred_mortalidad_depa.html', {'enc': enc,  'paises': paises})
        elif not columnaDepa or not paisEspecifico or not depaEspecifico or not columnaFecha or not columnaInfectados or not valorPrediccion:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'predicciones/pred_mortalidad_depa.html', {'enc': enc,  'paises': paises})

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
            global auxG
            global errorsT
            global metricsG
            global metricsGL
            global coefsG
            global valorPredG
            try:
                graphs, errors, metrics, coefs, valorPred, metricsL = getPrediction(columnaFecha, tempPais, columnaInfectados,
                                                                                    paisEspecifico, jsonArray, valorPrediccion)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                metricsGL = metricsL
                coefsG = coefs
                valorPredG = valorPred
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Error al analizar los datos")
                return render(request, 'predicciones/pred_mortalidad_pais.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados or not valorPrediccion:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'predicciones/pred_mortalidad_pais.html', {'enc': enc,  'paises': paises})

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
            global auxG
            global errorsT
            global metricsG
            global metricsGL
            global coefsG
            global valorPredG
            try:
                graphs, errors, metrics, coefs, valorPred, metricsL = getPrediction(columnaFecha, tempPais, columnaInfectados,
                                                                                    paisEspecifico, jsonArray, valorPrediccion)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                metricsGL = metricsL
                coefsG = coefs
                valorPredG = valorPred
                messages.success(
                    request, "Se analizo la informacion correctamente")
            except:
                messages.error(request, "Error al analizar la data")
                return render(request, 'predicciones/pred_casos_pais.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados or not valorPrediccion:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'predicciones/pred_casos_pais.html', {'enc': enc,  'paises': paises})

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
            global auxG
            global errorsT
            global metricsG
            global metricsGL
            global coefsG
            global valorPredG
            try:
                graphs, errors, metrics, coefs, valorPred, metricsL = getPrediction(columnaFecha, tempPais, columnaInfectados,
                                                                                    paisEspecifico, jsonArray, valorPrediccion)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                metricsGL = metricsL
                coefsG = coefs
                valorPredG = valorPred
                messages.success(
                    request, "Se analizo la informacion correctamente")
            except:
                messages.error(request, "Error al analizar los datos")
                return render(request, 'predicciones/pred_casos_dia.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados or not valorPrediccion:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'predicciones/pred_casos_dia.html', {'enc': enc,  'paises': paises})

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
            global auxG
            global errorsT
            global errorsT2
            global metricsG
            global metrics2G
            global metricsGL
            global metrics2GL
            global coefsG
            global coefs2G
            global valorPredG
            global valorPred2G
            try:
                graphs, errors, errors2, metrics, metrics2, coefs, coefs2, valorPred, valorPredD, metricsL, metricsLD = getDoublePrediction(columnaFecha, tempPais, columnaInfectados, columnaDeath,
                                                                                                                                            paisEspecifico, jsonArray, valorPrediccion)

                auxG = graphs
                errorsT = errors
                errorsT2 = errors2
                errorsT2.insert(0, ["Grado", "RMSE"])
                errorsT.insert(0, ["Grado", "RMSE"])

                metricsG = metrics
                metrics2G = metrics2
                metricsGL = metricsL
                metrics2GL = metricsLD
                coefsG = coefs
                coefs2G = coefs2
                valorPredG = valorPred
                valorPred2G = valorPredD
                messages.success(
                    request, "Se analizaron los datos correctamente")
            except:
                messages.error(request, "Error al analizar la informacion")
                return render(request, 'predicciones/pred_casos_muertes.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados or not columnaDeath or not valorPrediccion:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'predicciones/pred_casos_muertes.html', {'enc': enc,  'paises': paises})

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
            global auxG
            global errorsT
            global metricsG
            global metricsGL
            global coefsG
            global valorPredG
            try:
                graphs, errors, metrics, coefs, valorPred, metricsL = getPredictionLastDay(columnaFecha, tempPais, columnaInfectados,
                                                                                           paisEspecifico, jsonArray)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                metricsGL = metricsL
                coefsG = coefs
                valorPredG = valorPred
                messages.success(
                    request, "Se analizo la inforamcion correctamente")
            except:
                messages.error(request, "Error al analizar la data")
                return render(request, 'predicciones/pred_ultimo_primero.html', {'enc': enc,  'paises': paises})

        elif not columnaPais or not paisEspecifico or not columnaFecha or not columnaInfectados:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'predicciones/pred_ultimo_primero.html', {'enc': enc,  'paises': paises})

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
        global paisess
        global parameters
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

            parameters = []
            parameters.append(param2)
            parameters.append(param3)
            global auxG
            global errorsT
            global metricsG
            global coefsG
            try:
                graphs, errors, metrics, coefs = getGraph2(param2, tempPais, tempDepa, param3,
                                                           paisNN, depaa, jsonArray)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                coefsG = coefs
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Error al analizar la data")
                return render(request, 'tendencias/tendencia_casos_depa.html', {'enc': enc, 'parameters': parameters, 'paises': paisess})
        elif not pais or not depa or not paisN or not depaa or not param2 or not param3:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'tendencias/tendencia_casos_depa.html', {'enc': enc, 'parameters': parameters, 'paises': paisess})

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
    global parameters
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

            parameters = []
            parameters.append(param2)
            parameters.append(param3)
            global auxG
            global errorsT
            global metricsG
            global coefsG
            try:
                graphs, errors, metrics, coefs = getGraph(param2, tempPais, param3,
                                                          param1, jsonArray)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                coefsG = coefs
                messages.success(
                    request, "Alguno de los campos posee un error")
            except:
                messages.error(request, "Alguno de los campos posee un error")
                return render(request, 'tendencias/tendencia_vacuancion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
        elif not pais or not param1 or not param2 or not param3:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'tendencias/tendencia_vacuancion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})

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
    global parameters
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

            parameters = []
            parameters.append(param2)
            parameters.append(param3)
            global auxG
            global errorsT
            global metricsG
            global coefsG
            try:
                graphs, errors, metrics, coefs = getGraph(param2, tempPais, param3,
                                                          param1, jsonArray)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                coefsG = coefs
                messages.success(request, "Data analizada")
            except:
                messages.error(request, "Error en data")
                return render(request, 'tendencias/tendencia_infecxdia_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})

        elif not pais or not param1 or not param2 or not param3:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request, 'tendencias/tendencia_infecxdia_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})

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
    global parameters
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

            parameters = []
            parameters.append(param2)
            parameters.append(param3)
            global auxG
            global errorsT
            global metricsG
            global coefsG
            try:
                graphs, errors, metrics, coefs = getGraph(param2, tempPais, param3,
                                                          param1, jsonArray)

                auxG = graphs
                errorsT = errors
                errorsT.insert(0, ["Grado", "RMSE"])
                metricsG = metrics
                coefsG = coefs
                messages.success(request, "Se analizo la data correctamente")
            except:
                messages.error(request, "Error al analizar la informacion")
                return render(request,  'tendencias/tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})

        elif not pais or not param1 or not param2 or not param3:
            messages.error(request, "Alguno de los campos posee un error")
            return render(request,  'tendencias/tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})

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
        Paragraph("Modelo de regresi??n lineal simple", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n lineal son ampliamente usados en la ingenier??a ya que sirven para analizar el
    comportamiento de las variables de entrada (o regresora) y salida (o respuesta) estableciendo predicciones y
    estimaciones. En este trabajo la variable regresora corresponde a la distorsi??n arm??nica individual de corriente y
    la variable de respuesta corresponde a la distorsi??n arm??nica individual de tensi??n.
    La ecuaci??n, muestra la representaci??n de un modelo de regresi??n lineal simple, donde Y es la respuesta, X es la
    variable regresora, 0 y 1 son los par??metros del modelo o coeficientes de regresi??n y es el error del modelo""", styles['Cuerpo']))
    Elements.append(
        Paragraph("El coeficiente de determinaci??n R2 se expresa como un porcentaje que indica la variaci??n de los valores de la variable independiente que se puede explicar con la ecuaci??n de regresi??n", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> X <greek>e</greek>""", styles['Equations']))
    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Modelo de regresi??n polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n polinomial se usan cuando la variable de respuesta muestra un comportamiento curvil??neo
    o no lineal. La ecuaci??n, describe el modelo de regresi??n polinomial de orden k en una variable regresora y la
    ecuaci??n, muestra el modelo ajustado de regresi??n polinomial de orden k. Los estimadores de los par??metros del
    modelo se obtienen por el m??todo de los m??nimos cuadrados usando la ecuaci??n, donde y, X, X??? son vectores. En
    este trabajo se aplican los modelos de regresi??n polinomial de orden 2 y orden 3 en una variable regresora""", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> + <greek>b</greek><sub>2</sub> X<super>2</super> + ... + 
        <greek>b</greek><sub>k</sub> X<super>k</super>""", styles['Equations']))
    # Coeficiente de deteriminacion R2
    Elements.append(
        Paragraph("Coeficiente de determinaci??n R<super>2</super>", styles['Subtitulo']))
    Elements.append(Paragraph("""El coeficiente de determinaci??n R2 mide la proporci??n de la variaci??n de la respuesta Y que es explicada por el
    modelo de regresi??n. El coeficiente R2 se calcula usando la ecuaci??n, donde SSR es la medida de variabilidad
    del modelo de regresi??n y SST corresponde a la medida de variabilidad de Y sin considerar el efecto de la variable
    regresora X.""", styles['Cuerpo']))
    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distintas dispersiones y ploteos 
                              de cada caso en especifico, siendo evidente la necesidad de aplicar otro modelo diferente a una regresion lienal""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresi??n al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utiliz?? el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gr??ficas de dispersi??n de
    las variables de respuesta y regresoras y los resultados anal??ticos de los modelos. Las figuras, muestra el
    comportamiento gr??fico de los modelos de regresi??n lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

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
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresi??n lineal simple y regresi??n
    polinomial de orden 4 y regresi??n m??ltiple para describir la relaci??n entre la distorsi??n individual de
    datos de covid y la distorsi??n individual del paso del tiepo, siendo el modelo de regresi??n lineal
    m??ltiple el que mejor ajust?? los datos de las mediciones del proceso, con mejor coeficiente de determinaci??n R2
    ("""+str(metrics[2])+""").""", styles['Cuerpo']))

    Elements.append(Paragraph("""Los pron??sticos realizados con el modelo de regresi??n lineal m??ltiple, permiten estimar la variacion
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
        Paragraph("Modelo de regresi??n lineal simple", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n lineal son ampliamente usados en la ingenier??a ya que sirven para analizar el
    comportamiento de las variables de entrada (o regresora) y salida (o respuesta) estableciendo predicciones y
    estimaciones. En este trabajo la variable regresora corresponde a la distorsi??n arm??nica individual de corriente y
    la variable de respuesta corresponde a la distorsi??n arm??nica individual de tensi??n.
    La ecuaci??n, muestra la representaci??n de un modelo de regresi??n lineal simple, donde Y es la respuesta, X es la
    variable regresora, 0 y 1 son los par??metros del modelo o coeficientes de regresi??n y es el error del modelo""", styles['Cuerpo']))
    Elements.append(
        Paragraph("El coeficiente de determinaci??n R2 se expresa como un porcentaje que indica la variaci??n de los valores de la variable independiente que se puede explicar con la ecuaci??n de regresi??n", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> X <greek>e</greek>""", styles['Equations']))
    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Modelo de regresi??n polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n polinomial se usan cuando la variable de respuesta muestra un comportamiento curvil??neo
    o no lineal. La ecuaci??n, describe el modelo de regresi??n polinomial de orden k en una variable regresora y la
    ecuaci??n, muestra el modelo ajustado de regresi??n polinomial de orden k. Los estimadores de los par??metros del
    modelo se obtienen por el m??todo de los m??nimos cuadrados usando la ecuaci??n, donde y, X, X??? son vectores. En
    este trabajo se aplican los modelos de regresi??n polinomial de orden 2 y orden 3 en una variable regresora""", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> + <greek>b</greek><sub>2</sub> X<super>2</super> + ... + 
        <greek>b</greek><sub>k</sub> X<super>k</super>""", styles['Equations']))
    # Coeficiente de deteriminacion R2
    Elements.append(
        Paragraph("Coeficiente de determinaci??n R<super>2</super>", styles['Subtitulo']))
    Elements.append(Paragraph("""El coeficiente de determinaci??n R2 mide la proporci??n de la variaci??n de la respuesta Y que es explicada por el
    modelo de regresi??n. El coeficiente R2 se calcula usando la ecuaci??n, donde SSR es la medida de variabilidad
    del modelo de regresi??n y SST corresponde a la medida de variabilidad de Y sin considerar el efecto de la variable
    regresora X.""", styles['Cuerpo']))
    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distintas dispersiones y ploteos 
                              de cada caso en especifico, siendo evidente la necesidad de aplicar otro modelo diferente a una regresion lienal""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresi??n al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utiliz?? el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gr??ficas de dispersi??n de
    las variables de respuesta y regresoras y los resultados anal??ticos de los modelos. Las figuras, muestra el
    comportamiento gr??fico de los modelos de regresi??n lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

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
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Lineal", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metricsL[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metricsL[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
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

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresi??n lineal simple y regresi??n
    polinomial de orden 4 y regresi??n m??ltiple para describir la relaci??n entre la distorsi??n individual de
    datos de covid y la distorsi??n individual del paso del tiepo, siendo el modelo de regresi??n lineal
    m??ltiple el que mejor ajust?? los datos de las mediciones del proceso, con mejor coeficiente de determinaci??n R2
    ("""+str(metrics[2])+""").""", styles['Cuerpo']))

    Elements.append(Paragraph("""Los pron??sticos realizados con el modelo de regresi??n lineal m??ltiple, permiten estimar la variacion
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
        Paragraph("Modelo de regresi??n lineal simple", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n lineal son ampliamente usados en la ingenier??a ya que sirven para analizar el
    comportamiento de las variables de entrada (o regresora) y salida (o respuesta) estableciendo predicciones y
    estimaciones. En este trabajo la variable regresora corresponde a la distorsi??n arm??nica individual de corriente y
    la variable de respuesta corresponde a la distorsi??n arm??nica individual de tensi??n.
    La ecuaci??n, muestra la representaci??n de un modelo de regresi??n lineal simple, donde Y es la respuesta, X es la
    variable regresora, 0 y 1 son los par??metros del modelo o coeficientes de regresi??n y es el error del modelo""", styles['Cuerpo']))
    Elements.append(
        Paragraph("El coeficiente de determinaci??n R2 se expresa como un porcentaje que indica la variaci??n de los valores de la variable independiente que se puede explicar con la ecuaci??n de regresi??n", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> X <greek>e</greek>""", styles['Equations']))
    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Modelo de regresi??n polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n polinomial se usan cuando la variable de respuesta muestra un comportamiento curvil??neo
    o no lineal. La ecuaci??n, describe el modelo de regresi??n polinomial de orden k en una variable regresora y la
    ecuaci??n, muestra el modelo ajustado de regresi??n polinomial de orden k. Los estimadores de los par??metros del
    modelo se obtienen por el m??todo de los m??nimos cuadrados usando la ecuaci??n, donde y, X, X??? son vectores. En
    este trabajo se aplican los modelos de regresi??n polinomial de orden 2 y orden 3 en una variable regresora""", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> + <greek>b</greek><sub>2</sub> X<super>2</super> + ... + 
        <greek>b</greek><sub>k</sub> X<super>k</super>""", styles['Equations']))
    # Coeficiente de deteriminacion R2
    Elements.append(
        Paragraph("Coeficiente de determinaci??n R<super>2</super>", styles['Subtitulo']))
    Elements.append(Paragraph("""El coeficiente de determinaci??n R2 mide la proporci??n de la variaci??n de la respuesta Y que es explicada por el
    modelo de regresi??n. El coeficiente R2 se calcula usando la ecuaci??n, donde SSR es la medida de variabilidad
    del modelo de regresi??n y SST corresponde a la medida de variabilidad de Y sin considerar el efecto de la variable
    regresora X.""", styles['Cuerpo']))
    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distintas dispersiones y ploteos 
                              de cada caso en especifico, siendo evidente la necesidad de aplicar otro modelo diferente a una regresion lienal""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresi??n al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utiliz?? el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gr??ficas de dispersi??n de
    las variables de respuesta y regresoras y los resultados anal??ticos de los modelos. Las figuras, muestra el
    comportamiento gr??fico de los modelos de regresi??n lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

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
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Polinomial de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metrics2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metrics2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metrics2[2]), styles['Metrics']))

    # ----------------------------------------------------------------------------
    Elements.append(
        Paragraph("Metricas del Modelo Lineal de Casos", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metricsL[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metricsL[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metricsL[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Lineal de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metricsL2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metricsL2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
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

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresi??n lineal simple y regresi??n
    polinomial de orden 4 y regresi??n m??ltiple para describir la relaci??n entre la distorsi??n individual de
    datos de covid y la distorsi??n individual del paso del tiepo, siendo el modelo de regresi??n lineal
    m??ltiple el que mejor ajust?? los datos de las mediciones del proceso, con mejor coeficiente de determinaci??n R2
    ("""+str(metrics[2])+""").""", styles['Cuerpo']))

    Elements.append(Paragraph("""Los pron??sticos realizados con el modelo de regresi??n lineal m??ltiple, permiten estimar la variacion
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
        Paragraph("Modelo de regresi??n lineal simple", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n lineal son ampliamente usados en la ingenier??a ya que sirven para analizar el
    comportamiento de las variables de entrada (o regresora) y salida (o respuesta) estableciendo predicciones y
    estimaciones. En este trabajo la variable regresora corresponde a la distorsi??n arm??nica individual de corriente y
    la variable de respuesta corresponde a la distorsi??n arm??nica individual de tensi??n.
    La ecuaci??n, muestra la representaci??n de un modelo de regresi??n lineal simple, donde Y es la respuesta, X es la
    variable regresora, 0 y 1 son los par??metros del modelo o coeficientes de regresi??n y es el error del modelo""", styles['Cuerpo']))
    Elements.append(
        Paragraph("El coeficiente de determinaci??n R2 se expresa como un porcentaje que indica la variaci??n de los valores de la variable independiente que se puede explicar con la ecuaci??n de regresi??n", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> X <greek>e</greek>""", styles['Equations']))
    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Modelo de regresi??n polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n polinomial se usan cuando la variable de respuesta muestra un comportamiento curvil??neo
    o no lineal. La ecuaci??n, describe el modelo de regresi??n polinomial de orden k en una variable regresora y la
    ecuaci??n, muestra el modelo ajustado de regresi??n polinomial de orden k. Los estimadores de los par??metros del
    modelo se obtienen por el m??todo de los m??nimos cuadrados usando la ecuaci??n, donde y, X, X??? son vectores. En
    este trabajo se aplican los modelos de regresi??n polinomial de orden 2 y orden 3 en una variable regresora""", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> + <greek>b</greek><sub>2</sub> X<super>2</super> + ... + 
        <greek>b</greek><sub>k</sub> X<super>k</super>""", styles['Equations']))
    # Coeficiente de deteriminacion R2
    Elements.append(
        Paragraph("Coeficiente de determinaci??n R<super>2</super>", styles['Subtitulo']))
    Elements.append(Paragraph("""El coeficiente de determinaci??n R2 mide la proporci??n de la variaci??n de la respuesta Y que es explicada por el
    modelo de regresi??n. El coeficiente R2 se calcula usando la ecuaci??n, donde SSR es la medida de variabilidad
    del modelo de regresi??n y SST corresponde a la medida de variabilidad de Y sin considerar el efecto de la variable
    regresora X.""", styles['Cuerpo']))
    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distintas dispersiones y ploteos 
                              de cada caso en especifico, siendo evidente la necesidad de aplicar otro modelo diferente a una regresion lienal""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresi??n al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utiliz?? el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gr??ficas de dispersi??n de
    las variables de respuesta y regresoras y los resultados anal??ticos de los modelos. Las figuras, muestra el
    comportamiento gr??fico de los modelos de regresi??n lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

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
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Polinomial de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metrics2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metrics2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metrics2[2]), styles['Metrics']))

    # ----------------------------------------------------------------------------
    Elements.append(
        Paragraph("Metricas del Modelo Lineal de Casos", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metricsL[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metricsL[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metricsL[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Lineal de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metricsL2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metricsL2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metricsL2[2]), styles['Metrics']))

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresi??n lineal simple y regresi??n
    polinomial de orden 4 y regresi??n m??ltiple para describir la relaci??n entre la distorsi??n individual de
    datos de covid y la distorsi??n individual del paso del tiepo, siendo el modelo de regresi??n lineal
    m??ltiple el que mejor ajust?? los datos de las mediciones del proceso, con mejor coeficiente de determinaci??n R2
    ("""+str(metrics[2])+""").""", styles['Cuerpo']))

    Elements.append(Paragraph("""Los pron??sticos realizados con el modelo de regresi??n lineal m??ltiple, permiten estimar la variacion
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
        Paragraph("Modelo de regresi??n lineal simple", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n lineal son ampliamente usados en la ingenier??a ya que sirven para analizar el
    comportamiento de las variables de entrada (o regresora) y salida (o respuesta) estableciendo predicciones y
    estimaciones. En este trabajo la variable regresora corresponde a la distorsi??n arm??nica individual de corriente y
    la variable de respuesta corresponde a la distorsi??n arm??nica individual de tensi??n.
    La ecuaci??n, muestra la representaci??n de un modelo de regresi??n lineal simple, donde Y es la respuesta, X es la
    variable regresora, 0 y 1 son los par??metros del modelo o coeficientes de regresi??n y es el error del modelo""", styles['Cuerpo']))
    Elements.append(
        Paragraph("El coeficiente de determinaci??n R2 se expresa como un porcentaje que indica la variaci??n de los valores de la variable independiente que se puede explicar con la ecuaci??n de regresi??n", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> X <greek>e</greek>""", styles['Equations']))
    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Modelo de regresi??n polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n polinomial se usan cuando la variable de respuesta muestra un comportamiento curvil??neo
    o no lineal. La ecuaci??n, describe el modelo de regresi??n polinomial de orden k en una variable regresora y la
    ecuaci??n, muestra el modelo ajustado de regresi??n polinomial de orden k. Los estimadores de los par??metros del
    modelo se obtienen por el m??todo de los m??nimos cuadrados usando la ecuaci??n, donde y, X, X??? son vectores. En
    este trabajo se aplican los modelos de regresi??n polinomial de orden 2 y orden 3 en una variable regresora""", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> + <greek>b</greek><sub>2</sub> X<super>2</super> + ... + 
        <greek>b</greek><sub>k</sub> X<super>k</super>""", styles['Equations']))
    # Coeficiente de deteriminacion R2
    Elements.append(
        Paragraph("Coeficiente de determinaci??n R<super>2</super>", styles['Subtitulo']))
    Elements.append(Paragraph("""El coeficiente de determinaci??n R2 mide la proporci??n de la variaci??n de la respuesta Y que es explicada por el
    modelo de regresi??n. El coeficiente R2 se calcula usando la ecuaci??n, donde SSR es la medida de variabilidad
    del modelo de regresi??n y SST corresponde a la medida de variabilidad de Y sin considerar el efecto de la variable
    regresora X.""", styles['Cuerpo']))
    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distintas dispersiones y ploteos 
                              de cada caso en especifico, siendo evidente la necesidad de aplicar otro modelo diferente a una regresion lienal""", styles['Cuerpo']))

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresi??n al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utiliz?? el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gr??ficas de dispersi??n de
    las variables de respuesta y regresoras y los resultados anal??ticos de los modelos. Las figuras, muestra el
    comportamiento gr??fico de los modelos de regresi??n lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

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
        Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                        str(country.metrics[0]), styles['Metrics']))
        Elements.append(Paragraph(
            "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(country.metrics[1]), styles['Metrics']))
        Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                        str(country.metrics[2]), styles['Metrics']))

    # ----------------------------------------------------------------------------

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresi??n lineal simple y regresi??n
    polinomial de orden 4 y regresi??n m??ltiple para describir la relaci??n entre la distorsi??n individual de
    datos de covid y la distorsi??n individual del paso del tiepo, siendo el modelo de regresi??n lineal
    m??ltiple el que mejor ajust?? los datos de las mediciones del proceso, con mejor coeficiente de determinaci??n R2""", styles['Cuerpo']))

    Elements.append(Paragraph("""Los pron??sticos realizados con el modelo de regresi??n lineal m??ltiple, permiten estimar la variacion
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


def reportBarras(request, graphs):

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
        Paragraph("An??lisis de datos", styles['Subtitulo']))
    Elements.append(Paragraph(
        """ El an??lisis de datos consiste en la realizaci??n de las operaciones a las que el investigador someter?? 
        los datos con la finalidad de alcanzar los objetivos del estudio. Todas estas operaciones no pueden definirse 
        de antemano de manera r??gida. La recolecci??n de datos y ciertos an??lisis preliminares pueden revelar problemas y 
        dificultades que desactualizar??n la planificaci??n inicial del an??lisis de los datos. Sin embargo es importante planificar
        los principales aspectos del plan de an??lisis en funci??n de la verificaci??n de cada una de las hip??tesis formuladas ya
        que estas definiciones condicionar??n a su vez la fase de recolecci??n de datos..""", styles['Cuerpo']))

    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Muestra", styles['Subtitulo']))
    Elements.append(Paragraph(
        """Una muestra es un conjunto de medidas u observaciones tomadas a partir de una poblaci??n dada; 
        es un subconjunto de la poblaci??n. Desde luego, el n??mero de observaciones en una muestra es menor
        que el n??mero de posibles observaciones en la poblaci??n, de otra forma, la muestra ser?? la poblaci??n misma.
        Las muestras se toman debido a que no es factible desde el punto de vista econ??mico usar a toda la poblaci??n.""", styles['Cuerpo']))

    Elements.append(
        Paragraph("Estad??stica descriptiva e inferencial", styles['Subtitulo']))
    Elements.append(Paragraph(
        """En base a lo que se ha dicho se concluye, que la Estad??stica como disciplina o ??rea de estudio comprende t??cnicas 
        descriptivas como inferenciales. Incluye la observaci??n y tratamiento de datos num??ricos y el empleo de los datos estad??sticos
        con fines inferenciales""", styles['Cuerpo']))

    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distribucion de los valores segun la categorizacion 
                              eleccionada y se observa su distribucion en general""", styles['Cuerpo']))

    # ---------------------------------
    Elements.append(Paragraph("""Para obtener los presentes resultados se procesa la data ingresada y se realizan los calculos
            necesarios para obtener una data limpia y poder brindar los datos por seleccion y comparar la informacion
            de manera sencilla y adecauda.""", styles['Cuerpo']))

    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # --------------------------------------------------------------------------
    Elements.append(Paragraph("""Por supuesto el resultado depende completamente de la fuente de datos y se limita a 
            generar la informacion solicitada en el archivo de entrada para su mejor comprension""", styles['Cuerpo']))

    # ------------------------------------------------------------------------------------------

    Elements.append(Paragraph("""Los calculos realizados con aplicacion estadistica, permiten ontener el valor
    individual de datos por medio de dos variables y su categorizacion y poder asi trabajar
    y ajustar el proceso de analisis.""", styles['Cuerpo']))
    Elements.append(PageBreak())

    doc.addPageTemplates([
        PageTemplate(id='TwoCol', frames=[
            frame1, frame2]),
    ])

    doc.build(Elements)
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename='reportes.pdf')


def reportPie(request, graphs):

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
        Paragraph("An??lisis de datos", styles['Subtitulo']))
    Elements.append(Paragraph(
        """ El an??lisis de datos consiste en la realizaci??n de las operaciones a las que el investigador someter?? 
        los datos con la finalidad de alcanzar los objetivos del estudio. Todas estas operaciones no pueden definirse 
        de antemano de manera r??gida. La recolecci??n de datos y ciertos an??lisis preliminares pueden revelar problemas y 
        dificultades que desactualizar??n la planificaci??n inicial del an??lisis de los datos. Sin embargo es importante planificar
        los principales aspectos del plan de an??lisis en funci??n de la verificaci??n de cada una de las hip??tesis formuladas ya
        que estas definiciones condicionar??n a su vez la fase de recolecci??n de datos..""", styles['Cuerpo']))

    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Muestra", styles['Subtitulo']))
    Elements.append(Paragraph(
        """Una muestra es un conjunto de medidas u observaciones tomadas a partir de una poblaci??n dada; 
        es un subconjunto de la poblaci??n. Desde luego, el n??mero de observaciones en una muestra es menor
        que el n??mero de posibles observaciones en la poblaci??n, de otra forma, la muestra ser?? la poblaci??n misma.
        Las muestras se toman debido a que no es factible desde el punto de vista econ??mico usar a toda la poblaci??n.""", styles['Cuerpo']))

    Elements.append(
        Paragraph("Grafico Circular", styles['Subtitulo']))
    Elements.append(Paragraph(
        """Un gr??fico circular se divide en ??reas o sectores. Cada sector representa el conteo o porcentaje de
        observaciones de un nivel de la variable. Los gr??ficos circulares se usan a menudo en la empresa. Algunos 
        ejemplos son representar porcentajes de tipos de cliente, de ingresos para diferentes productos, o de beneficios
        en distintos pa??ses. Los gr??ficos circulares pueden ser ??tiles para ilustrar la relaci??n de las partes con el todo 
        cuando hay un n??mero reducido de niveles.""", styles['Cuerpo']))

    Elements.append(
        Paragraph("Datos categ??ricos o nominales", styles['Subtitulo']))
    Elements.append(Paragraph(
        """Los gr??ficos circulares tienen sentido a la hora de representar la relaci??n parte a 
        todo de datos categ??ricos o nominales. Las secciones de un gr??fico circular suelen representar 
        porcentajes del total. En datos categ??ricos, la muestra suele dividirse en grupos, y las respuestas tienen un orden definido.
        Con los datos nominales, la muestra tambi??n se divide en grupos, pero sin un orden particular.
        Un ejemplo de variable nominal ser??a el pa??s de residencia. Puede usar una abreviatura del pa??s,
        o c??digos num??ricos para asociarlos al nombre. De un modo u otro, solo le est?? poniendo nombre a 
        los distintos grupos de datos.""", styles['Cuerpo']))

    Elements.append(
        Paragraph("Datos continuos", styles['Subtitulo']))
    Elements.append(Paragraph(
        """Normalmente los gr??ficos circulares no tienen sentido en datos continuos. 
        Puesto que los datos continuos se miden en una escala con multitud de valores posibles,
        representar la relaci??n parte a todo no tiene sentido. """, styles['Cuerpo']))

    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distribucion de los valores segun la categorizacion 
                              eleccionada y se observa su distribucion en general""", styles['Cuerpo']))

    # ---------------------------------
    Elements.append(Paragraph("""Para obtener los presentes resultados se procesa la data ingresada y se realizan los calculos
            necesarios para obtener una data limpia y poder brindar los datos por seleccion y comparar la informacion
            de manera sencilla y adecauda.""", styles['Cuerpo']))

    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    Elements.append(Paragraph("""Los datos se muestran en porcentaje o con el valor absoluto de cada unos de los datos
         seleccionados para una comparacion directa entra ambos y tener una idea de la magnitud de los datos calculados""", styles['Cuerpo']))

    Elements.append(Image("data:image/png;base64," +
                    graphs[1], width=200, height=150))

    # --------------------------------------------------------------------------
    Elements.append(Paragraph("""Por supuesto el resultado depende completamente de la fuente de datos y se limita a 
            generar la informacion solicitada en el archivo de entrada para su mejor comprension""", styles['Cuerpo']))

    # ------------------------------------------------------------------------------------------

    Elements.append(Paragraph("""Los calculos realizados con aplicacion estadistica, permiten ontener el valor
    individual de datos por medio de dos variables y su categorizacion y poder asi trabajar
    y ajustar el proceso de analisis.""", styles['Cuerpo']))
    Elements.append(PageBreak())

    doc.addPageTemplates([
        PageTemplate(id='TwoCol', frames=[
            frame1, frame2]),
    ])

    doc.build(Elements)
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename='reportes.pdf')


def reportTasas(request, graphs, errorsT, errorsT2, metrics, metrics2, coefs, coefs2):

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
        Paragraph("Modelo de regresi??n lineal simple", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n lineal son ampliamente usados en la ingenier??a ya que sirven para analizar el
    comportamiento de las variables de entrada (o regresora) y salida (o respuesta) estableciendo predicciones y
    estimaciones. En este trabajo la variable regresora corresponde a la distorsi??n arm??nica individual de corriente y
    la variable de respuesta corresponde a la distorsi??n arm??nica individual de tensi??n.
    La ecuaci??n, muestra la representaci??n de un modelo de regresi??n lineal simple, donde Y es la respuesta, X es la
    variable regresora, 0 y 1 son los par??metros del modelo o coeficientes de regresi??n y es el error del modelo""", styles['Cuerpo']))
    Elements.append(
        Paragraph("El coeficiente de determinaci??n R2 se expresa como un porcentaje que indica la variaci??n de los valores de la variable independiente que se puede explicar con la ecuaci??n de regresi??n", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> X <greek>e</greek>""", styles['Equations']))
    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Modelo de regresi??n polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n polinomial se usan cuando la variable de respuesta muestra un comportamiento curvil??neo
    o no lineal. La ecuaci??n, describe el modelo de regresi??n polinomial de orden k en una variable regresora y la
    ecuaci??n, muestra el modelo ajustado de regresi??n polinomial de orden k. Los estimadores de los par??metros del
    modelo se obtienen por el m??todo de los m??nimos cuadrados usando la ecuaci??n, donde y, X, X??? son vectores. En
    este trabajo se aplican los modelos de regresi??n polinomial de orden 2 y orden 3 en una variable regresora""", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> + <greek>b</greek><sub>2</sub> X<super>2</super> + ... + 
        <greek>b</greek><sub>k</sub> X<super>k</super>""", styles['Equations']))
    # Coeficiente de deteriminacion R2
    Elements.append(
        Paragraph("Coeficiente de determinaci??n R<super>2</super>", styles['Subtitulo']))
    Elements.append(Paragraph("""El coeficiente de determinaci??n R2 mide la proporci??n de la variaci??n de la respuesta Y que es explicada por el
    modelo de regresi??n. El coeficiente R2 se calcula usando la ecuaci??n, donde SSR es la medida de variabilidad
    del modelo de regresi??n y SST corresponde a la medida de variabilidad de Y sin considerar el efecto de la variable
    regresora X.""", styles['Cuerpo']))
    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distintas dispersiones y ploteos 
                              de cada caso en especifico, siendo evidente la necesidad de aplicar otro modelo diferente a una regresion lienal""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresi??n al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utiliz?? el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gr??ficas de dispersi??n de
    las variables de respuesta y regresoras y los resultados anal??ticos de los modelos. Las figuras, muestra el
    comportamiento gr??fico de los modelos de regresi??n lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

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
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metrics[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metrics[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metrics[2]), styles['Metrics']))

    Elements.append(
        Paragraph("Metricas del Modelo Polinomial de Muertes", styles['Subtitulo']))
    Elements.append(Paragraph("Error Cuadr??tico Medio (MSE) = " +
                    str(metrics2[0]), styles['Metrics']))
    Elements.append(Paragraph(
        "Ra??z del Error Cuadr??tico Medio (RMSE) = " + str(metrics2[1]), styles['Metrics']))
    Elements.append(Paragraph("Coeficiente de Determinaci??n R2 =  " +
                    str(metrics2[2]), styles['Metrics']))

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresi??n lineal simple y regresi??n
    polinomial de orden 4 y regresi??n m??ltiple para describir la relaci??n entre la distorsi??n individual de
    datos de covid y la distorsi??n individual del paso del tiepo, siendo el modelo de regresi??n lineal
    m??ltiple el que mejor ajust?? los datos de las mediciones del proceso, con mejor coeficiente de determinaci??n R2
    ("""+str(metrics[2])+""").""", styles['Cuerpo']))

    Elements.append(Paragraph("""Los pron??sticos realizados con el modelo de regresi??n lineal m??ltiple, permiten estimar la variacion
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


def reportTresG(request, graphs):

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
        Paragraph("Modelo de regresi??n lineal simple", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n lineal son ampliamente usados en la ingenier??a ya que sirven para analizar el
    comportamiento de las variables de entrada (o regresora) y salida (o respuesta) estableciendo predicciones y
    estimaciones. En este trabajo la variable regresora corresponde a la distorsi??n arm??nica individual de corriente y
    la variable de respuesta corresponde a la distorsi??n arm??nica individual de tensi??n.
    La ecuaci??n, muestra la representaci??n de un modelo de regresi??n lineal simple, donde Y es la respuesta, X es la
    variable regresora, 0 y 1 son los par??metros del modelo o coeficientes de regresi??n y es el error del modelo""", styles['Cuerpo']))
    Elements.append(
        Paragraph("El coeficiente de determinaci??n R2 se expresa como un porcentaje que indica la variaci??n de los valores de la variable independiente que se puede explicar con la ecuaci??n de regresi??n", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> X <greek>e</greek>""", styles['Equations']))
    # Texte descriptivo de Regresion Polinomial
    Elements.append(
        Paragraph("Modelo de regresi??n polinomial", styles['Subtitulo']))
    Elements.append(Paragraph(""" Los modelos de regresi??n polinomial se usan cuando la variable de respuesta muestra un comportamiento curvil??neo
    o no lineal. La ecuaci??n, describe el modelo de regresi??n polinomial de orden k en una variable regresora y la
    ecuaci??n, muestra el modelo ajustado de regresi??n polinomial de orden k. Los estimadores de los par??metros del
    modelo se obtienen por el m??todo de los m??nimos cuadrados usando la ecuaci??n, donde y, X, X??? son vectores. En
    este trabajo se aplican los modelos de regresi??n polinomial de orden 2 y orden 3 en una variable regresora""", styles['Cuerpo']))
    Elements.append(Paragraph(
        """Y = <greek>b</greek><sub>0</sub> + <greek>b</greek><sub>1</sub> + <greek>b</greek><sub>2</sub> X<super>2</super> + ... + 
        <greek>b</greek><sub>k</sub> X<super>k</super>""", styles['Equations']))
    # Coeficiente de deteriminacion R2
    Elements.append(
        Paragraph("Coeficiente de determinaci??n R<super>2</super>", styles['Subtitulo']))
    Elements.append(Paragraph("""El coeficiente de determinaci??n R2 mide la proporci??n de la variaci??n de la respuesta Y que es explicada por el
    modelo de regresi??n. El coeficiente R2 se calcula usando la ecuaci??n, donde SSR es la medida de variabilidad
    del modelo de regresi??n y SST corresponde a la medida de variabilidad de Y sin considerar el efecto de la variable
    regresora X.""", styles['Cuerpo']))
    # Resultados y Conclusiones
    Elements.append(
        Paragraph("Resultados y Conclusiones", styles['Subtitulo']))
    Elements.append(Paragraph("""Al analizar los datos proporcionados por los archivos de entrada se observa las distintas dispersiones y ploteos 
                              de cada caso en especifico, siendo evidente la necesidad de aplicar otro modelo diferente a una regresion lienal""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[0], width=200, height=150))

    # ---------------------------------
    Elements.append(Paragraph("""Para aplicar los modelos de regresi??n al ajuste de los datos de las mediciones de los campos especificos de informacion del covid, se
    se utiliz?? el software Scikit Learn. Utilizando el paquete Sklearn se obtuvieron las gr??ficas de dispersi??n de
    las variables de respuesta y regresoras y los resultados anal??ticos de los modelos. Las figuras, muestra el
    comportamiento gr??fico de los modelos de regresi??n lineal simple y polinomial de orden 4.""", styles['Cuerpo']))

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
    # ------------------------------------------------------------------------------------------
    Elements.append(Paragraph("""Ya obtenido el grado que mejor se ajusta para los datos obtenidos se muestra la grafica que mejor se adopta
        y utiliza este grado optimo, ademas con la cual se pueden obtener las metricas que mas se acercan y reflejan
        resutlados adecuados.""", styles['Cuerpo']))
    Elements.append(Image("data:image/png;base64," +
                    graphs[3], width=200, height=150))

    Elements.append(Paragraph("""En este trabajo, se probaron los modelos de regresi??n lineal simple y regresi??n
    polinomial de orden 4 y regresi??n m??ltiple para describir la relaci??n entre la distorsi??n individual de
    datos de covid y la distorsi??n individual del paso del tiepo, siendo el modelo de regresi??n lineal
    m??ltiple el que mejor ajust?? los datos de las mediciones del proceso, con mejor coeficiente de determinaci??n R2.""", styles['Cuerpo']))

    Elements.append(Paragraph("""Los pron??sticos realizados con el modelo de regresi??n lineal m??ltiple, permiten estimar la variacion
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
