from django.shortcuts import render, HttpResponse
from numpy.lib.function_base import gradient
from .forms import ModelForm, ParamSelector
from django.http import FileResponse
import pickle
import io
import base64
import urllib
import csv
import json
import pandas
from .ml_model.tendencia_infeccion_pais import getGraph
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
enc = []
jsonArray = []
csvActual = None
parameters = []

tempPais = ""
auxG = None


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


def predict_model(request):
    # if this is a POST request we need to process the form data

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:

        form = ModelForm(request.POST)

        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            sepal_length = form.cleaned_data['sepal_length']
            sepal_width = form.cleaned_data['sepal_width']
            petal_length = form.cleaned_data['petal_length']
            petal_width = form.cleaned_data['petal_width']

            # Run new features through ML model
            model_features = [
                [sepal_length, sepal_width, petal_length, petal_width]]

            loaded_model = pickle.load(
                open("ml_model/iris_model.pkl", 'rb'))

            prediction = loaded_model[0].predict(model_features)[0]

            buf = io.BytesIO()
            loaded_model[1].savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = urllib.parse.quote(string)

            prediction_dict = [{'name': 'setosa',
                                'img': 'https://alchetron.com/cdn/iris-setosa-0ab3145a-68f2-41ca-a529-c02fa2f5b02-resize-750.jpeg'},
                               {'name': 'versicolor',
                                'img': 'https://wiki.irises.org/pub/Spec/SpecVersicolor/iversicolor07.jpg'},
                               {'name': 'virginica',
                                'img': 'https://www.gardenia.net/storage/app/public/uploads/images/detail/xUM027N8JI22aQPImPoH3NtIMpXkm89KAIKuvTMB.jpeg'}]

            prediction_name = prediction_dict[prediction]['name']
            prediction_img = prediction_dict[prediction]['img']

            return render(request, 'hello.html', {'form': form, 'prediction': prediction,
                                                  'prediction_name': prediction_name,
                                                  'prediction_img': prediction_img,
                                                  'graph': uri, 'enc': enc})
        else:
            temp = request.POST.get("selectorP")
            parameters.append(temp)

   # if a GET (or any other method) we'll create a blank form

    else:
        form = ModelForm()
    return render(request, 'hello.html', {'form': form, 'enc': enc, 'parameters': parameters})


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
            print(tempPais)
            graphs, errors, metrics = getGraph(param2, tempPais, param3,
                                               param1, jsonArray)
            global auxG
            auxG = graphs[0]

        return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises, 'graphs': graphs, 'errors': errors, 'metrics': metrics})

    elif request.GET.get('Down') == 'Down':
        return some_view(request, auxG)
        # return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})
    else:

        return render(request, 'tendencia_infeccion_pais.html', {'enc': enc, 'parameters': parameters, 'paises': paises})


def some_view(request, gra):
    # Create a file-like buffer to receive PDF data.
    buffer = io.BytesIO()

    # Create the PDF object, using the buffer as its "file."
    p = canvas.Canvas(buffer)

    # Draw things on the PDF. Here's where the PDF generation happens.
    # See the ReportLab documentation for the full list of functionality.
    p.drawString(100, 100, "Hello world.")

    p.drawImage("data:image/png;base64,"+gra, 10, 500, mask='auto')
    # Close the PDF object cleanly, and we're done.
    p.showPage()
    p.save()

    # FileResponse sets the Content-Disposition header so that browsers
    # present the option to save the file.
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename='hello.pdf')
