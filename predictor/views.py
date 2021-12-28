from django.shortcuts import render, HttpResponse
from .forms import ModelForm, ParamSelector
import pickle
import io
import base64
import urllib
import csv
import json

enc = []
jsonArray = []
csvActual = None
parameters = []


def home(request):

    return render(request, 'base.html')


def upload(request):
    if request.method == 'POST':

        csvReader = request.FILES['formFile']
        global csvActual
        csvActual = csvReader
        decoded_file = csvReader.read().decode('utf-8').splitlines()
        reader = csv.DictReader(decoded_file)
        global jsonArray
        jsonArray = []
        for rows in reader:
            for key, value in dict(rows).items():
                if key is None or value is None:
                    del rows[key]

            jsonArray.append(rows)

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
            print("AYUDAAA"+temp)

   # if a GET (or any other method) we'll create a blank form

    else:
        form = ModelForm()
    return render(request, 'hello.html', {'form': form, 'enc': enc, 'parameters': parameters})


def predict_model(request):
    # if this is a POST request we need to process the form data

    if request.method == 'POST':
        # create a form instance and populate it with data from the request:

        print("Buenas")

    else:
        form = ModelForm()
    return render(request, 'tendencia_infeccion_pais.html', {'form': form, 'enc': enc, 'parameters': parameters})
