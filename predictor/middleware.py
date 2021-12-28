

def handle_uploaded_file(f):
    with open('models/actual.csv', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
