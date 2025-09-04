from django.http import HttpResponse
from django.shortcuts import render
from .models import cal

# Create your views here.
def index(request):
    return render(request,'index.html')

def calPage(request):
    return render(request,'cal.html')

def calculate(request):
    if request.method == "POST":
        value1 = request.POST['value1']
        value2 = request.POST['value2']
        result = int(value2) + int(value1)
        print(int(value1),' + ',int(value2),' = ',result)

        cal.objects.create(value1 = value1,value2 = value2,result = result)

        return render(request,'results.html',context={'data':result})
    else:
        return HttpResponse('Please use POST method')

def calList(request):
    data = cal.objects.all()
    return render(request,'list.html',context={'data':data})

def delData(request):
    cal.objects.all().delete()
    return HttpResponse('data deleted')
