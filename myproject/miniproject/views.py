from django.shortcuts import render, redirect
from .form import DataInputForm
import requests
from django.contrib import messages
# Create your views here.
def input_data(request):
    if request.method == 'POST':
       form = DataInputForm(request.POST)
       if form.is_valid():
          data = form.cleaned_data
          response = requests.post('http://localhost:5000/process_data', json=data)
          result = response.json()
        #   return render(request, 'result.html', {'result': result})
          if(result[0] == 1):
              messages.error(request, f'Diagnosis: You are at risk for diabetes')
          else:
              messages.success(request, f'Diagnosis: You are not at risk for diabetes')
          return redirect('input_data')
       else:
            print(form.errors)
            return render(request, 'input_data.html', {'form': form})
    else:
        form = DataInputForm()
    return render(request, 'input_data.html', {'form': form})