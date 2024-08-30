from django import forms
class DataInputForm(forms.Form):
    Pregnancies = forms.IntegerField(min_value=0, max_value=20, required=True, label='Pregnancies')
    Glucose = forms.FloatField(min_value=0, max_value=200, required=True, label='Glucose')
    BloodPressure = forms.FloatField(min_value=0, max_value=200, required=True, label='Blood Pressure')
    SkinThickness = forms.FloatField(min_value=0, max_value=100, required=True, label='Skin Thickness')
    Insulin = forms.FloatField(min_value=0, max_value=900, required=True, label='Insulin')
    BMI = forms.FloatField(min_value=0, max_value=100, required=True, label='BMI')
    DiabetesPedigreeFunction = forms.FloatField(min_value=0, max_value=2.5, required=True, label='Diabetes Pedigree Function')
    Age = forms.IntegerField(min_value=0, max_value=120, required=True, label='Age')
   
    def clean(self):
        cleaned_data = super().clean()

        # Kiểm tra tất cả các trường không được trống và không nhỏ hơn 0
        for field, value in cleaned_data.items():
            if value is None:
                self.add_error(field, f"{self.fields[field].label} không được để trống.")
            elif value < 0:
                self.add_error(field, f"{self.fields[field].label} không thể nhỏ hơn 0.")

        # glucose = cleaned_data.get('Glucose')
        # if glucose is not None and glucose > 200:
        #     self.add_error('Glucose', 'Glucose không được lớn hơn 200.')

        # Trả về dữ liệu đã làm sạch
        return cleaned_data