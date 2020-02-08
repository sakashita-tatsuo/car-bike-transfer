from django import forms

class PhotoForm(forms.Form):
    #class名をこのimageにつける説明をattrsで記述
    image = forms.ImageField(widget=forms.FileInput(attrs={'class':'custom-file-input'}))