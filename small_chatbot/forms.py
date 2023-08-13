from django import forms

class ChatForm(forms.Form):
    topic = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Type your topic here'}), required=True)  
    num_source=forms.IntegerField(required=False, min_value=1, max_value=15, label="num Sources", initial=3)

    
