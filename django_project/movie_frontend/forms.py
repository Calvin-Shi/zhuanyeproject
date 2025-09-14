# movie_frontend/forms.py
from django import forms
from django.contrib.auth.models import User
from films_recommender_system.models import UserProfile

class ProfileInfoForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['nickname', 'bio']
        widgets = {
            'bio': forms.Textarea(attrs={'rows': 4, 'placeholder': '写点什么介绍一下自己吧...'}),
            'nickname': forms.TextInput(attrs={'placeholder': '你的昵称'})
        }

class UserEmailForm(forms.ModelForm):
    email = forms.EmailField(required=True, help_text="请输入一个有效的邮箱地址。")

    class Meta:
        model = User
        fields = ['email']

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exclude(pk=self.instance.pk).exists():
            raise forms.ValidationError("该邮箱已被其他用户注册。")
        return email

class AvatarUploadForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['avatar']

class BackgroundUploadForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['profile_background']