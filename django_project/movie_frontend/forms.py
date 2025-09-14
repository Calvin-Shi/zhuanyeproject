from django import forms
from django.contrib.auth.models import User
from films_recommender_system.models import UserProfile, UserReview


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

    class Meta: model = User; fields = ['email']

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exclude(pk=self.instance.pk).exists():
            raise forms.ValidationError("该邮箱已被其他用户注册。")
        return email


class AvatarUploadForm(forms.ModelForm):
    class Meta: model = UserProfile; fields = ['avatar']


class BackgroundUploadForm(forms.ModelForm):
    class Meta: model = UserProfile; fields = ['profile_background']


# --- 新增用户评论表单 ---
class UserReviewForm(forms.ModelForm):
    # 使用 ChoiceField 创建一个下拉选择框用于评分
    rating = forms.ChoiceField(
        choices=[(i, f"{i}.0") for i in range(10, 0, -1)],
        label="评分",
        help_text="请选择你的评分 (10为最高)"
    )
    review = forms.CharField(
        label="评论内容",
        widget=forms.Textarea(attrs={'rows': 5, 'placeholder': '分享你的看法...'}),
    )

    class Meta:
        model = UserReview
        fields = ['rating', 'review']

    def __init__(self, *args, **kwargs):
        # 我们需要 user 和 movie 对象来进行验证，所以从外部传入
        self.user = kwargs.pop('user', None)
        self.movie = kwargs.pop('movie', None)
        super().__init__(*args, **kwargs)

    def clean_review(self):
        # "反刷屏"验证逻辑
        review_text = self.cleaned_data.get('review')
        if self.user and self.movie:
            # 检查当前用户是否已对该电影发布过完全相同内容的评论
            # 如果是编辑表单 (self.instance.pk 存在), 则要排除当前正在编辑的这条评论
            query = UserReview.objects.filter(user=self.user, movie=self.movie, review=review_text)
            if self.instance and self.instance.pk:
                query = query.exclude(pk=self.instance.pk)

            if query.exists():
                raise forms.ValidationError("你已经发布过一条完全相同的评论了。")
        return review_text