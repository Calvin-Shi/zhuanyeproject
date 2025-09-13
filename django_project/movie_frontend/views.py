from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login as auth_login
from django.views.decorators.http import require_POST
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.db import transaction
from django import forms
from django.http import JsonResponse
from django.db.models import Prefetch
from django.db.models import Prefetch, Q  # 导入 Q 对象


from films_recommender_system.models import (
    Movie, MovieTitle, Genre, Review, Recommendation, UserProfile
)


# -------------------- 工具函数 (无变化) --------------------
def _display_title(movie):
    zh = movie.titles.filter(language__icontains='zh').first()
    if zh: return zh.title_text
    en = movie.titles.filter(language__icontains='en').first()
    if en: return en.title_text
    return getattr(movie, 'original_title', str(movie.id))


def _attach_display_titles(movies):
    for m in movies:
        setattr(m, 'display_title', _display_title(m))
    return movies


def _get_fav_ids(request):
    return set(map(int, request.session.get('favorite_movie_ids', [])))


def _set_fav_ids(request, ids):
    request.session['favorite_movie_ids'] = list(map(int, ids))


# -------------------- 视图 --------------------
def signup(request):
    # ... (无变化)
    if request.user.is_authenticated:
        return redirect('movie_frontend:home')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('movie_frontend:home')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})


def home(request):
    # ... (无变化)
    movies = Movie.objects.all().prefetch_related('titles', 'genres')[:12]
    movies = _attach_display_titles(list(movies))
    context = {
        'movies': movies,
        'fav_ids': _get_fav_ids(request),
    }
    return render(request, 'index.html', context)


def movie_list(request):
    qs = Movie.objects.all().prefetch_related('titles', 'genres')

    # MODIFIED: 添加搜索查询处理
    search_query = request.GET.get('q')
    if search_query:
        # 使用 Q 对象进行 OR 查询，同时搜索中文译名和原始标题
        qs = qs.filter(
            Q(titles__title_text__icontains=search_query) |
            Q(original_title__icontains=search_query)
        )

    selected_genre_id = request.GET.get('genre')
    if selected_genre_id:
        try:
            gid = int(selected_genre_id)
            qs = qs.filter(genres__id=gid)
        except (ValueError, TypeError):
            pass

    movies = _attach_display_titles(list(qs.distinct()))
    genres = list(Genre.objects.all())
    context = {
        'movies': movies,
        'genres': genres,
        'selected_genre_id': int(selected_genre_id) if selected_genre_id and selected_genre_id.isdigit() else None,
        'fav_ids': _get_fav_ids(request),
        'query': search_query,  # MODIFIED: 将查询词传回模板
    }
    return render(request, 'movie_list.html', context)


def movie_detail(request, movie_id):
    # ... (无变化)
    movie = get_object_or_404(
        Movie.objects.prefetch_related('titles', 'genres', 'directors', 'actors', 'scriptwriters'), pk=movie_id)
    display_title = _display_title(movie)
    setattr(movie, 'display_title', display_title)
    reviews = Review.objects.filter(movie=movie).select_related('source').order_by('-id')[:20]
    context = {
        'movie': movie,
        'display_title': display_title,
        'reviews': reviews,
        'fav_ids': _get_fav_ids(request),
    }
    return render(request, 'movie_detail.html', context)


# ... recommendations, choose_favorites, toggle_favorite 视图无变化 ...
@login_required
def recommendations(request):
    fav_ids = _get_fav_ids(request)
    movies = _attach_display_titles(list(Movie.objects.filter(id__in=fav_ids).prefetch_related('titles', 'genres')))
    rec, _ = Recommendation.objects.get_or_create(user=request.user)
    with transaction.atomic():
        rec.recommended_movies.set(movies)
    return render(request, 'recommendations.html', {'movies': movies})


def choose_favorites(request):
    if request.method == 'POST':
        ids = set(map(int, request.POST.getlist('favorites')))
        _set_fav_ids(request, ids)
        if request.user.is_authenticated:
            return redirect('movie_frontend:recommendations')
        else:
            return redirect('movie_frontend:home')
    movies = _attach_display_titles(list(Movie.objects.all().prefetch_related('titles', 'genres')))
    return render(request, 'choose_favorites.html', {'movies': movies, 'fav_ids': _get_fav_ids(request)})


@require_POST
def toggle_favorite(request, movie_id):
    favs = _get_fav_ids(request)
    if movie_id in favs:
        favs.remove(movie_id)
    else:
        favs.add(movie_id)
    _set_fav_ids(request, favs)
    next_url = request.META.get('HTTP_REFERER') or reverse('movie_frontend:home')
    return HttpResponseRedirect(next_url)

class ProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['nickname', 'signature']
        widgets = {
            'signature': forms.Textarea(attrs={'rows': 3, 'cols': 40}),  # 签名用多行文本框
        }



@login_required
def profile(request):
    """个人资料视图：展示并允许修改昵称和签名，以及用户喜欢的电影"""
    # 获取或创建用户资料
    profile, created = UserProfile.objects.get_or_create(user=request.user)

    # 处理表单提交
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            return JsonResponse({
                'status': 'success',
                'nickname': profile.nickname,
                'signature': profile.signature
            })
        return JsonResponse({'status': 'error', 'errors': form.errors})

    # "喜欢的电影"处理逻辑
    def _get_fav_ids(request):
        """获取用户勾选的喜欢的电影ID"""
        return request.session.get('favorite_movie_ids', [])

    def _attach_display_titles(movies):
        """为电影添加显示标题"""
        for movie in movies:
            primary_title = movie.titles.filter(is_primary=True).first()
            movie.display_title = primary_title.title_text if primary_title else movie.original_title
        return movies

    fav_ids = _get_fav_ids(request)
    favorite_movies = _attach_display_titles(
        list(Movie.objects.filter(
            id__in=fav_ids
        ).prefetch_related(
            'titles',
            'genres'
        ))
    )

    with transaction.atomic():
        user_recommendation, _ = Recommendation.objects.get_or_create(user=request.user)
        user_recommendation.recommended_movies.set(favorite_movies)

    # 传递数据到模板
    context = {
        'favorite_movies': favorite_movies,
        'profile': profile
    }

    return render(request, 'profile.html', context)