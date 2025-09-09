
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login as auth_login
from django.views.decorators.http import require_POST
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.db import transaction
from django.db.models import Prefetch

from films_recommender_system.models import (
    Movie, MovieTitle, Genre, Review, Recommendation
)

# -------------------- 工具函数 --------------------
def _display_title(movie):
    # 优先中文标题，其次英文标题，最后 original_title
    zh = movie.titles.filter(language__icontains='zh').first()
    if zh:
        return zh.title_text
    en = movie.titles.filter(language__icontains='en').first()
    if en:
        return en.title_text
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
    movies = Movie.objects.all().prefetch_related('titles','genres')[:12]
    movies = _attach_display_titles(list(movies))
    context = {
        'movies': movies,
        'fav_ids': _get_fav_ids(request),
    }
    return render(request, 'index.html', context)

def movie_list(request):
    qs = Movie.objects.all().prefetch_related('titles','genres')
    selected_genre_id = request.GET.get('genre')
    if selected_genre_id:
        try:
            gid = int(selected_genre_id)
            qs = qs.filter(genres__id=gid)
        except ValueError:
            pass
    movies = _attach_display_titles(list(qs.distinct()))
    genres = list(Genre.objects.all())
    context = {
        'movies': movies,
        'genres': genres,
        'selected_genre_id': int(selected_genre_id) if selected_genre_id and selected_genre_id.isdigit() else None,
        'fav_ids': _get_fav_ids(request),
        'query': request.GET.get('q',''),
    }
    return render(request, 'movie_list.html', context)

def movie_detail(request, movie_id):
    movie = get_object_or_404(Movie.objects.prefetch_related('titles','genres','directors','actors','scriptwriters'), pk=movie_id)
    display_title = _display_title(movie)
    setattr(movie, 'display_title', display_title)
    reviews = Review.objects.filter(movie=movie).order_by('-id')[:20]
    context = {
        'movie': movie,
        'display_title': display_title,
        'reviews': reviews,
        'fav_ids': _get_fav_ids(request),
    }
    return render(request, 'movie_detail.html', context)

@login_required
def recommendations(request):
    # 简化：展示用户在“选择你喜欢的电影”中勾选的电影
    fav_ids = _get_fav_ids(request)
    movies = _attach_display_titles(list(Movie.objects.filter(id__in=fav_ids).prefetch_related('titles','genres')))
    # 记录/更新到 Recommendation，便于后续扩展
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
            # 未登录也允许暂存喜好，返回首页
            return redirect('movie_frontend:home')
    movies = _attach_display_titles(list(Movie.objects.all().prefetch_related('titles','genres')))
    return render(request, 'choose_favorites.html', {'movies': movies, 'fav_ids': _get_fav_ids(request)})

@require_POST
def toggle_favorite(request, movie_id):
    favs = _get_fav_ids(request)
    if movie_id in favs:
        favs.remove(movie_id)
    else:
        favs.add(movie_id)
    _set_fav_ids(request, favs)
    # 回到来源页
    next_url = request.META.get('HTTP_REFERER') or reverse('movie_frontend:home')
    return HttpResponseRedirect(next_url)
