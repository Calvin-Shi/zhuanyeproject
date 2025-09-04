from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from films_recommender_system.models import Movie, Recommendation, Genre
def home(request):
    movies = Movie.objects.all().prefetch_related('titles').order_by('-release_year')[:8]
    return render(request, 'index.html', {'movies': movies})

def movie_list(request):
    selected = request.GET.get('genre')  # 字符串或 None
    movies = Movie.objects.all().prefetch_related('titles', 'genres').order_by('-release_year')
    if selected:
        movies = movies.filter(genres__id=selected)

    # 传给模板用来渲染下拉框与选中态
    try:
        selected_genre_id = int(selected) if selected else None
    except (TypeError, ValueError):
        selected_genre_id = None

    genres = Genre.objects.all().order_by('name')

    return render(request, 'movie_list.html', {
        'movies': movies,
        'genres': genres,
        'selected_genre_id': selected_genre_id,
    })

def movie_detail(request, movie_id):
    movie = get_object_or_404(
        Movie.objects.prefetch_related('titles', 'genres', 'directors', 'actors', 'scriptwriters', 'reviews'),
        pk=movie_id
    )
    return render(request, 'movie_detail.html', {'movie': movie})

@login_required
def recommendations(request):
    try:
        rec = Recommendation.objects.prefetch_related('recommended_movies__titles').get(user=request.user)
        movies = rec.recommended_movies.all().prefetch_related('titles')
        context = {'movies': movies}
    except Recommendation.DoesNotExist:
        context = {'error': '暂无推荐数据，请先评分一些电影'}
    return render(request, 'recommendations.html', context)
