from django.http import JsonResponse
import json
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.contrib.auth import login as auth_login, update_session_auth_hash
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.db import transaction
from django.db.models import Prefetch, Q
from django.db.models import F

from films_recommender_system.models import (
    Movie, MovieTitle, Genre, Review, Recommendation, UserProfile,
    UserReview, BrowsingHistory,Person
)
from .forms import (
    ProfileInfoForm, UserEmailForm, AvatarUploadForm, BackgroundUploadForm,UserReviewForm
)

from django.http import HttpResponseForbidden
from django.core.paginator import Paginator


# 其他的工具函数
def _display_title(movie):
    primary_title = movie.titles.filter(is_primary=True, language__icontains='zh').first()
    if not primary_title: primary_title = movie.titles.filter(is_primary=True).first()
    if primary_title: return primary_title.title_text
    zh_title = movie.titles.filter(language__icontains='zh').first()
    if zh_title: return zh_title.title_text
    return movie.original_title


def _attach_display_titles(movies):
    movie_ids = [m.id for m in movies]
    titles = MovieTitle.objects.filter(movie_id__in=movie_ids)
    titles_map = {m_id: [] for m_id in movie_ids}
    for title in titles:
        titles_map[title.movie_id].append(title)
    for movie in movies:
        movie_titles = titles_map.get(movie.id, [])
        display_title = next((t.title_text for t in movie_titles if t.is_primary and 'zh' in t.language.lower()), None)
        if not display_title: display_title = next((t.title_text for t in movie_titles if t.is_primary), None)
        if not display_title: display_title = next((t.title_text for t in movie_titles if 'zh' in t.language.lower()),
                                                   None)
        if not display_title: display_title = movie.original_title
        setattr(movie, 'display_title', display_title)
    return movies


def _get_fav_ids(request):
    return set(map(int, request.session.get('favorite_movie_ids', [])))


def _set_fav_ids(request, ids):
    request.session['favorite_movie_ids'] = list(map(int, ids))


# ... (signup, home, movie_list 视图无变化) ...
def signup(request):
    if request.user.is_authenticated: return redirect('movie_frontend:home')
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
    # --- 热门电影展示逻辑 ---

    # 1. 在这里指定希望固定展示的电影ID
    # 可以通过后台管理或直接查询数据库来获取这些ID
    featured_movie_ids = [263, 953, 100, 2, 90, 932, 481, 1288, 1010, 126, 1765,16]

    # 2. 设置首页总共要展示的电影数量
    total_movies_on_home = 12

    # 3. 首先获取所有“精选”的电影对象
    featured_movies = list(Movie.objects.filter(id__in=featured_movie_ids).prefetch_related('titles', 'genres'))

    # 4. 计算还需要多少部随机电影
    num_random_movies_needed = total_movies_on_home - len(featured_movies)

    random_movies = []
    if num_random_movies_needed > 0:
        # 5. 从数据库中随机获取所需数量的电影
        #    - 使用 exclude() 来确保不会重复选中精选电影
        #    - 使用 order_by('?') 来实现随机排序
        random_movies = list(
            Movie.objects.exclude(id__in=featured_movie_ids)
            .order_by('?')
            .prefetch_related('titles', 'genres')
            [:num_random_movies_needed]
        )

    # 6. 合并列表（精选电影在前，随机电影在后）
    final_movies_list = featured_movies + random_movies

    # --- 逻辑结束，后续处理不变 ---

    movies = _attach_display_titles(final_movies_list)
    context = {
        'movies': movies,
        'fav_ids': _get_fav_ids(request),
    }
    return render(request, 'index.html', context)


# --- 新增：处理AJAX请求的视图 ---
def refresh_popular_movies(request):
    # 这里的逻辑与 home 视图中的电影获取逻辑完全相同
    featured_movie_ids = []
    total_movies_on_home = 12
    featured_movies = list(Movie.objects.filter(id__in=featured_movie_ids).prefetch_related('titles', 'genres'))
    num_random_movies_needed = total_movies_on_home - len(featured_movies)
    random_movies = []
    if num_random_movies_needed > 0:
        random_movies = list(
            Movie.objects.exclude(id__in=featured_movie_ids)
            .order_by('?')
            .prefetch_related('titles', 'genres')
            [:num_random_movies_needed]
        )
    final_movies_list = featured_movies + random_movies
    movies = _attach_display_titles(final_movies_list)

    # 关键区别：只渲染模板片段，而不是整个页面
    return render(request, 'partials/_movie_grid.html', {'movies': movies})


def movie_list(request):
    qs = Movie.objects.all().prefetch_related('titles', 'genres').order_by('-release_year')

    search_query = request.GET.get('q')
    if search_query:
        qs = qs.filter(
            Q(titles__title_text__icontains=search_query) | Q(original_title__icontains=search_query)).distinct()

    selected_genre_id = request.GET.get('genre')
    if selected_genre_id and selected_genre_id.isdigit():
        gid = int(selected_genre_id)
        qs = qs.filter(genres__id=gid)

    # --- 分页逻辑 ---
    paginator = Paginator(qs, 24)  # 每页显示24部电影
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    movies_with_titles = _attach_display_titles(list(page_obj.object_list))
    page_obj.object_list = movies_with_titles  # 用处理过的列表替换

    genres = list(Genre.objects.all())
    context = {
        'page_obj': page_obj,  # 传递 page_obj 而不是 movies
        'genres': genres,
        'selected_genre_id': int(selected_genre_id) if selected_genre_id and selected_genre_id.isdigit() else None,
        'query': search_query,
    }
    return render(request, 'movie_list.html', context)


def movie_detail(request, movie_id):
    movie = get_object_or_404(
        Movie.objects.prefetch_related('titles', 'genres', 'directors', 'actors', 'scriptwriters'), pk=movie_id)

    if request.method == 'POST' and 'review' in request.POST and request.user.is_authenticated:
        review_form = UserReviewForm(request.POST, user=request.user, movie=movie)
        if review_form.is_valid():
            new_review = review_form.save(commit=False)
            new_review.user = request.user
            new_review.movie = movie
            new_review.save()
            messages.success(request, "你的评论已成功发布！")
            return redirect('movie_frontend:movie_detail', movie_id=movie.id)
    else:
        review_form = UserReviewForm()

    setattr(movie, 'display_title', _display_title(movie))

    external_reviews = Review.objects.filter(movie=movie).select_related('source').order_by('-content_date')[:20]
    user_reviews = UserReview.objects.filter(movie=movie).select_related('user').order_by('-timestamp')

    is_in_watchlist = False
    is_in_favorites = False
    liked_review_ids = set()  # 新增: 初始化一个空集合

    if request.user.is_authenticated:
        BrowsingHistory.objects.update_or_create(user=request.user, movie=movie)
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        is_in_watchlist = profile.watchlist.filter(pk=movie.id).exists()
        is_in_favorites = movie.id in _get_fav_ids(request)

        # 新增: 获取当前用户已点赞的评论ID
        liked_review_ids = set(
            UserReview.objects.filter(
                id__in=user_reviews.values_list('id', flat=True),
                liked_by=request.user
            ).values_list('id', flat=True)
        )

    context = {
        'movie': movie,
        'external_reviews': external_reviews,
        'user_reviews': user_reviews,
        'review_form': review_form,
        'fav_ids': _get_fav_ids(request),
        'is_in_watchlist': is_in_watchlist,
        'is_in_favorites': is_in_favorites,
        'liked_review_ids': liked_review_ids,  # 传递到模板
    }
    return render(request, 'movie_detail.html', context)


# --- 新增点赞视图 ---
@require_POST
@login_required
def like_review(request, review_id):
    review = get_object_or_404(UserReview, pk=review_id)
    user = request.user

    # 检查用户是否已经点赞
    if user in review.liked_by.all():
        # 如果已点赞，则取消
        review.liked_by.remove(user)
        # 使用F()表达式确保原子性操作，防止竞争条件
        review.likes_count = F('likes_count') - 1
    else:
        # 如果未点赞，则添加
        review.liked_by.add(user)
        review.likes_count = F('likes_count') + 1

    review.save()

    # 重定向回电影详情页
    return redirect('movie_frontend:movie_detail', movie_id=review.movie.id)


# --- 新增视图 ---

@login_required
def edit_review(request, review_id):
    review = get_object_or_404(UserReview, pk=review_id)

    # 权限检查：确保是评论作者本人
    if request.user != review.user:
        return HttpResponseForbidden("你没有权限修改他人的评论。")

    if request.method == 'POST':
        form = UserReviewForm(request.POST, instance=review, user=request.user, movie=review.movie)
        if form.is_valid():
            form.save()
            messages.success(request, "评论修改成功！")
            return redirect('movie_frontend:movie_detail', movie_id=review.movie.id)
    else:
        form = UserReviewForm(instance=review)

    # 创建一个单独的模板来处理编辑
    return render(request, 'edit_review.html', {'form': form, 'review': review})


@require_POST
@login_required
def delete_review(request, review_id):
    review = get_object_or_404(UserReview, pk=review_id)
    movie_id = review.movie.id

    # 权限检查
    if request.user != review.user:
        return HttpResponseForbidden("你没有权限删除他人的评论。")

    review.delete()
    messages.success(request, "评论已删除。")
    return redirect('movie_frontend:movie_detail', movie_id=movie_id)


# MODIFIED: 重写 toggle_favorite 视图
@require_POST
@login_required
def toggle_favorite(request, movie_id):
    movie = get_object_or_404(Movie, pk=movie_id)

    # 1. 更新 Session (用于UI即时反馈)
    fav_ids = _get_fav_ids(request)
    is_favorited = movie_id in fav_ids

    if is_favorited:
        fav_ids.remove(movie_id)
    else:
        fav_ids.add(movie_id)
    _set_fav_ids(request, fav_ids)

    # 2. 更新数据库 (用于数据持久化)
    recommendation, created = Recommendation.objects.get_or_create(user=request.user)
    if is_favorited:
        # 如果之前是喜欢状态，现在移除
        recommendation.favorite_movies.remove(movie)
    else:
        # 如果之前不是喜欢状态，现在添加
        recommendation.favorite_movies.add(movie)

    next_url = request.META.get('HTTP_REFERER') or reverse('movie_frontend:home')
    return HttpResponseRedirect(next_url)


# ... (recommendations, choose_favorites, toggle_watchlist 视图无变化) ...
@login_required
def recommendations(request):
    rec, _ = Recommendation.objects.get_or_create(user=request.user)
    favorite_movies_from_db = list(rec.favorite_movies.all().values_list('id', flat=True))
    _set_fav_ids(request, favorite_movies_from_db)
    movies = _attach_display_titles(list(rec.favorite_movies.all().prefetch_related('titles', 'genres')))
    return render(request, 'recommendations.html', {'movies': movies})


def choose_favorites(request):
    if request.method == 'POST':
        # MODIFIED: 读取由JS动态生成的完整喜好列表
        favorite_ids_str = request.POST.get('favorite_ids_json', '[]')
        try:
            ids = set(map(int, json.loads(favorite_ids_str)))
        except (json.JSONDecodeError, ValueError):
            ids = set()

        # Session 和 数据库同步更新为最终状态
        _set_fav_ids(request, list(ids))
        if request.user.is_authenticated:
            rec, _ = Recommendation.objects.get_or_create(user=request.user)
            rec.favorite_movies.set(Movie.objects.filter(id__in=ids))
        return redirect('movie_frontend:recommendations')

    # --- GET 请求处理 (与上一版完全相同) ---
    query = request.GET.get('q', '').strip()
    main_tab = request.GET.get('main_tab', 'search')
    sub_tab = request.GET.get('sub_tab', 'directors')

    all_genres = Genre.objects.all().order_by('name')
    directors_qs = Person.objects.filter(directed_movies__isnull=False).distinct().order_by('name')
    actors_qs = Person.objects.filter(acted_in_movies__isnull=False).distinct().order_by('name')

    movies_qs = Movie.objects.all().prefetch_related('titles').order_by('-release_year')
    if query:
        movies_qs = movies_qs.filter(
            Q(titles__title_text__icontains=query) | Q(original_title__icontains=query) |
            Q(directors__name__icontains=query) | Q(actors__name__icontains=query)
        ).distinct()

    movies_paginator = Paginator(movies_qs, 18)
    directors_paginator = Paginator(directors_qs, 30)
    actors_paginator = Paginator(actors_qs, 30)
    page_number = request.GET.get('page', 1)
    page_obj_movies = movies_paginator.get_page(page_number)
    page_obj_directors = directors_paginator.get_page(page_number)
    page_obj_actors = actors_paginator.get_page(page_number)
    movies_with_titles = _attach_display_titles(list(page_obj_movies.object_list))
    page_obj_movies.object_list = movies_with_titles

    favorite_people_ids, favorite_genre_ids, initial_favorite_movie_ids = set(), set(), []
    if request.user.is_authenticated:
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        favorite_people_ids = set(profile.favorite_people.values_list('id', flat=True))
        favorite_genre_ids = set(profile.favorite_genres.values_list('id', flat=True))
        rec, _ = Recommendation.objects.get_or_create(user=request.user)
        initial_favorite_movie_ids = list(rec.favorite_movies.values_list('id', flat=True))
        _set_fav_ids(request, initial_favorite_movie_ids)

    context = {
        'page_obj_movies': page_obj_movies,
        'page_obj_directors': page_obj_directors,
        'page_obj_actors': page_obj_actors,
        'all_genres': all_genres,
        'main_tab': main_tab,
        'sub_tab': sub_tab,
        'query': query,
        'favorite_people_ids': favorite_people_ids,
        'favorite_genre_ids': favorite_genre_ids,
        'initial_favorite_movie_ids_json': json.dumps(initial_favorite_movie_ids),
    }
    return render(request, 'choose_favorites.html', context)


# --- 新增：处理AJAX请求的视图 ---
@require_POST
@login_required
def toggle_favorite_entity(request):
    try:
        data = json.loads(request.body)
        entity_type = data.get('entity_type')
        entity_id = data.get('entity_id')

        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        action = ''

        if entity_type == 'person':
            person = get_object_or_404(Person, pk=entity_id)
            if profile.favorite_people.filter(pk=person.id).exists():
                profile.favorite_people.remove(person)
                action = 'removed'
            else:
                profile.favorite_people.add(person)
                action = 'added'

        elif entity_type == 'genre':
            genre = get_object_or_404(Genre, pk=entity_id)
            if profile.favorite_genres.filter(pk=genre.id).exists():
                profile.favorite_genres.remove(genre)
                action = 'removed'
            else:
                profile.favorite_genres.add(genre)
                action = 'added'

        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid entity type'}, status=400)

        return JsonResponse({'status': 'ok', 'action': action})

    except (json.JSONDecodeError, KeyError):
        return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

@require_POST
@login_required
def toggle_watchlist(request, movie_id):
    movie = get_object_or_404(Movie, pk=movie_id)
    profile = request.user.profile
    if profile.watchlist.filter(pk=movie.id).exists():
        profile.watchlist.remove(movie)
    else:
        profile.watchlist.add(movie)
    next_url = request.META.get('HTTP_REFERER') or reverse('movie_frontend:home')
    return HttpResponseRedirect(next_url)


# MODIFIED: 重构 profile 视图的 POST 处理逻辑
@login_required
def profile(request):
    user_profile, created = UserProfile.objects.get_or_create(user=request.user)
    user = request.user

    if request.method == 'POST':
        form_type = request.POST.get('form_type')
        active_tab = request.POST.get('active_tab', 'tab-profile')  # 获取当前激活的tab
        redirect_url = reverse('movie_frontend:profile') + f'#{active_tab}'  # 构建带片段的URL

        if form_type == 'profile_info':
            form = ProfileInfoForm(request.POST, instance=user_profile)
            if form.is_valid():
                form.save()
                messages.success(request, '个人资料更新成功！')
        elif form_type == 'user_email':
            form = UserEmailForm(request.POST, instance=user)
            if form.is_valid():
                form.save()
                messages.success(request, '邮箱更新成功！')
        elif form_type == 'avatar_upload':
            form = AvatarUploadForm(request.POST, request.FILES, instance=user_profile)
            if form.is_valid():
                form.save()
                messages.success(request, '头像上传成功！')
        elif form_type == 'background_upload':
            form = BackgroundUploadForm(request.POST, request.FILES, instance=user_profile)
            if form.is_valid():
                form.save()
                messages.success(request, '个人背景更新成功！')
        elif form_type == 'password_change':
            form = PasswordChangeForm(user, request.POST)
            if form.is_valid():
                user = form.save()
                update_session_auth_hash(request, user)
                messages.success(request, '密码修改成功！')
            else:
                for field, errors in form.errors.items():
                    for error in errors: messages.error(request, f"{field}: {error}")

        return redirect(redirect_url)  # 重定向到带片段的URL

    # GET 请求处理 (无变化)
    info_form, email_form, avatar_form, background_form, password_form = ProfileInfoForm(
        instance=user_profile), UserEmailForm(instance=user), AvatarUploadForm(
        instance=user_profile), BackgroundUploadForm(instance=user_profile), PasswordChangeForm(user)
    recommendation, _ = Recommendation.objects.get_or_create(user=user)
    favorite_movies = _attach_display_titles(
        list(recommendation.favorite_movies.all().prefetch_related('titles', 'genres')))
    watchlist_movies = _attach_display_titles(list(user_profile.watchlist.all().prefetch_related('titles', 'genres')))
    user_reviews = UserReview.objects.filter(user=user).select_related('movie').prefetch_related('movie__titles')
    for review in user_reviews: setattr(review.movie, 'display_title', _display_title(review.movie))
    browsing_history = BrowsingHistory.objects.filter(user=user).select_related('movie').prefetch_related(
        'movie__titles')[:50]
    for history_item in browsing_history: setattr(history_item.movie, 'display_title',
                                                  _display_title(history_item.movie))
    context = {'profile': user_profile, 'info_form': info_form, 'email_form': email_form, 'avatar_form': avatar_form,
               'background_form': background_form, 'password_form': password_form, 'favorite_movies': favorite_movies,
               'watchlist_movies': watchlist_movies, 'user_reviews': user_reviews, 'browsing_history': browsing_history}
    return render(request, 'profile.html', context)