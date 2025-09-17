from django.template.loader import render_to_string
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
from django.db.models import Prefetch
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
# --- 新增：导入高级查询工具 ---
from django.db.models import Q, Case, When, Value, IntegerField

from django.core.cache import cache


# --- 辅助函数：带优先级排序的内存搜索 ---
def search_from_index(query):
    query = query.lower()
    search_index = cache.get('global_search_index')

    if not search_index:
        # 降级方案：如果缓存不存在，执行带优先级的数据库查询
        qs = Movie.objects.filter(
            Q(titles__title_text__icontains=query) | Q(original_title__icontains=query) |
            Q(genres__name__icontains=query) | Q(directors__name__icontains=query) | Q(actors__name__icontains=query)
        ).distinct()
        qs = qs.annotate(
            match_priority=Q(Case(
                When(Q(titles__title_text__icontains=query) | Q(original_title__icontains=query), then=Value(3)),
                When(Q(directors__name__icontains=query) | Q(actors__name__icontains=query), then=Value(2)),
                When(Q(genres__name__icontains=query), then=Value(1)),
                default=Value(0),
                output_field=IntegerField()
            ))
        ).order_by('-match_priority', '-truth_score')
        return qs

    # 在内存中进行高速的、带优先级的查找
    matches = {}  # 使用字典来存储匹配结果和最高优先级
    for item in search_index:
        priority = 0
        if query in item['title_text']:
            priority = 3
        elif query in item['people_text']:
            priority = 2
        elif query in item['genre_text']:
            priority = 1

        if priority > 0:
            # 只存储每个ID的最高匹配优先级
            if item['id'] not in matches or priority > matches[item['id']]['priority']:
                matches[item['id']] = {
                    'priority': priority,
                    'truth_score': item['truth_score']
                }

    if not matches:
        return Movie.objects.none()

    # 对匹配结果进行排序：主关键字为优先级，次关键字为真值分数
    sorted_ids = sorted(matches.keys(), key=lambda id: (matches[id]['priority'], matches[id]['truth_score']),
                        reverse=True)

    # 为了保持排序，需要使用Case/When从数据库中按指定顺序获取对象
    ordering = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(sorted_ids)])
    return Movie.objects.filter(pk__in=sorted_ids).order_by(ordering)

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
    # --- 升级：应用智能搜索逻辑 ---
    query = request.GET.get('q', '').strip()

    # --- 升级：智能高亮逻辑 ---
    # 1. 优先从URL参数获取用户明确点击的 genre ID
    selected_genre_id = request.GET.get('genre')

    # 2. 如果没有明确的 genre ID，但有文本查询，则尝试将文本匹配为 Genre
    if query and not selected_genre_id:
        # 使用 iexact 进行不区分大小写的精确匹配
        matching_genre = Genre.objects.filter(name__iexact=query).first()
        if matching_genre:
            # 如果找到了匹配的类型，就使用它的ID作为高亮目标
            selected_genre_id = matching_genre.id
    # --- 结束 ---

    if query:
        # --- 升级：使用缓存索引进行搜索 ---
        qs = search_from_index(query)
    else:
        qs = Movie.objects.all().order_by('-release_year')

    # 如果用户点击了类型标签（即使是在搜索结果页），进一步过滤
    if selected_genre_id:
        qs = qs.filter(genres__id=selected_genre_id)

    paginator = Paginator(qs, 24)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    movies_with_titles = _attach_display_titles(list(page_obj.object_list))
    page_obj.object_list = movies_with_titles

    genres = list(Genre.objects.all())
    context = {
        'page_obj': page_obj,
        'genres': genres,
        # 确保将最终确定的 selected_genre_id (可能为 str) 转换为 int
        'selected_genre_id': int(selected_genre_id) if selected_genre_id else None,
        'query': query,
    }
    return render(request, 'movie_list.html', context)


def movie_detail(request, movie_id):
    movie = get_object_or_404(
        Movie.objects.prefetch_related('titles', 'genres', 'directors', 'actors', 'scriptwriters'), pk=movie_id)

    if request.method == 'POST' and request.user.is_authenticated:
        review_form = UserReviewForm(request.POST, user=request.user, movie=movie)
        if review_form.is_valid():
            # --- 核心修复：正确的保存逻辑 ---
            # 1. 先创建一个模型实例，但不提交到数据库
            new_review = review_form.save(commit=False)
            # 2. 手动关联必需的 user 和 movie 字段
            new_review.user = request.user
            new_review.movie = movie
            # 3. 现在可以安全地保存了
            new_review.save()
            # --- 修复结束 ---

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                html = render_to_string(
                    'partials/_comment_card.html',
                    {'review': new_review, 'user': request.user, 'liked_review_ids': set()},
                    request=request
                )
                return JsonResponse({'status': 'ok', 'html': html})

            messages.success(request, "你的评论已成功发布！")
            return redirect('movie_frontend:movie_detail', movie_id=movie.id)
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'errors': review_form.errors}, status=400)

    # ... (GET 请求处理部分无修改) ...
    review_form = UserReviewForm()
    setattr(movie, 'display_title', _display_title(movie))
    external_reviews = Review.objects.filter(movie=movie).select_related('source').order_by('-content_date')[:20]
    user_reviews = UserReview.objects.filter(movie=movie).select_related('user').order_by('-timestamp')
    is_in_watchlist = False
    is_in_favorites = False
    liked_review_ids = set()
    if request.user.is_authenticated:
        BrowsingHistory.objects.update_or_create(user=request.user, movie=movie)
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        is_in_watchlist = profile.watchlist.filter(pk=movie.id).exists()
        recommendation, _ = Recommendation.objects.get_or_create(user=request.user)
        is_in_favorites = recommendation.favorite_movies.filter(pk=movie.id).exists()
        liked_review_ids = set(
            UserReview.objects.filter(id__in=user_reviews.values_list('id', flat=True), liked_by=request.user)
            .values_list('id', flat=True)
        )
    context = {
        'movie': movie,
        'external_reviews': external_reviews,
        'user_reviews': user_reviews,
        'review_form': review_form,
        'is_in_watchlist': is_in_watchlist,
        'is_in_favorites': is_in_favorites,
        'liked_review_ids': liked_review_ids,
    }
    return render(request, 'movie_detail.html', context)


# --- 新增点赞视图 ---
@require_POST
@login_required
def like_review(request, review_id):
    review = get_object_or_404(UserReview, pk=review_id)
    user = request.user

    is_liked = False
    if user in review.liked_by.all():
        review.liked_by.remove(user)
        review.likes_count = F('likes_count') - 1
    else:
        review.liked_by.add(user)
        review.likes_count = F('likes_count') + 1
        is_liked = True

    review.save()
    review.refresh_from_db()  # 从数据库重新加载以获取最新的 likes_count

    # 返回JSON响应
    return JsonResponse({
        'status': 'ok',
        'likes_count': review.likes_count,
        'is_liked': is_liked
    })


# --- 新增视图 ---

@login_required
def edit_review(request, review_id):
    review = get_object_or_404(UserReview, pk=review_id, user=request.user)

    if request.method == 'POST':
        form = UserReviewForm(request.POST, instance=review, user=request.user, movie=review.movie)
        if form.is_valid():
            updated_review = form.save()
            # --- 升级：为AJAX请求返回JSON ---
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'ok',
                    'rating': updated_review.rating,
                    'review_text': updated_review.review
                })
            messages.success(request, "评论修改成功！")
            return redirect('movie_frontend:movie_detail', movie_id=review.movie.id)
        else:
             if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)

    # GET请求保持不变，仍然渲染编辑页面
    form = UserReviewForm(instance=review)
    return render(request, 'edit_review.html', {'form': form, 'review': review})


@require_POST
@login_required
def delete_review(request, review_id):
    review = get_object_or_404(UserReview, pk=review_id, user=request.user)
    movie_id = review.movie.id
    review.delete()

    # --- 升级：为AJAX请求返回JSON ---
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'status': 'ok', 'message': '评论已删除。'})

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
# ... (其他 import 保持不变) ...

@login_required
def recommendations(request):
    # 这个视图现在非常简单，只负责渲染一个“容器”页面。
    # 所有的推荐数据都将通过前端的AJAX请求，从 films_recommender_system 的API获取。
    return render(request, 'recommendations.html')


def choose_favorites(request):
    if request.method == 'POST':
        # 确保用户已登录才能提交喜好
        if not request.user.is_authenticated:
            return redirect('login')

        favorite_ids_str = request.POST.get('favorite_ids_json', '[]')
        try:
            ids = set(map(int, json.loads(favorite_ids_str)))
        except (json.JSONDecodeError, ValueError):
            ids = set()

        # 获取或创建用户的推荐配置
        rec, _ = Recommendation.objects.get_or_create(user=request.user)

        # 使用 .set() 方法高效地更新多对多关系
        # 它会自动处理添加、删除和清空的操作
        rec.favorite_movies.set(Movie.objects.filter(id__in=ids))

        messages.success(request, "你的喜好已更新！正在为你生成新的推荐...")

        # 直接重定向到推荐页面
        # 推荐页面加载时，会自动调用API获取基于新喜好的推荐
        return redirect('movie_frontend:recommendations')

    # --- GET 请求处理 ---
    query = request.GET.get('q', '').strip()
    partial_target = request.GET.get('partial', None)

    genres_results = Person.objects.none()
    people_results = Genre.objects.none()
    if query:
        genres_results = Genre.objects.filter(name__icontains=query)
        people_results = Person.objects.filter(name__icontains=query)

    if query:
        movies_qs = search_from_index(query)
    else:
        # 无搜索时，按真值分数排序
        movies_qs = Movie.objects.all().order_by('-truth_score')

    # --- 演员和导演列表查询 (逻辑不变) ---
    all_genres = Genre.objects.all().order_by('name')
    directors_qs = Person.objects.filter(directed_movies__isnull=False).distinct().order_by('name')
    actors_qs = Person.objects.filter(acted_in_movies__isnull=False).distinct().order_by('name')

    # --- 分页逻辑 (保持不变) ---
    movie_page_number = request.GET.get('movie_page', 1)
    director_page_number = request.GET.get('director_page', 1)
    actor_page_number = request.GET.get('actor_page', 1)

    movies_paginator = Paginator(movies_qs, 18)
    directors_paginator = Paginator(directors_qs, 30)
    actors_paginator = Paginator(actors_qs, 30)

    page_obj_movies = movies_paginator.get_page(movie_page_number)
    page_obj_directors = directors_paginator.get_page(director_page_number)
    page_obj_actors = actors_paginator.get_page(actor_page_number)

    movies_with_titles = _attach_display_titles(list(page_obj_movies.object_list))
    page_obj_movies.object_list = movies_with_titles

    # --- 获取用户喜好 (逻辑不变) ---
    favorite_people_ids, favorite_genre_ids, initial_favorite_movie_ids = set(), set(), []
    if request.user.is_authenticated:
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        favorite_people_ids = set(profile.favorite_people.values_list('id', flat=True))
        favorite_genre_ids = set(profile.favorite_genres.values_list('id', flat=True))
        rec, _ = Recommendation.objects.get_or_create(user=request.user)
        initial_favorite_movie_ids = list(rec.favorite_movies.values_list('id', flat=True))

    context = {
        'page_obj_movies': page_obj_movies,
        'page_obj_directors': page_obj_directors,
        'page_obj_actors': page_obj_actors,
        'all_genres': all_genres,
        'query': query,
        'favorite_people_ids': favorite_people_ids,
        'favorite_genre_ids': favorite_genre_ids,
        'initial_favorite_movie_ids_json': json.dumps(initial_favorite_movie_ids),
        'genres_results': genres_results,
        'people_results': people_results,
    }

    # --- 升级4：根据 partial 参数返回不同的HTML片段 ---
    if partial_target == 'directors':
        return render(request, 'partials/_people_grid.html',
                      {'page_obj': page_obj_directors, 'favorite_people_ids': favorite_people_ids,
                       'param_name': 'director_page'})

    if partial_target == 'actors':
        return render(request, 'partials/_people_grid.html',
                      {'page_obj': page_obj_actors, 'favorite_people_ids': favorite_people_ids,
                       'param_name': 'actor_page'})

    # 主搜索的AJAX请求
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'partials/_favorites_search_results.html', context)

    # 正常的全页面加载
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