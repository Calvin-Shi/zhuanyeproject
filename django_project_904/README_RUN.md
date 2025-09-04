
# Movie Recommender (Patched)

## Quickstart
```bash
# 1) create venv & install
pip install -r requirements.txt

# 2) migrate
python manage.py makemigrations
python manage.py migrate

# 3) create admin
python manage.py createsuperuser

# 4) run
python manage.py runserver
```

## Routes
- Frontend: `/` (home), `/movies/`, `/movies/<id>/`, `/recommendations/` (login required)
- Auth pages: `/accounts/login/` (uses your templates/registration/login.html)
- API: `/api/movies/`, `/api/user-reviews/`
- JWT: `/api/login` (obtain), `/api/token/refresh`

## Notes
- Patched files:
  - `config/urls.py`
  - `films_recommender_system/models.py`
  - `films_recommender_system/serializers.py`
  - `films_recommender_system/views.py`
  - `films_recommender_system/urls.py`
  - `movie_frontend/views.py`
  - `movie_frontend/urls.py`
  - `config/settings.py`
  - `requirements.txt` (added)
  - `README_RUN.md` (added)
