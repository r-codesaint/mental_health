# mental Django app

Minimal Django app skeleton for mental health notes.

How to use

1. Copy the `mental` folder into your Django project (or place at project root and add `'mental'` to INSTALLED_APPS in `settings.py`).
2. Run migrations: `python manage.py makemigrations mental` then `python manage.py migrate`.
3. Start the development server and visit the app's URLs (include `path('', include('mental.urls'))` in your project's urls). 

Requirements

- Django

Quick start (PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r mental/requirements.txt
```

3. Run migrations and start server:

```powershell
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Then open http://127.0.0.1:8000/ in your browser. The admin site is at /admin/ (create a superuser with `python manage.py createsuperuser`).

