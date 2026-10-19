# Подготовка Docker CPU

Профиль содержит только E0, без torch/Laya/GPU. Docker Engine на машине подготовки отсутствует, поэтому **сборка/запуск контейнера ещё не проверены**. Linux wheels скачаны и сверены с версиями исходного E0, но проверка wheels на Windows не считается Linux installation test.

В готовом комплекте Linux wheelhouse и E0-артефакты уже присутствуют. Образ устанавливает зависимости офлайн из wheelhouse с SHA-256. Базовый Python image впервые потребуется получить из официального registry; он закреплён digest в Dockerfile и `base_image.json`. Для последующих полностью офлайн запусков нужен предварительно собранный image в локальном Docker. Веса моделей при старте не скачиваются.

Из исходного рабочего проекта для повторной подготовки E0:

```powershell
.\.venv-app\Scripts\python.exe scripts\prepare_model_artifacts.py
```

Проверка Docker после установки и запуска Engine, из корня комплекта:

```powershell
docker version
docker compose version
docker compose config
docker compose build --no-cache
docker compose up -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/ready
.\.venv-app\Scripts\python.exe scripts\smoke_release.py --url http://127.0.0.1:8000 --label docker-first
docker compose restart
.\.venv-app\Scripts\python.exe scripts\smoke_release.py --url http://127.0.0.1:8000 --label docker-restart
docker compose down
```

Smoke ожидает baseline ready, сравнивает настоящие ответы шести синтетических писем с сохранёнными локальными ответами и проверяет пустой/повреждённый/слишком большой/chunked файл, отказ на нулевых признаках и недоступность recall/review. Новый benchmark не выполняется. Название отчёта `docker-first` само по себе не доказательство Docker: обязательно сохранить `docker version`, `docker compose ps` и inspect/health контейнера. На машине подготовки эти команды не запускали Docker, потому что CLI/Engine отсутствуют.

Снаружи порт опубликован только на 127.0.0.1. Внутри контейнера bind 0.0.0.0 необходим для port mapping; это не публикация на внешнем интерфейсе хоста. Артефакты подключены read-only как `/artifacts/models/...`; пользователь контейнера непривилегированный, root filesystem read-only, дополнительные capabilities сняты. Сервис не монтирует домашний каталог или исходный проект.

Windows CUDA Laya остаётся отдельным проверенным локальным способом запуска. GPU Docker-профиль не объявлен: его невозможно проверить на этой машине без Docker/WSL. Linux neural runtime не подменяется Windows venv и не переносится копированием site-packages.

Первичные справочные документы: [Dockerfile](https://docs.docker.com/reference/dockerfile/), [Compose services](https://docs.docker.com/reference/compose-file/services/), [build context](https://docs.docker.com/build/concepts/context/).
