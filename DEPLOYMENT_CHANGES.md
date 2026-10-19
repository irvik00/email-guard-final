# Дополнение UI

В новой версии изменены отображение и email.category (необученные тематические правила); см. UI_UPDATE.md. Утверждение ниже «категории не добавлены» относится к предыдущей адаптации. Модели/пороги/препроцессинг/итоговые метрики по-прежнему сохранены.

# Отдельная адаптация для Render: статус и сохранность

Основа: предоставленный email_guard_submission.zip, SHA-256 cf4b6f8f6d15eadea8de51d303bf25295b15a1d43bd8306c10398fcf0f5b26a2. Исходный архив не изменён. Исходные manifests и SHA256SUMS сохранены в deployment/original_*. Старые отчёты проверки описывают исходный релиз; они не являются аудитом изменённых файлов этой копии.

Изменено только размещение и отображение статуса:

- Dockerfile включает сохранённый E0 в /artifacts/models/day3; .dockerignore явно разрешает только эти два файла модели.
- Launcher читает PORT, оставляя один uvicorn worker; healthcheck использует тот же порт. Локальный default остаётся 8000/127.0.0.1.
- app/deployment_config.py добавляет один явный HTTPS-адрес из EMAIL_GUARD_PUBLIC_URL или Render RENDER_EXTERNAL_URL. Сохраняются TrustedHostMiddleware, проверка Origin, отсутствие CORS и запрет чужого Origin.
- Интерфейс сообщает об отправке письма на сервер и отключает недоступные режимы. Категории/фишинг не добавлены.
- .gitignore исключает локальные окружения/данные; .gitattributes сохраняет точные байты при Git checkout.
- Добавлены инструкция PUBLISH_STEPS.md, integrity-проверка verify_render_package.py, public HTTP smoke и 4 теста границы публичного адреса.
- Обновлены хеши изменённых файлов приложения и manifest этой отдельной копии. Исходные финальные таблицы, MODEL_CARD, ML-файлы, протокол и зависимости сохранены.

Не изменены: E0 bundle, model_manifest, все девять shared preprocessing файлов, исходный финальный протокол, E0 threshold=0.3037252359712881, temperature=1, исходные weights/configs Laya в manifest. Laya weights не входят в CPU-копию. Дообучения, выбора новых порогов и inference на test/challenge/reserve не было.

Реально выполнено в отдельной Linux-среде Python 3.12.14 (не контейнер Python 3.12.4):

- Установлены все 24 Linux wheels из предоставленного архива с --require-hashes и без сети. pip check: No broken requirements found.
- 18 исходных тестов test_app.py и 4 новых test_public_deployment.py прошли. Предупреждение deprecation об httpx/TestClient не является ошибкой; версии зависимостей не менялись.
- Из отдельно скопированного набора runtime-файлов, соответствующих Docker COPY, запущен настоящий HTTP-сервис с PORT=10879, внешним artifact root, отключённой Laya и заданным RENDER_EXTERNAL_URL.
- При первом запуске и после остановки/нового процесса прошли по 20 HTTP smoke checks на шести вымышленных .eml. Максимальное абсолютное различие E0 scores с сохранёнными Windows-ответами: 0.
- На каждом запуске дополнительно проверены: запрос с HTTPS Origin публичного домена к HTTP-backend успешен; чужой Origin → 403; чужой Host → 400. Всего по 23 проверки на запуск.
- SHA E0/препроцессинга проверены; целостность комплекта проверяется scripts/verify_render_package.py. Подробности runtime: reports/deployment/linux_http_verification.json и сохранённые public-smoke JSON с метками linux-cpu-*.

Не выполнено: Docker build/run, запуск на Render, HTTPS через реальный reverse proxy Render, cold start Free instance, лимиты памяти Render, внешний сетевой smoke и restart облачного сервиса. Подготовка и Linux HTTP-проверка не подменяют эти проверки. Внешний URL ещё не создан. После публикации выполнить шаги 5–8 PUBLISH_STEPS.md и сохранить фактические результаты отдельно.

Предполагаемый публичный профиль: CPU E0, baseline=true, recall=false, review=false; neural runtime остаётся локальным Windows CUDA показом. Публикация не меняет оценённое качество модели и ограничения teacher overlap.
