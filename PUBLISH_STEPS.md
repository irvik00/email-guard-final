**Обновление UI:** добавлены отдельные тематические подсказки по правилам RU/EN; это не обученная модель категорий. См. UI_UPDATE.md. Сайт по-прежнему использует сохранённый E0; Laya требует отдельно подготовленного CUDA worker.

# Публикация Email Guard: действия и проверки

Схема: код и E0 на GitHub → Render собирает Docker-образ → преподаватель открывает HTTPS-ссылку. Публично работает baseline E0. Laya и исследовательские окружения остаются в исходном локальном проекте. Эту копию не нужно переносить поверх исходного проекта.

## 1. Разложить файлы

| Файл / папка | Куда | Назначение |
|---|---|---|
| Исходный email_guard_submission.zip | Оставить отдельной резервной копией | Исходная сдача от 18 октября |
| Исходные SUBMISSION_AUDIT.json и archives.json | Рядом с исходным ZIP | Проверки/хеш именно исходного архива; это не настройки хостинга |
| Новый email_guard_render.zip | Распаковать в отдельный каталог | Копия с изменениями для публичного CPU-демо |
| Содержимое email_guard_render/ | В корень нового GitHub-репозитория | Код, документы, E0 и Linux wheels для сборки |
| Сохранённая Laya, .venv-app, .venv-decision | Оставить в исходном проекте на ПК | Локальный показ всех режимов |
| Личная почта, сырые датасеты и тест преподавателя | Оставить локально | В публичный репозиторий и на хостинг не загружаются |

В PowerShell, если ZIP скачан в Downloads:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\email_guard_render.zip" -DestinationPath "D:\Study\MLSecOps\publish"
Set-Location "D:\Study\MLSecOps\publish\email_guard_render"
py -3.12 .\scripts\verify_render_package.py
```

Если браузер сохранил файл с другим именем, подставить его имя. Не использовать -Force для существующего каталога: выбрать новый пустой каталог.

Проверка: последняя команда выдаёт JSON со status=passed. В текущем каталоге непосредственно видны Dockerfile, app/, artifacts/, deployment/, scripts/. Если они находятся внутри ещё одной папки, перейти внутрь неё. Для проверки файлов не нужны сторонние Python-библиотеки.

## 2. Создать GitHub-репозиторий

На https://github.com/new создать пустой public-репозиторий `email-guard-final`. Не добавлять через сайт README, .gitignore или лицензию: они уже есть/не выбраны. Скопировать HTTPS-адрес репозитория из страницы GitHub.

Проверка: GitHub показывает страницу пустого репозитория и адрес вида https://github.com/ТВОЙ_ЛОГИН/email-guard-final.git.

## 3. Загрузить подготовленную копию

Команды выполнять из каталога шага 1. `ТВОЙ_ЛОГИН` заменить своим GitHub-логином.

```powershell
git --version
git init
git add .
git status --short
```

Проверка перед коммитом: в списке есть исходники, Dockerfile, artifacts/models/day3, Linux wheels. Нет .venv, личных .eml и data/raw. Шесть вымышленных .eml в examples/app нужны для показа. Архив ZIP целиком в GitHub добавлять не нужно.

```powershell
git commit -m "Publish Email Guard CPU demo"
git branch -M main
git remote add origin https://github.com/ТВОЙ_ЛОГИН/email-guard-final.git
git push -u origin main
git status --short
```

Git может открыть браузер для входа в GitHub. Пароль/токен не отправлять в чат. Если Git сообщает об отсутствии имени автора, задать `git config user.name "Имя"` и `git config user.email "Ваш адрес для коммитов"` внутри этого репозитория, затем повторить commit. Для адреса можно использовать собственный noreply-адрес из GitHub Settings → Emails. Если Git не установлен, установить Git for Windows с https://git-scm.com/downloads/win и заново открыть PowerShell.

Проверка: push завершился без ошибки; последняя команда ничего не выводит. После обновления страницы GitHub видны Dockerfile, app/, src/, artifacts/, deployment/. Файл artifacts/models/day3/baseline_bundle.joblib присутствует. Папку с Linux wheels сохранить: Docker собирается без скачивания пакетов. Крупнейший файл около 35.3 MB; загружать через Git, поскольку веб-загрузчик GitHub ограничивает отдельный файл 25 MiB.

## 4. Создать Render Web Service

На https://dashboard.render.com войти, выбрать New → Web Service, подключить GitHub и выбрать email-guard-final. Для Docker-сервиса использовать следующие настройки:

| Поле | Значение |
|---|---|
| Name | email-guard-final или доступное имя |
| Branch | main |
| Language / Runtime | Docker |
| Root Directory | Пусто: Dockerfile лежит в корне репозитория |
| Dockerfile Path | ./Dockerfile |
| Docker Build Context Directory, если есть | . |
| Docker Command | Пусто: команда берётся из Dockerfile |
| Instance Type | Free |
| Health Check Path, в Advanced/Settings | /ready |

В Environment добавить:

| Key | Value |
|---|---|
| EMAIL_GUARD_DISABLE_LAYA | 1 |
| EMAIL_GUARD_ARTIFACT_ROOT | /artifacts |
| EMAIL_GUARD_HOST | 0.0.0.0 |

Эти значения уже есть в Dockerfile; их явное указание в панели облегчает проверку. PORT и RENDER_EXTERNAL_URL Render задаёт автоматически. EMAIL_GUARD_PUBLIC_URL заполнять не нужно: он нужен только для другого/собственного домена.

Проверка перед созданием: выбран Docker и именно Free. Не добавлять диск, базу данных, GPU или платный тариф. Compose и файл .env на Render не загружаются: Render использует Dockerfile и Environment. Локально устанавливать Docker для этого маршрута не требуется.

Нажать Deploy Web Service / Create Web Service и дождаться окончания сборки. Не выбирать Python runtime и не вписывать pip install / uvicorn вручную.

## 5. Проверить запуск на хостинге

Открыть Logs/Events. Должны завершиться установка пакетов с --require-hashes и pip check, сборка образа и запуск сервиса. Статус сервиса должен стать Live. Затем скопировать URL вида https://email-guard-final-XXXX.onrender.com — использовать настоящий адрес из панели.

В браузере открыть URL + /ready. Ожидается HTTP 200 и:

```json
{"modes":{"baseline":true,"recall":false,"review":false},"default":"baseline"}
```

В полном ответе есть также components и причины недоступности. Недоступность Laya в CPU-демо ожидаема. /health должен вернуть status=alive и upload_persistence=false. /ready?mode=recall должен вернуть HTTP 503.

Проверка: готовность подтверждается не только открывшейся главной страницей, но и baseline=true.

## 6. Проверить интерфейс

Открыть главную страницу. Выбрать baseline. Загрузить по очереди:

1. examples/app/demo_02.eml → «Спам».
2. examples/app/demo_04.eml → «Вердикт отсутствует», причина insufficient_features.
3. examples/app/demo_03.eml → «Обычное».

Открыть «Оценки компонентов и правило решения». У E0 сохранён порог 0.3037252359712881. Recall/review недоступны, поскольку Laya не запущена. На этом сайте письмо отправляется на сервер; для показа использовать только вымышленные примеры.

Проверка: нет Invalid host header, origin_forbidden для своего сайта и сетевой ошибки загрузки. Наличие ошибок классификации у других писем не исправлять сменой модели/порога: это измеренное ограничение модели.

## 7. Выполнить автоматическую HTTP-проверку

Из папки публикации на Windows, подставив настоящий URL:

```powershell
py -3.12 .\scripts\smoke_public.py --url https://ТВОЙ-СЕРВИС.onrender.com --label render-first
```

Проверка: status=passed, checks=20. Скрипт отправляет только шесть вымышленных примеров, сверяет ответы и scores с сохранёнными значениями, проверяет ошибки и запрет чужого Origin. Результат сохраняется в reports/deployment/public-smoke-...-render-first.json. Это функциональная проверка, а не новый benchmark качества модели. Прежний smoke_release.py предназначен для localhost и для публичного URL не подходит.

Если видна страница пробуждения Render, сначала дождаться нормального /ready, затем повторить проверку.

## 8. Проверить повторный запуск

В Render открыть страницу Deploys → Manual Deploy → Restart service. Дождаться Live и успешного /ready. Затем:

```powershell
py -3.12 .\scripts\smoke_public.py --url https://ТВОЙ-СЕРВИС.onrender.com --label render-after-restart
```

Проверка: снова passed, 20 проверок. Сохранить оба JSON, ссылку и Git commit SHA из Render. Только после этого можно утверждать: «Docker-образ собран и запущен на Render, HTTP-проверки до и после перезапуска прошли». Не писать, что docker compose проверен локально: это другой маршрут.

## 9. Подготовить сдачу

В презентацию и LMS, согласно формату сдачи, добавить:

- Публичную ссылку приложения.
- Ссылку на GitHub с кодом, сохранённым E0, MODEL_CARD и отчётами.
- Презентацию/ссылку на неё.
- Если нужен архив сдачи — исходный архив и отдельную копию публикации с пояснением, либо новую зафиксированную версию с честным журналом изменений. Исходный аудит относится к исходному ZIP.

На слайде приложения: «Публичное CPU-демо E0; E0+Laya/OR/review — локальная демонстрация». Категории писем и надёжная детекция фишинга не реализованы. Не представлять опубликованный E0 как подтверждение мультиязычного качества.

Free-сервис Render засыпает после 15 минут без входящего трафика; пробуждение обычно занимает около минуты. Перед защитой открыть сайт и проверить /ready. Локальный рабочий запуск оставить запасным способом показа.

## Если проверка не прошла

| Симптом | Что проверить |
|---|---|
| Dockerfile not found | Dockerfile должен быть в корне GitHub; Root Directory пустой; путь ./Dockerfile |
| COPY artifacts/... not found | В репозитории должны быть оба файла E0; использовать эту новую .dockerignore |
| Приложение не обнаруживает порт | Runtime Docker; Docker Command пусто; EMAIL_GUARD_HOST=0.0.0.0 |
| Invalid host header | Запущена именно новая копия app/web.py и app/deployment_config.py; Render должен задать RENDER_EXTERNAL_URL |
| origin_forbidden на собственном сайте | Открыть прямой https://...onrender.com URL; убрать ошибочный EMAIL_GUARD_PUBLIC_URL |
| /ready возвращает 503 для baseline | Посмотреть components.E0.reason и последние строки Logs; не менять модель/хеши ради прохождения |
| Нет Free / просят другой тариф | Не покупать автоматически; уточнить доступные варианты по текущей странице |
| Скрипт сообщает несовпадение score/решения | Сохранить сообщение и проверить версию/commit/файлы; не подгонять expected outputs |

Официальные инструкции, проверенные 19 октября 2026:
https://render.com/docs/docker
https://render.com/docs/environment-variables
https://render.com/docs/free
https://render.com/docs/deploys
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
