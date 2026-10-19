**Обновление интерфейса и тематических подсказок:** [UI_UPDATE.md](UI_UPDATE.md). Публичное CPU-демо остаётся на E0; Laya доступна только при фактически готовом CUDA worker.

# Версия для публикации CPU E0

Это отдельная адаптация исходной сдачи. Начните с [PUBLISH_STEPS.md](PUBLISH_STEPS.md): GitHub → Render Web Service, Docker, Free. Изменения и границы проверки: [DEPLOYMENT_CHANGES.md](DEPLOYMENT_CHANGES.md). Веса/препроцессинг/порог E0 и финальные метрики сохранены. В образ включён E0; Laya не включена. Приведённые далее локальные инструкции и результаты относятся к исходной сдаче; на Render используется новый раздел публикации.

# Email Guard

Локальная учебная система классификации спама в `.eml`: сохранённый char TF-IDF + Logistic Regression (E0) и два дополнительных режима с адаптированной Laya_day5. Веб-интерфейс на русском, FastAPI API. Письмо анализируется в памяти и по умолчанию не сохраняется; ссылки, HTML/JavaScript и вложения не исполняются.

Это демонстрация воспроизводимого ML-проекта, а не готовая почтовая защита. «Обычное» означает только решение spam-классификатора. Подлинность From, безопасность вложений, фишинг и тематическая категория не подтверждаются. Ограничения данных, coverage и пересечения — в [MODEL_CARD.md](MODEL_CARD.md).

## Что входит в сдачу

В `release/email_guard_submission/` — чистый код, зависимости, Docker/config, демонстрации, компактные таблицы и небольшой E0 bundle. `release/email_guard_submission.zip` — тот же комплект. Большие Laya-артефакты находятся отдельно в `release/email_guard_laya_artifacts/`; подключение описано ниже. Контрольные суммы — `SUBMISSION_MANIFEST.json`, `SHA256SUMS.txt` внутри комплекта и внешний `release/archives.json`.

Личные письма, сырые/обработанные корпуса, teacher archive, приватные predictions, `.venv`, исходные кеши и лишние исследовательские веса в комплект не входят. Локальный SDK wheel Laya включён отдельно как проверенная зависимость, без всего исследовательского окружения.

## Windows: обязательный CPU baseline

Перейти в распакованный `email_guard_submission`. Нужен Python 3.12 x64; фактически проверенная версия — **3.12.4**. Если `.venv-app` ещё нет:

```powershell
New-Item -ItemType Directory -Force .cache\app-tmp | Out-Null
$env:TEMP = (Resolve-Path .cache\app-tmp).Path
$env:TMP = $env:TEMP
py -3.12 -m venv .venv-app
.\.venv-app\Scripts\python.exe -m pip install -r requirements-app.lock.txt
```

При первой установке Python-пакетов нужен доступ к package index. Веса моделей уже подготовлены; inference работает без их скачивания. В исходном рабочем проекте окружение уже существует, повторная установка не нужна.

```powershell
$env:PYTHONUTF8 = '1'
$env:EMAIL_GUARD_ARTIFACT_ROOT = (Resolve-Path .\artifacts).Path
.\.venv-app\Scripts\python.exe -m app.run --disable-laya
```

Открыть **http://127.0.0.1:8000/**. Ctrl+C останавливает сервер. Использовать один экземпляр, без нескольких ASGI workers/reload. Baseline работает без torch, CUDA, Laya и интернета. `/ready` проверяет baseline; `/ready?mode=recall` отдельно сообщает недоступность зависимого режима.

## Дополнительные recall/review: сохранённая локальная Laya

Проверенный способ — Windows + имеющийся доверенный CUDA runtime, с версиями из `requirements-app-laya.lock.txt`. На машине автора это `.venv-decision`, GPU RTX 4060 Ti. Не копировать Windows venv в Linux.

В распакованном комплекте объединить отдельный пакет Laya с каталогом артефактов:

```powershell
.\.venv-app\Scripts\python.exe scripts\prepare_model_artifacts.py --source ..\email_guard_laya_artifacts --destination artifacts --only-laya
$env:EMAIL_GUARD_ARTIFACT_ROOT = (Resolve-Path artifacts).Path
# Указать существующий проверенный interpreter, а не путь к новой модели:
$env:EMAIL_GUARD_LAYA_PYTHON = (Resolve-Path ..\trusted-laya-runtime\Scripts\python.exe).Path
.\.venv-app\Scripts\python.exe -m app.run
```

В исходном проекте без отдельного пакета: `scripts/prepare_model_artifacts.py --with-laya`, затем обычный `python -m app.run`. Операторская переменная `EMAIL_GUARD_LAYA_PYTHON` задаёт только interpreter; модель/вопрос/пороги фиксированы и проверяются по manifest. Настройки не доступны через API.

Для новой машины отдельный neural environment требуется подготовить по точному freeze и CUDA-сборке torch; installer этого окружения на новой машине не проверялся. Не запускать `laya.Router()` с автоматическим скачиванием/выбором checkpoints. Используется прямой локальный `laya.load(..., fast=False)` и только Laya_day5. Для проверки исходного SDK в `vendor/` есть ранее проверенный wheel 0.3.20 с хешем в manifest. Нет молчаливого CPU fallback нейросети.

## Docker: подготовлен CPU-образ E0

**Docker CLI/Desktop отсутствуют, WSL не установлен на машине подготовки. Сборка, запуск, restart и HTTP внутри Docker здесь не выполнялись.** Реальные Windows-проверки не объявляются Docker-проверками. GPU Docker-профиля нет: его было невозможно проверить.

После установки и запуска Docker Engine с Linux containers, из корня комплекта:

```powershell
docker version
docker compose config
docker compose build --no-cache
docker compose up -d
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Linux/amd64 dependencies уже лежат в `deployment/wheelhouse-linux/`; установка в образе офлайн, версии и SHA закреплены в `requirements-docker.lock.txt`. Базовый Python 3.12.4 image закреплён digest, но его нужно предварительно получить из registry. Сеть не нужна для inference подготовленного image с локальными артефактами. Нейросетевые пакеты и веса в CPU-образ не входят.

Compose публикует только `127.0.0.1:8000`, запускает непривилегированного пользователя, подключает `artifacts/` read-only. Внутренний bind 0.0.0.0 нужен для port mapping. При занятом порте задать `$env:EMAIL_GUARD_PORT='8002'`. `.env.example` не содержит секретов. Не менять публикацию на внешний интерфейс ради защиты проекта.

Полная последовательность smoke/restart — [deployment/README.md](deployment/README.md). Непроверенный Docker-профиль нельзя показывать как выполненный эксперимент.

## Модели, конфигурация и отказы

| Режим | Правило | При отказе компонента |
|---|---|---|
| baseline, default | E0 score ≥ 0.3037252359712881 → spam | `decision=null`, явный status |
| recall | E0 OR Laya, только два успешных бинарных ответа | нет автоматического решения |
| review | совпавшие решения → общий ответ | разногласие/отказ → needs_review |

Laya: вопрос B `Is this message unsolicited spam?`, noul, yes→spam; T=0.8553398254508641, threshold=0.5228186297984043; первые 700 state tokens, context 1024/head 256. E0 использует исходный полный builder, Laya — прежний clean_builder_v2. Scores не усредняются; вероятности ансамбля нет. Режим OR не объявлен независимо выбранным победителем и не назначен default по teacher test.

Фиксированные артефакты перечислены в `app/release_manifest.json` и `MODEL_ARTIFACTS.md`. По умолчанию код ищет `artifacts/models/...` (если каталога artifacts нет — прежний project/models); `EMAIL_GUARD_ARTIFACT_ROOT` переносит только корень хранения. SHA-256 и относительные идентификаторы остаются неизменными. Исторические manifests/bundle могут содержать пути машины автора как provenance; они не используются для открытия моделей при запуске и не переписывались.

HTTP 200 может содержать `insufficient_features` или `needs_review`, что не является ham. При инфраструктурной недоступности необходимого компонента — 503; при занятости — 429; при пустом/повреждённом файле — 400; превышение 10 MiB — 413. Исходный MIME parser терпим к восстановимым дефектам: они показываются как предупреждения. Нулевой E0 вектор — отказ, а не intercept ham. Наличие отписки, знакомого бренда/From не служит безусловным пропуском.

## API и проверки без нового benchmark

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/analyze?mode=baseline" -H "Content-Type: message/rfc822" --data-binary "@examples/app/demo_02.eml"
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
.\.venv-app\Scripts\python.exe scripts\smoke_release.py --url http://127.0.0.1:8000 --label local-cpu
.\.venv-app\Scripts\python.exe -m unittest discover -s tests -p test_app.py -v
.\.venv-app\Scripts\python.exe -m unittest discover -s tests -p test_release.py -v
```

Для работающей Windows Laya добавить `--with-laya` к smoke. Документация API — `/api-docs`, схема — `/openapi.json`. Тело запроса — сырые `.eml` bytes, не multipart; имя файла/путь/метка не передаются моделям. Лимит при потоковом чтении — 10 MiB. API не загружает пользовательские модели и не принимает thresholds/questions.

## Ресурсы

Фактическая машина: Windows, Python 3.12.4, системная физическая RAM31.84 GiB (конфигурация32 GB), RTX4060Ti16 GiB. E0 реально запускается на CPU; его основной процесс после шести синтетических примеров имел RSS196,214,784 байт (около187 MiB; launcher отдельно около4 MiB). Точные наблюдения текущего запуска — `reports/release/runtime_probe.json` и RELEASE_STATUS. Это не измерение худшего случая на файле 10 MiB и не нагрузочный тест.

Прежний final Laya inference показал peak allocated VRAM **1,364,760,064 байт** и load 5.84 s на historical test в своём запуске; это память torch, не вся зарезервированная VRAM процесса, и не требование к любому GPU. Сам файл Laya weights около1.29 GB. Рекомендуемый стартовый запас для демонстрации: CPU2 cores/RAM2–4 GiB для E0, совместимая CUDA GPU с несколькими GiB свободной VRAM для Laya; это ориентиры, минимумы на других машинах не измерены. Docker CPU на этой машине не профилировался.

## Структура

```text
app/                      FastAPI, безопасный UI, fixed inference и worker
artifacts/models/day3/     E0 bundle и проверяемый manifest
src/                      исходные ML/data/evaluation scripts, без запуска при старте
config/                   неизменные configs экспериментов
deployment/               CPU wheelhouse, base-image metadata, примеры запуска
examples/app/              шесть вымышленных .eml
tests/                    app и regression tests
reports/release/           компактные агрегаты, synthetic reference и smoke
vendor/                   проверенный SDK wheel Laya
Dockerfile, compose.yaml   CPU-only контейнер; baseline не зависит от нейросети
```

## Воспроизведение выполненных обучения и оценок

В этом финальном этапе обучение и полный ML-benchmark **не запускались**. Комплект содержит исходный код и configs, но не корпуса. Для научного повторения сначала нужны разрешённые исходные данные, прежние нормализованные splits/manifests и указанные в них хеши. Зафиксированные версии исследовательских окружений — `requirements-lock.txt`, `requirements-decision-lock.txt`; они отделены от runtime приложения.

В отдельной исследовательской копии с данными:

```powershell
# Исторический baseline, 5-fold CV и validation:
.\.venv\Scripts\python.exe -B src\train_baselines.py
# Ограниченная day5 head-only adaptation, calibration/dev и validation:
.\.venv\Scripts\python.exe -B src\run_day5.py
# Прежний final evaluation runner; повтор создаст новый протокол/run:
.\.venv\Scripts\python.exe -B src\run_final_evaluation.py run
```

Это команды реального нового обучения/оценки, не команды демонстрации. Повторное открытие уже измеренных tests не возвращает им независимость. Внутренние ограничения доступа и markers нельзя снимать ради удобного повтора. Не скачивать/восстанавливать security_blocked 2026-09 и не менять Defender. Детали выполненных шагов — `docs/DAY5_REPRODUCTION.md`, `docs/EXTRA_EXPERIMENTS_REPRODUCTION.md`, `docs/DAY3_BASELINES.md`; компактные сохранённые оценки — `reports/release/FINAL_TABLES.md`. Новые исследования/пороги/вопросы в релизе не добавлены.

Материалы защиты: [PRESENTATION_HANDOFF.md](PRESENTATION_HANDOFF.md), [DEFENSE_QA.md](DEFENSE_QA.md), [RELEASE_STATUS.md](RELEASE_STATUS.md).
