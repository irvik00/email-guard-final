# Примечание к копии для хостинга

Ниже сохранён исторический статус исходной сдачи от 18 октября. Проверки этой адаптации и оставшийся этап реальной публикации описаны в DEPLOYMENT_CHANGES.md. Старый SUBMISSION_AUDIT не удостоверяет изменённые файлы этой копии.

# Email Guard — финальный релиз

Финальный комплект подготовлен 18 октября 2026. Основной E0 и сохранённая Laya_day5 используют исходные preprocessing/T/thresholds; обучение, новые варианты и полный benchmark не выполнялись. Рабочие артефакты и прежние финальные результаты сохранены.

## Готово

Dockerfile/Compose CPU E0, default-deny .dockerignore, фиксированные Linux dependencies с SHA, digest официального Python base, read-only model mount, localhost port. Исходные модели и финальные метрики сохранены. Созданы README, MODEL_CARD, материалы презентации/вопросов, компактные обезличенные агрегаты, подготовщик локальных артефактов и functional smoke.

## Что реально запущено и проверено

| Проверка | Фактический результат |
|---|---|
| Прежние app regression tests | 18/18, `reports/release/unit_app.txt` |
| Тесты переноса в чистом комплекте | 3/3, `reports/release/unit_release.txt`; настоящий E0 reload из отдельного каталога, совпадение scores, отказы, запрет неизвестного overwrite |
| Windows CPU из release/email_guard_submission | 30 smoke checks: readiness,6 демо×3 режима/score comparison, ошибки и потоковый лимит |
| Windows CPU после остановки и нового процесса | Ещё30 smoke checks passed; это Windows restart, не Docker restart |
| Windows CUDA Laya из чистого комплекта, с внешним model mount/interpreter | 54 smoke checks passed; baseline/recall/review, истинные компоненты, сверка с сохранёнными синтетическими ответами |
| Linux dependencies |24 wheels/65,828,660 байт подготовлены; версии совпали с исходным app lock; SHA закреплены; Linux installation не запускалась |
| Docker base | Официальный registry manifest реально получен, digest проверен по его байтам; Python3.12.4-slim-bookworm pinned |
| Комплект/архив | Allowlist, пути/типы файлов, SHA-256, E0/Laya identities и42 агрегатные metric rows проверяются независимым `audit_submission.py`; итоговый журнал — `release/SUBMISSION_AUDIT.json` |

HTTP smoke — functional equivalence, не новая независимая оценка моделей. Источник ожидаемых ответов: шесть вымышленных `.eml` первого этапа приложения; модели/вопросы/пороги не подгонялись. Результаты находятся в `reports/release/smoke_runs/` внутри сдачи. Сырые test/challenge/reserve и teacher archive не читались для inference или упаковки; компактные таблицы получены из существующих публичных итоговых Markdown-отчётов, без private inputs/predictions.

Исправленные реальные блокеры: Windows-specific путь interpreter Laya заменён переносимым выбором/операторской конфигурацией; model root можно монтировать отдельно, сохраняя relative IDs/SHA. Первый новый smoke-клиент urllib разорвал отправку oversized body при раннем413; заменён уже закреплённым httpx, серверный лимит не менялся. Первая release-фикстура ошибочно принудила ROOT/models вместо artifacts; исправлена только фикстура. Журнал первых ошибок сохранён в `initial_relocation_failures.json`; завершённые прогоны сохранены отдельно.

## Docker: точный блокер

Docker CLI не найден в PATH или стандартном каталоге Docker Desktop; Docker Desktop и service com.docker.service отсутствуют. `wsl --list --verbose` сообщил «Подсистема Windows для Linux не установлена». Поэтому build/up/readiness/upload/restart **в Docker не выполнялись**. Linux wheels и registry digest реально подготовлены, но это не проверка контейнера. GPU Docker-профиль не добавлен. Не устанавливались системные компоненты/WSL и не менялись параметры защиты ОС.

## Команды защиты

На текущей машине проверенное приложение из чистого комплекта запущено на **http://127.0.0.1:8000/**, доступны baseline/recall/review. Дополнительный временный CPU сервер8002 после smoke останавливается. Для нового запуска из исходного корня проекта (остановить прежний экземпляр перед запуском на том же порту):

```powershell
Set-Location .\release\email_guard_submission
$env:PYTHONUTF8 = '1'
$env:EMAIL_GUARD_ARTIFACT_ROOT = (Resolve-Path .\artifacts).Path
& ..\..\.venv-app\Scripts\python.exe -m app.run --disable-laya
```

Показать demo_02 (spam), demo_04 (insufficient_features), раскрыть отдельный score/threshold. Полная проверка в другом терминале из той же папки:

```powershell
& ..\..\.venv-app\Scripts\python.exe scripts\smoke_release.py --url http://127.0.0.1:8000 --label defense-cpu
```

Для всех трёх режимов на этой же машине, после остановки CPU-only экземпляра:

```powershell
$env:EMAIL_GUARD_ARTIFACT_ROOT = (Resolve-Path ..\..\artifacts).Path
$env:EMAIL_GUARD_LAYA_PYTHON = (Resolve-Path ..\..\.venv-decision\Scripts\python.exe).Path
& ..\..\.venv-app\Scripts\python.exe -m app.run
# Второй терминал, из папки комплекта:
& ..\..\.venv-app\Scripts\python.exe scripts\smoke_release.py --url http://127.0.0.1:8000 --label defense-laya --with-laya
```

Дождаться `/ready?mode=recall`. Demo_05 review показывает разногласие, инертный HTML-подобный текст и статические наблюдения; demo_06 показывает усечение. Весь показ использует настоящие ответы. Для переноса на новую Windows-машину создать `.venv-app` по README; отдельные Laya файлы подключаются через prepare_model_artifacts, trusted CUDA runtime требуется подготовить заранее.

Проверка Docker при наличии Engine:

```powershell
docker version
docker compose config
docker compose build --no-cache
docker compose up -d
Invoke-RestMethod http://127.0.0.1:8000/ready
.\.venv-app\Scripts\python.exe scripts\smoke_release.py --url http://127.0.0.1:8000 --label docker-first
docker compose restart
.\.venv-app\Scripts\python.exe scripts\smoke_release.py --url http://127.0.0.1:8000 --label docker-restart
docker compose down
```

## Комплект

`release/email_guard_submission/`, архив `release/email_guard_submission.zip`; отдельная Laya — `release/email_guard_laya_artifacts/`. Внутренний SUBMISSION_MANIFEST и SHA256SUMS, внешний `release/archives.json` фиксируют состав/хеши. Публикации/отправки в GitHub нет. Сырые датасеты и личная почта не включаются.

Большая Laya не находится в CPU-архиве/образе:6 точных файлов, около1.32GB, отдельный MODEL_FILES.json. E0 bundle2,474,353 байт и model_manifest включены в artifacts. КодML-скриптов/configs поставлен для научного воспроизведения с отдельно полученными данными/manifests, не запускается приложением. Тексты собственной почты, per-email final predictions и исходные модели C/Qwen/W0/W1/W2 отсутствуют.

## Остаточные блокеры и пределы проверки

Единственный обязательный непроверенный этап — реальная Docker сборка/запуск/restart из-за отсутствия Engine/CLI/WSL. Готовы файлы и точная последовательность проверки; Windows/PyPI/registry проверки не подменяют её. GPU Docker-профиль отсутствует. Перенос CUDA runtime на другую машину также не проверен; на текущем доверенном Windows runtime Laya работает. E0 от него не зависит.

Около187MiB RSS E0 наблюдалось после небольших synthetic inputs; это не peak RAM для10MiB и не load test. Предельная нагрузка, production мониторинг, mail integration, online дообучение, публичный хостинг и сетевой auth не реализованы. Качество и teacher overlap ограничения подробно раскрыты в MODEL_CARD; финальные метрики не изменены. Исходные модели проверены по SHA, старые model file sizes/mtimes сохранены; предыдущий app-код/manifest архивирован в reports/release/app_before перед переносимыми исправлениями.
