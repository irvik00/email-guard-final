# Email Guard: локальное приложение

Актуальный переносимый запуск и комплект сдачи описаны в корневых `README.md` и `RELEASE_STATUS.md`. Этот файл сохраняет подробности первого этапа приложения; рабочие модели и их настройки не изменены. Артефакты теперь можно подключать через `EMAIL_GUARD_ARTIFACT_ROOT`, interpreter Laya — через `EMAIL_GUARD_LAYA_PYTHON`; текущий SDK wheel расположен в `vendor/`.

Запуск из корня проекта в PowerShell:

```powershell
.\.venv-app\Scripts\python.exe -m app.run
```

Открыть `http://127.0.0.1:8000`. Остановить — Ctrl+C в терминале запуска. Использовать один экземпляр приложения; launcher фиксирует один процесс, без reload. Для запуска E0 при отсутствии CUDA/Laya:

```powershell
.\.venv-app\Scripts\python.exe -m app.run --disable-laya
```

E0 работает в отдельной `.venv-app`. Laya загружается один раз в дочерний процесс из существующего доверенного `.venv-decision`, который приложение не изменяет. Фактические зависимости экспортированы отдельно в `requirements-app.lock.txt` и `requirements-app-laya.lock.txt`; последний файл описывает используемый CUDA runtime, а не обещает переносимость на любой компьютер. Загрузка Laya идёт в фоне; до готовности доступны baseline и диагностика readiness. Никаких сетевых загрузок весов.

Восстановление web/E0 окружения при необходимости, без изменения глобального Python и старых окружений:

```powershell
New-Item -ItemType Directory -Force .cache\app-tmp | Out-Null
$env:TEMP = (Resolve-Path .cache\app-tmp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m venv .venv-app
.\.venv-app\Scripts\python.exe -m pip install --cache-dir .cache\app-pip -r requirements-app.lock.txt
```

Модельные артефакты выбирает исключительно локальный код. API принимает байты письма и один из трёх режимов; путь к модели, эталонная метка и имя файла в этот контракт не входят.

## Архитектура

`web.py` читает тело запроса порциями с пределом 10 MiB, без файлов и multipart-буфера на диске. `inference.py` вызывает прежний MIME parser и прежние builders; E0 использует полный подготовленный текст, Laya — clean_builder_v2, literal-mask sanitation и фиксированные первые 700 токенов. `worker_client.py` передаёт только подготовленный текст в приватный stdin/stdout процесс `laya_worker.py`, удерживая один neural slot. В API нет загрузки модели и в worker нет доступа к HTTP.

`contract.py` сверяет хеш финального протокола и артефактов, использует прежнее голосование. Модели, T, вопросы и operating thresholds не выбираются интерфейсом. Все исходы отказа имеют `decision=null`; диагностический score E0 на нулевом векторе не считается решением. При недостатке признаков recall не принимает результат одной Laya. Review при отказе/разногласии требует ручной проверки; инфраструктурный отказ дополнительно возвращает HTTP 503.

Интерфейс строит DOM только через `textContent`, не показывает активный HTML/тело письма и не превращает адреса в ссылки. Вывод ограничен темой/From и статическими наблюдениями. CSP, same-origin POST, loopback binding и отключённый access log дополняют этот контракт. По умолчанию загрузки, тексты и персональные поля не записываются; synthetic verification сохраняет только искусственные примеры, development verification — IDs и агрегаты совпадения. Категории не определяются: проверенного классификатора нет. Подлинность отправителя и phishing-вердикт отсутствуют.

## Проверки и воспроизведение

```powershell
$env:PYTHONUTF8 = '1'
.\.venv-app\Scripts\python.exe -m unittest discover -s tests -p test_app.py -v
.\.venv-app\Scripts\python.exe scripts\verify_app.py components
# В отдельном терминале уже запущено приложение:
.\.venv-app\Scripts\python.exe scripts\verify_app.py http
# Необязательный отдельный E0-only процесс на другом порту:
.\.venv-app\Scripts\python.exe -m app.run --port 8001 --disable-laya
# В третьем терминале:
.\.venv-app\Scripts\python.exe scripts\verify_app.py http --base-url http://127.0.0.1:8001 --baseline-only
.\.venv-app\Scripts\python.exe scripts\verify_app.py preservation
```

`verify_app.py components` обращается только к уже просмотренным 22 development-входам и предсказаниям, не к исходным личным .eml или закрытым split. Это проверка совпадения реализации, не новая оценка качества. Повторные проверки создают новый каталог отчёта. Одновременно с `components` не запускать второй экземпляр Laya: проверка сама временно загружает модель и освобождает её в конце.

`scripts/build_app_release.py` создаёт новый manifest приложения на основе фиксированного финального протокола, проверяя хеши весов и экспортируя версии. Для воспроизведения inference пересобирать manifest не нужно. После намеренного изменения кода приложения manifest пересобирают и снова проверяют, сохраняя прошлый релиз отдельно.

## Для второго этапа

Нужны `app/`, `requirements-app.lock.txt`, `requirements-app-laya.lock.txt`, точные E0/Laya файлы из `app/release_manifest.json`, перечисленные там shared-модули `src/` и `config/day4_experiments.json`, источник FINAL_EVALUATION_PROTOCOL.json по зафиксированному пути. Для SDK доступен ранее проверенный локальный wheel `.cache/day4/runtime_review/laya-0.3.20-py3-none-any.whl`; его SHA-256 указан в APP_STATUS. Исследовательские C/Qwen/W0/W1/W2, корпуса и личные письма рабочему inference не нужны. Дополнительно взять демонстрации `examples/app/`, тесты и отчёт APP_STATUS. Перенос CUDA окружения, контейнер, авторизация для сетевого доступа и публикация — отдельный этап, здесь не выполнялись.
