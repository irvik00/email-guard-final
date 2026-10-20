# День 5: воспроизведение выполненного ограниченного эксперимента

Рабочая папка `D:\Study\MLSecOps\final\email_guard`. Основная `.venv`
и отдельная `.venv-decision` сохранены. Установок/обновлений в день5 не было;
версии закреплены существующими requirements-lock/requirements-decision-lock.
Локальный checkpoint из manifest дня4, revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`, subfolder multilingual.
Запуск полностью офлайн, с запретом socket.connect/bind и file allowlist.
Не запускать pipeline/audit дня2 и не скачивать security_blocked2026-09.

## Выполненный запуск

Run `20261012T083529Z-0950e3dc34`. Архив отчётов `reports/day5/runs/<run_id>`,
модели `models/day5/runs/<run_id>`. День3/4, Laya-A и диагностика сохранены.
Конфигурация `config/day5_laya.json` записана до prepare/обучения/оценки.
Вопрос B одинаковый до/после. Обучены существующие head/type_emb/scorer,
encoder и act_head frozen, raw false/true CE. Это не RLCD/LoRA/Qwen.

Первый training-runtime snapshot сохранён в code_snapshots и проверен SHA из
training_started.json. После fit helper дополнен только final-inference error
coverage/real_device metadata; training math не менялась. Actual code hashes
начала training и final validation сохранены раздельно. PyTorch предупредил
о недетерминированном CUDA attention backward: seed/order воспроизводимы,
бит-в-бит новые веса не обещаются. Перезагрузка текущих весов проверена точно.

## Новый полный запуск (повторно обучит один новый candidate)

```powershell
.\.venv\Scripts\python.exe -B src/run_day5.py
```

Driver последовательно выполняет команды ниже, создаёт новый run_id, не
перезаписывает архивы старых запусков. Не выполнять его ради просмотра отчётов:
это реальное новое обучение и новый заключительный validation comparison.

```powershell
.\.venv\Scripts\python.exe -B src/prepare_day5.py
.\.venv-decision\Scripts\python.exe -u -B src/train_laya_day5.py
.\.venv-decision\Scripts\python.exe -u -B src/evaluate_laya_day5.py
.\.venv\Scripts\python.exe -u -B src/evaluate_frozen_lr_day5.py
.\.venv\Scripts\python.exe -B src/personal_email_review.py
.\.venv\Scripts\python.exe -B src/check_day5.py
.\.venv-decision\Scripts\python.exe -B src/reload_day5.py
.\.venv-decision\Scripts\python.exe -B src/verify_day5.py
.\.venv\Scripts\python.exe -B src/report_day5.py
```

Три первых model/data scripts принимают `--run-id`, LR evaluator тоже.
Training_started и validation_started запрещают незаметно повторять fit/final
comparison в одном run. Если возник сбой, сохранить журнал/реальную причину;
не удалять marker ради скрытого перезапуска. Script-level OOM retry один:
microbatch1/accumulation16, effective batch16, тот же набор fit и optimizer step.
В фактическом запуске retries0; завершены все300 steps за~190с с dev/checks.

Fit pool6000, один seeded shuffle;300×16 предъявлений =4800unique, первые3
steps — реальный gradient smoke, входят в300. Dev checkpoint0/100/200/300;
выбор только среди обученных checkpoints. Calibration1000 отдельно после
выбора checkpoint, threshold по600internal dev. Manifest всех кандидатов/T/
threshold сохранён ДО открытия existing validation1275/5990.

## Только проверки/отчёты существующих артефактов

```powershell
.\.venv\Scripts\python.exe -B src/check_day5.py
.\.venv-decision\Scripts\python.exe -B src/verify_day5.py
.\.venv\Scripts\python.exe -B src/report_day5.py
```

Эти команды не обучают, не открывают corpus validation/test/challenge и не
делают нового inference benchmark. Audit читает сохранённые predictions,
membership, source/candidate safetensors, fingerprints и access logs.
Проверка reload была реально выполнена training script: штатный SDK,
12internal-dev records на candidate, logit difference/решения/T/mapping.
Независимый reload_day5.py дополнительно сравнил ОБА сырых false/true logits,
score и SDK T на16internal-dev records на candidate (4наdomain/class,
фиксированный SHA-отбор, не по ошибкам). Это отдельный subprocess без fit.
Он не входит в три команды artifact-only проверки выше, поскольку выполняет
небольшой реальный inference на internal dev; full validation не повторяется.

## Собственные письма

См. PERSONAL_EMAIL_REVIEW.md и personal_email_manifest_template.csv.
`data/personal_review` отдельно и gitignored. Пустой шаблон → templates_ready.
Сегодня никаких модельных оценок/обучения на собственных письмах.
Сначала пользователь фиксирует review/dev или отложенную диагностическую роль.
CSV/parser command — в документе; лимит20MiB, статический MIME, no network.

## Для разбора и презентации

Передать только DAY5_STATUS.md, training.md, calibration.md, comparison.md,
error_analysis.md и PRESENTATION_NOTES_DAY5.md. Не передавать weights,
private visible fragments и большие JSONL. Validation использовалась ранее;
современная почта/финальная независимая оценка остаются следующими этапами.
Default E0 с порогом0.3037252359712881 не менялся.
