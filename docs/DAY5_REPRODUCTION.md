# День 5: воспроизведение выполненного ограниченного эксперимента

# Что было сделано

- Работа выполнена в `D:\Study\MLSecOps\final\email_guard` с сохранением основной `.venv` и отдельной `.venv-decision`.
- Установок и обновлений в день 5 не было; версии закреплены существующими `requirements-lock` / `requirements-decision-lock`.
- Использован локальный checkpoint из manifest дня 4: revision `7b928d828b7b0e022f929d9bd2e44165aa270148`, subfolder `multilingual`.
- Запуск выполнен полностью офлайн: с запретом `socket.connect/bind` и file allowlist.
- Pipeline/audit дня 2 не запускались; `security_blocked2026-09` не скачивался.
- Выполнен run `20261012T083529Z-0950e3dc34`; отчёты сохранены в `reports/day5/runs/<run_id>`, модели — в `models/day5/runs/<run_id>`.
- Артефакты дня 3/4, Laya-A и диагностика сохранены.
- Конфигурация `config/day5_laya.json` записана до prepare/обучения/оценки; вопрос B одинаковый до/после.
- Обучены существующие `head/type_emb/scorer`; `encoder` и `act_head` frozen; raw `false/true` CE. Это не RLCD/LoRA/Qwen.
- Первый training-runtime snapshot сохранён в `code_snapshots` и проверен по SHA из `training_started.json`.
- После fit helper дополнен только final-inference error coverage / real_device metadata; training math не менялась.
- Actual code hashes начала training и final validation сохранены раздельно.
- PyTorch предупредил о недетерминированном CUDA attention backward: seed/order воспроизводимы, бит-в-бит новые веса не обещаются.
- Перезагрузка текущих весов проверена точно.
- Driver последовательно выполнил: `prepare_day5.py`, `train_laya_day5.py`, `evaluate_laya_day5.py`, `evaluate_frozen_lr_day5.py`, `personal_email_review.py`, `check_day5.py`, `reload_day5.py`, `verify_day5.py`, `report_day5.py`; создан новый `run_id`, старые архивы не перезаписывались.
- Три первых model/data scripts и LR evaluator принимают `--run-id`; `training_started` и `validation_started` запрещают незаметно повторять fit/final comparison в одном run.
- Script-level OOM retry предусмотрен один: `microbatch1/accumulation16`, effective batch16, тот же набор fit и optimizer step; в фактическом запуске retries = 0.
- Фактически завершены все 300 steps примерно за 190 с вместе с dev/checks.
- Fit pool — 6000, один seeded shuffle; 300×16 предъявлений = 4800 unique; первые 3 steps — реальный gradient smoke, входят в 300.
- Dev checkpoint — 0/100/200/300; выбор только среди обученных checkpoints.
- Calibration — 1000 отдельно после выбора checkpoint; threshold по 600 internal dev.
- Manifest всех кандидатов/T/threshold сохранён ДО открытия existing validation 1275/5990.
- Выполнены только проверки/отчёты существующих артефактов: `check_day5.py`, `verify_day5.py`, `report_day5.py`; они не обучают, не открывают corpus validation/test/challenge и не делают нового inference benchmark.
- Audit читает сохранённые predictions, membership, source/candidate safetensors, fingerprints и access logs.
- Проверка reload реально выполнена training script: штатный SDK, 12 internal-dev records на candidate, logit difference/решения/T/mapping.
- Независимый `reload_day5.py` дополнительно сравнил оба сырых `false/true` logits, score и SDK T на 16 internal-dev records на candidate (4 на domain/class, фиксированный SHA-отбор, не по ошибкам); это отдельный subprocess без fit.
- Этот reload не входит в три artifact-only команды выше, поскольку выполняет небольшой реальный inference на internal dev; full validation не повторяется.
- Для собственных писем: см. `PERSONAL_EMAIL_REVIEW.md` и `personal_email_manifest_template.csv`; `data/personal_review` отдельно и gitignored; пустой шаблон → `templates_ready`.
- Сегодня никаких модельных оценок/обучения на собственных письмах не было; сначала пользователь фиксирует review/dev или отложенную диагностическую роль.
- CSV/parser command — в документе; лимит 20 MiB, статический MIME, no network.
- Для разбора и презентации передаются только `DAY5_STATUS.md`, `training.md`, `calibration.md`, `comparison.md`, `error_analysis.md` и `PRESENTATION_NOTES_DAY5.md`.
- Weights, private visible fragments и большие JSONL не передавались.
- Validation использовалась ранее; современная почта/финальная независимая оценка остаются следующими этапами.
- Default E0 с порогом `0.3037252359712881` не менялся.
