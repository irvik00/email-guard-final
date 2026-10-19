# Дополнительные эксперименты: воспроизведение

Из корня `D:\Study\MLSecOps\final\email_guard`, PowerShell.
Основная `.venv` и `.venv-decision` используются без обновления пакетов.
E5 feature extraction использует существующие torch/transformers в decision-env,
LR — существующий sklearn в основной `.venv`; новых numerical dependencies нет.
Отдельный end-to-end latency-процесс E5 читает sklearn stack основной `.venv`
через process-local sys.path, без установки/изменения окружений; версии совпадают
с manifest, предсказания сверяются с frozen dev logits с float32 tolerance.

## Новый полный run

```powershell
.\.venv\Scripts\python.exe -B src/run_extra_experiments.py
```

Создаёт новый run_id, протокол и архив; выполняет A → context-control → B →
C (public download, embeddings, один LR fit) → matched inner-dev control →
sender/history demo → D (два full-train fit) → общий existing validation →
reload → tests → independent audit → reports.
Каждый training script запрещает второй fit в том же run. GPU stages последовательны.
Максимум 1000 optimizer steps для A/B, smoke входит в этот бюджет, train limit
3600s на ветку; общий GPU budget 4h от frozen protocol. Внешний validation
открывается после frozen manifests; режим resume не подбирает настройки заново.

```powershell
.\.venv\Scripts\python.exe -B src/run_extra_experiments.py --resume
```

Продолжает только незавершённые стадии текущего run. Незавершённый fit с
training_started автоматически не перезапускается; техническая причина
сохраняется в driver_status/process logs, остальные ветки продолжаются.

## Явные этапы (в новом подготовленном run)

```powershell
.\.venv\Scripts\python.exe -B src/extra_common.py
.\.venv-decision\Scripts\python.exe -B src/train_extra_laya.py --branch A
.\.venv-decision\Scripts\python.exe -B src/extra_context.py
.\.venv-decision\Scripts\python.exe -B src/train_extra_laya.py --branch B
.\.venv\Scripts\python.exe -B src/setup_extra_public.py e5
.\.venv-decision\Scripts\python.exe -B src/extra_e5_embeddings.py
.\.venv\Scripts\python.exe -B src/fit_extra_e5.py
.\.venv-decision\Scripts\python.exe -B src/extra_context.py --e5-matched
.\.venv\Scripts\python.exe -B src/setup_extra_public.py psl
.\.venv\Scripts\python.exe -B src/demo_sender_evidence.py
.\.venv\Scripts\python.exe -B src/extra_metadata.py
.\.venv-decision\Scripts\python.exe -B src/evaluate_extra.py
.\.venv-decision\Scripts\python.exe -B src/extra_e5_embeddings.py --validation
.\.venv\Scripts\python.exe -B src/fit_extra_e5.py --validation
.\.venv\Scripts\python.exe -B src/extra_metadata.py --validation
.\.venv-decision\Scripts\python.exe -B src/reload_extra_laya.py
.\.venv-decision\Scripts\python.exe -B src/extra_e5_embeddings.py --reload
.\.venv\Scripts\python.exe -B src/fit_extra_e5.py --reload
.\.venv-decision\Scripts\python.exe -B src/check_extra_tokenizer.py
.\.venv\Scripts\python.exe -B src/profile_extra_lr.py
.\.venv-decision\Scripts\python.exe -B src/profile_extra_e5.py
.\.venv\Scripts\python.exe -B src/check_extra_tests.py
.\.venv-decision\Scripts\python.exe -B src/verify_extra.py
.\.venv\Scripts\python.exe -B src/report_extra.py
.\.venv\Scripts\python.exe -B src/finalize_extra.py
```

Не повторять outer validation в том же run ради настройки. Для A/B используются
первые shuffle ID дня5, далее seed42+epoch. CUDA nondeterministic backward
warn_only сохранён; новый training не обещан побитовым. Reload существующих
весов проверяется отдельно на inner dev, без fit. E5 revision pinned в
`config/extra_e5_reference.json`; источник/лицензия/все SHA — e5_source.json.
PSL snapshot версионирован; при повторе используется тот же локальный файл,
без обновления registry на основе содержимого письма.

## Проверки текущих артефактов без обучения/outer inference

```powershell
.\.venv\Scripts\python.exe -B src/check_extra_tests.py
.\.venv-decision\Scripts\python.exe -B src/reload_extra_laya.py
.\.venv-decision\Scripts\python.exe -B src/extra_e5_embeddings.py --reload
.\.venv\Scripts\python.exe -B src/fit_extra_e5.py --reload
.\.venv-decision\Scripts\python.exe -B src/check_extra_tokenizer.py
.\.venv-decision\Scripts\python.exe -B src/verify_extra.py
.\.venv\Scripts\python.exe -B src/report_extra.py
.\.venv\Scripts\python.exe -B src/finalize_extra.py
```

## Приём собственных .eml

Сначала прежний consent/label/purpose intake: `docs/PERSONAL_EMAIL_REVIEW.md`.
Новый evidence CLI — локальное наблюдение, без scores/обучения/SMTP/DNS/переходов:

```powershell
.\.venv\Scripts\python.exe -B src/sender_evidence.py data/personal_review/your_message.eml --output reports/personal_sender_evidence.json
```

Опционально `--history path/to/local_history.json --user-id local-user
--trusted-received-at 2026-10-12T12:00:00+03:00`. Путь .eml задаётся явно;
corpus data/raw/processed запрещены. History без доверенного cutoff не используется.
Feedback требует `user_label_at`, неизвестное время не считается прошлым событием.
Синтетические демонстрации находятся только в новом run/sender_demo и явно fictitious.

Никаких test/challenge/teacher/reserve/Nazario/modern/fullParquet/общего pipeline;
E0 default сохранён. Артефакты дней3–5 не перезаписываются. Большие веса/private
JSONL уже покрыты `.gitignore`; не включать их в git или публичную презентацию.
