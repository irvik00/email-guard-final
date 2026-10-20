# Дополнительные эксперименты: воспроизведение

# Что было сделано

- Использованы существующие окружения `.venv` и `.venv-decision` без обновления пакетов и без новых numerical dependencies.
- E5 feature extraction выполнен на существующих `torch/transformers` в decision-env; LR — на существующем `sklearn` в основной `.venv`.
- Выполнен новый полный run через `run_extra_experiments.py`: создан новый `run_id`, протокол и архив.
- Пройдены стадии: A → context-control → B → C (public download, embeddings, один LR fit) → matched inner-dev control → sender/history demo → D (два full-train fit) → общая existing validation → reload → tests → independent audit → reports/finalize.
- Реализован режим `--resume`: продолжение только незавершённых стадий текущего run; начатый fit автоматически не перезапускается, причина сохраняется в `driver_status/process logs`.
- Соблюдены ограничения: максимум 1000 optimizer steps для A/B, train limit 3600s на ветку, общий GPU budget 4h, последовательные GPU stages, outer validation только после frozen manifests.
- Выполнены явные этапы: `extra_common`, `train_extra_laya --branch A/B`, `extra_context`, `setup_extra_public e5/psl`, `extra_e5_embeddings`, `fit_extra_e5`, `demo_sender_evidence`, `extra_metadata`, `evaluate_extra`, `reload_extra_laya`, `check_extra_tokenizer`, `profile_extra_lr/e5`, `check_extra_tests`, `verify_extra`, `report_extra`, `finalize_extra`.
- Проверены текущие артефакты без обучения и outer inference: tests, reload Laya, E5 reload embeddings/fit, tokenizer, verify, report, finalize.
- Для собственных `.eml` использован intake по `docs/PERSONAL_EMAIL_REVIEW.md` и локальный `sender_evidence.py`: без scores/обучения/SMTP/DNS; history — только с доверенным cutoff; feedback требует `user_label_at`; synthetic demos только в `run/sender_demo` и явно fictitious.
- Обеспечена воспроизводимость: E5 revision pinned в `config/extra_e5_reference.json`, SHA/лицензия/источник в `e5_source.json`, PSL snapshot версионирован и повторно используется тот же локальный файл.
- Для A/B использованы первые shuffle ID дня 5, далее `seed42+epoch`; CUDA nondeterministic backward `warn_only` сохранён; побитовость нового training не обещалась.
- Reload существующих весов проверен отдельно на inner dev, без fit.
- Сохранены ограничения: никаких test/challenge/teacher/reserve/Nazario/modern/fullParquet/общего pipeline; E0 default сохранён; артефакты дней 3–5 не перезаписывались.
- Большие веса и private JSONL остаются в `.gitignore`; не включались в git или публичную презентацию.
