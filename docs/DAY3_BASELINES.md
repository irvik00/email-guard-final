# Email Guard — день 3: базовые модели и анализ ошибок

# Что было сделано

- Изучены существующие инструкции, код, отчёты дня 2, схема, конфигурация и manifest подготовки; pipeline дня 2 повторно не запускался.
- Для baseline использованы только `data/processed/splits/spam_email/train.jsonl` и `validation.jsonl`; проверены ожидаемые численности: train 5953 (ham4502/spam1451), validation 1275 (ham964/spam311).
- Закрытые данные не открывались: `spam_email/test.jsonl`, русские splits, modern challenges, frozen reserve, modern train candidates, Nazario/phishing-positive-test, тест преподавателя, needs_review Untroubled; `security_blocked2026-09` не скачивался.
- Реализованы CLI-модули для экспериментов и предсказания, конфигурация и переиспользование существующих утилит; данные читались потоково, проверены `record_id`, `duplicate_group_id`, метки 0/1, непустой `model_text`, отсутствие пересечений train/validation.
- Зафиксированы отпечатки train/validation/config/code; source, ID, путь, label_status и filter headers не добавлялись в признаки.
- Использован установленный scikit-learn в `.venv`; записаны точные версии scikit-learn/numpy/scipy/joblib.
- Реализованы три Pipeline с одинаковым входом `model_text`:
  - `word_nb`: TF-IDF слова/пары слов (1,2) + MultinomialNB(alpha=1.0);
  - `word_lr`: тот же TF-IDF + LogisticRegression(C=1.0);
  - `char_lr`: TF-IDF `char_wb` (3,5) + LogisticRegression(C=1.0).
- TF-IDF и классификатор объединены в sklearn Pipeline; fit словаря, IDF и модели выполнялся только на train-части каждого фолда, transform — на validation-части.
- Использован единый сохранённый mapping 5 train-CV фолдов из manifest; проверены охват, группы и оба класса в каждом фолде.
- Проведена 5-fold CV для каждого Pipeline; сохранены OOF-оценки с ID/группой/источником; каждая train-запись получила одну OOF-оценку от модели, её не обучавшей.
- Порог сравнения моделей заранее зафиксирован 0.5; baseline выбран по максимальному среднему binary F1 по фолдам для spam=1, с tie-break по precision, затем word_lr, char_lr, word_nb.
- После CV каждый вариант обучен на полном train и предсказан validation; сохранены binary precision/recall/F1 (positive=spam), accuracy, TP/FP/TN/FN, FPR, support классов, а также source-разрез для SpamAssassin и Ling-Spam.
- Добавлен `DummyClassifier(strategy=most_frequent)` как контроль на validation без CV-поиска.
- Для выбранного baseline сравнены три режима порога: 0.5; порог с максимальным validation F1; максимальный validation F1 при observed FPR ≤1%.
- Сохранены отдельно `threshold_default`, `threshold_f1`, `threshold_conservative` и выбранная политика; порог подбирался только на validation у заранее выбранной модели.
- Для recall и FPR выбранного baseline добавлены 95% Wilson interval и исходные числители/знаменатели для каждого режима.
- Выполнен анализ ошибок только на validation выбранного baseline: до 5 FP и 5 FN при provisional threshold, несколько правильных примеров; сохранены ID, source, метки, score, порог, короткий обезличенный фрагмент и наблюдение.
- Для LR сохранены глобальные признаки с положительными/отрицательными весами; локальное объяснение проверено как вклад `x_jw_j` + intercept, сумма сверена с `decision_function`.
- Сохранены три fit-на-train Pipeline и выбранный baseline в `models/day3/` с воспроизводимым manifest; bundle содержит threshold config, classes, версии, параметры `model_text` builder, хеши входов/config и признаки.
- Реализован CLI для одного `.eml`: существующий парсер → тот же `model_text` builder → transform/predict без fit → `spam_score` и binary verdict при заданном пороге.
- Проверены save/reload: оценки на небольшой validation-подвыборке совпадают до/после сохранения в допуске; CLI проверен на искусственном `.eml` из учебных examples/fixtures.
- Добавлены содержательные тесты на искусственных данных: validation-only слово не появляется в train-словаре; фолды не пересекают группы; confusion/FPR для известного случая; направление positive class и threshold; локальное объяснение сходится с логарифмической оценкой; bundle после загрузки даёт те же оценки; CLI использует тот же builder текста и не делает fit при предсказании; проверен запрет закрытых путей в runner.
- Запущены новые тесты и подходящие существующие тесты на фикстурах, затем реальные 15 CV fit для трёх моделей, 3 fit на полном train, validation, выбор порога, отчёты, reload и CLI smoke.
- Подготовлены артефакты: `src/train_baselines.py`, `src/predict_email.py`, конфигурация, актуальные requirements, `docs/DAY3_EXPERIMENTS.md`, `models/day3/` (три Pipeline, baseline bundle, `model_manifest.json`), `reports/day3/experiments.md/.json`, `thresholds.md/.json`, `error_analysis.md`, `predictions/` (OOF и validation компактно), `reports/DAY3_STATUS.md`.
- Исходники и splits не изменялись; сырые письма и приватные отчёты не публиковались; артефакты моделей и предсказаний исключены из git по существующей политике; ничего не пушилось.
- Для разбора пользователю предназначены `reports/DAY3_STATUS.md` и три Markdown-отчёта: `experiments.md`, `thresholds.md`, `error_analysis.md`.
