# Точные артефакты моделей

Все ссылки ниже относительны к EMAIL_GUARD_ARTIFACT_ROOT. В CPU-комплекте корень — artifacts. Загружать другие модели через API нельзя.

| Компонент | Относительный файл | Байты | SHA-256 |
|---|---|---:|---|
| E0 | `models/day3/baseline_bundle.joblib` | 2474353 | `c8acd768db0733559904f5ed2f216fcc7366b57899fc38084cdf5bdad560c6d3` |
| E0 | `models/day3/model_manifest.json` | 22476 | `c275c19d8dfd0dbb10e939256e3f2a36cedbaa5fa8022e73ecbb06364a0bed7a` |
| Laya_day5 | `models/day5/runs/20261012T083529Z-0950e3dc34/adapted_calibrated/model.safetensors` | 1287653720 | `dceaf1e38cbf53bcef1318bb0b223987b6bba1adb2d8c9fb829e8c3ff540867e` |
| Laya_day5 | `models/day5/runs/20261012T083529Z-0950e3dc34/adapted_calibrated/rl_agent_config.json` | 539 | `9cb0b6e76337016b4aa153c063e4c342927465f76889062d5a51ff20c6146fed` |
| Laya_day5 | `models/day5/runs/20261012T083529Z-0950e3dc34/adapted_calibrated/encoder/config.json` | 1938 | `83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4` |
| Laya_day5 | `models/day5/runs/20261012T083529Z-0950e3dc34/adapted_calibrated/tokenizer/tokenizer.json` | 34363188 | `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f` |
| Laya_day5 | `models/day5/runs/20261012T083529Z-0950e3dc34/adapted_calibrated/tokenizer/tokenizer_config.json` | 524 | `6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1` |
| Laya_day5 | `models/day5/runs/20261012T083529Z-0950e3dc34/candidate_manifest.json` | 7127 | `026e3f9dae3ab24161720ba28573d746705f434a7aa5cfe73b17ee214b2b16e9` |

E0 включён в artifacts/models/day3. Laya weights/tokenizer/config и candidate_manifest поставляются отдельно в email_guard_laya_artifacts/models/day5; использовать prepare_model_artifacts.py --only-laya для слияния. Проверка SHA выполняется перед копированием/загрузкой, существующие неизвестные файлы не перезаписываются.

SDK wheel vendor/laya-0.3.20-py3-none-any.whl — SHA-256 6039e802fa5effb8dd492061cd7ad39a43087beadc4a4fa4a649614e77eb83d4. Пакеты и модели уже локальные; скачивание checkpoint при inference не используется.

Источник настроек: FINAL_EVALUATION_PROTOCOL.json run20261017T132144Z-d663811e, SHA-256 9bc7e77f7b9c06f64e9660957c0498dc90928450b4322387ade42c532a523b33. E0/Laya preprocessing, вопрос B, T, thresholds и caps сохранены в app/release_manifest.json.
