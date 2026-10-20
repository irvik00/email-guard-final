# Локальная Laya и показ преподавательского теста

## Публикация дополнения

email_guard_defense_update.zip включает предыдущий UI-патч. Его можно применить к исходной папке публикации email_guard_render либо поверх UI-патча. Веса, зависимости, пороги, протокол и финальные ML-метрики не меняются. Обновление не предназначено для исходного исследовательского проекта.

Из папки публикации, где находится Dockerfile:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\email_guard_defense_update.zip" -DestinationPath (Get-Location).Path -Force
py -3.12 .\scripts\verify_render_package.py
git add .
git commit -m "Add archived teacher evaluation page"
git push
```

После Live открыть https://email-guard-final.onrender.com/evaluation. На главной странице появилась ссылка «Тест преподавателя». На странице шесть строк: E0/Laya/OR на полном teacher1600 и на 612 без найденных train/validation совпадений. Показаны покрытие, отказы, F1, FP/FN/FPR, источник и 988 overlap. Это сохранённый отчёт, не новый запуск Laya в облаке. Сырые письма/teacher archive/индивидуальные предсказания не публикуются. JSON-агрегаты доступны по /evaluation-data.json.

Если дедлайн близко, сначала отправить имеющиеся ссылки/презентацию по правилам LMS, затем дождаться сборки и проверить URL; не откладывать отправку ради дополнительных экспериментов.

## Локальная Laya с новым интерфейсом

Оставаться в папке публикации. Рабочие веса и окружения берутся из исходного проекта, который не перезаписывается. Перед запуском проверить три пути:

```powershell
$researchRoot = "D:\Study\MLSecOps\final\email_guard"
Test-Path "$researchRoot\artifacts\models\day5\runs\20261012T083529Z-0950e3dc34\adapted_calibrated\model.safetensors"
Test-Path "$researchRoot\.venv-decision\Scripts\python.exe"
Test-Path "$researchRoot\.venv-app\Scripts\python.exe"
```

Все True. Затем:

```powershell
Remove-Item Env:EMAIL_GUARD_DISABLE_LAYA -ErrorAction SilentlyContinue
$env:EMAIL_GUARD_ARTIFACT_ROOT = (Resolve-Path "$researchRoot\artifacts").Path
$env:EMAIL_GUARD_LAYA_PYTHON = (Resolve-Path "$researchRoot\.venv-decision\Scripts\python.exe").Path
$env:EMAIL_GUARD_HOST = "127.0.0.1"
& "$researchRoot\.venv-app\Scripts\python.exe" -m app.run --port 8001
```

Не добавлять --disable-laya. Если 8001 занят, остановить собственный прежний процесс Ctrl+C либо выбрать другой свободный порт и использовать его во всех командах. Окно с сервером оставить открытым. В другом PowerShell:

```powershell
Invoke-RestMethod "http://127.0.0.1:8001/ready?mode=review" | ConvertTo-Json -Depth 6
```

Дождаться HTTP200 и modes baseline/recall/review=true, components.Laya_day5.status=ready. Загрузка может занимать до 180 секунд. Во время загрузки 503 для review ожидаем, после ожидания остающийся 503 означает недоступность, а не готовность.

Открыть http://127.0.0.1:8001/. Новый UI активирует три режима только при реальной готовности. Для показа выбрать review, загрузить examples/app/demo_05.eml: ожидается разногласие/needs_review и отдельные настоящие оценки E0/Laya. Demo_06 показывает усечение входа Laya. Не менять пороги ради приятного ответа.

Для полной функциональной проверки локального CUDA сервиса, из папки публикации:

```powershell
& "$researchRoot\.venv-app\Scripts\python.exe" scripts\smoke_release.py --url http://127.0.0.1:8001 --label defense-laya --with-laya
```

Это проверка настоящих HTTP-ответов на вымышленных примерах, не новая оценка качества. Здесь, в среде подготовки патча, GPU/веса Laya отсутствуют, поэтому новый CUDA запуск не выполнялся. Команды используют существовавший проверенный локальный runtime.

## Категории

Сейчас topic hints основаны на ручных RU/EN правилах. Для ML-категорий нужны отдельные тематические метки, данные по каждой категории, train/validation/test и оценка macro-F1/ошибок по классам. Бинарных spam/ham-меток и пяти иллюстративных примеров недостаточно для достоверно проверенного категоризатора. Преподавательский тест не использовать для обучения или выбора категорий/порогов.
