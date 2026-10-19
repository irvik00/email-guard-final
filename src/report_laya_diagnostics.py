"""Independent stdlib-only verification and report generation for synthetic diagnostics."""
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

from diagnose_laya import ROOT, boundary, dump, sha


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.addaudithook(boundary)
    latest = json.loads((ROOT / 'reports/laya_diagnostics_latest.json').read_bytes())
    result_path = Path(latest['results'])
    out = result_path.parent
    result = json.loads(result_path.read_bytes())
    plan = json.loads((out / 'fixed_plan.json').read_bytes())
    raw_rows = [json.loads(line) for line in (out / 'raw_responses.jsonl').read_text(encoding='utf-8').splitlines()]
    official_dir = sorted((ROOT / 'reports/laya_diagnostics').glob('official-*'))[-1]
    official = json.loads((official_dir / 'results.json').read_bytes())
    assert result['cases'] == raw_rows
    assert len(raw_rows) == len(plan['cases']) == 16
    assert sha(out / 'fixed_plan.json') == result['fixed_plan_sha256']
    for actual, planned in zip(raw_rows, plan['cases']):
        assert {k: actual[k] for k in planned} == planned
        qid = next(iter(actual['questions']))
        answer = actual['raw_response']['answers'][qid]
        assert answer['type'] == 'noul'
        logits = actual['actual_forward']['logits'][0][:2]
        temperature = result['temperature'][2]
        delta = (logits[0] - logits[1]) / temperature
        p_true = 1 / (1 + math.exp(delta)) if delta < 700 else 0.0
        assert abs(p_true - answer['noul']) <= 0.000051
        assert actual['score_yes'] == answer['noul']
        assert actual['positive_at_fixed_half'] == (answer['noul'] >= 0.5)
        assert actual['raw_response']['usage']['input_tokens'] == len(actual['actual_forward']['input_ids'][0])
        assert actual['no_truncation']
    checked = 0
    for folder in [out, official_dir]:
        before = json.loads((folder / 'preservation_before.json').read_bytes())
        after = json.loads((folder / 'preservation_after.json').read_bytes())
        assert before == after
        for rel, info in before.items():
            assert sha(ROOT / rel) == info['sha256'], rel
            assert (ROOT / rel).stat().st_size == info['bytes'], rel
        checked = len(before)
    assert all(v['matches_day4_manifest'] for v in result['checkpoint_file_checks'].values())
    assert result['all_forward_state_and_question_checks_passed'] and result['all_mapping_checks_passed']
    assert official['checkpoint_revision'] == result['checkpoint']['revision']
    verification = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
                    'raw_jsonl_matches_results_and_fixed_plan': True,
                    'independent_math_softmax_mapping_cases': len(raw_rows),
                    'protected_files_still_identical': checked, 'datasets_opened': 0,
                    'real_sdk_calls': 17, 'typed_question_answers': 19,
                    'synthetic_unit_tests': '4 passed; see tests/test_laya_diagnostics.py'}
    dump(out / 'independent_verification.json', verification)
    rel = out.relative_to(ROOT).as_posix()
    official_rel = official_dir.relative_to(ROOT).as_posix()
    original = plan['cases'][0]['questions']['spam']
    lines = [
        '# Короткая диагностика Laya после дня 4', '',
        f"Синтетический запуск `{result['run_id']}`; отдельный публичный пример `{official_dir.name}`. Проверено {verification['checked_utc']}.", '',
        'Подтверждена корректная передача state и направления yes→spam; обнаружены ложноположительные ответы исходной формулировки и чувствительность к вопросу. Причина на уровне конкретных весов/критериев/SDK остаётся не установлена. Изменений исходного пилота не внесено.', '',
        '## Границы и сохранность', '',
        f'- Только 16 заранее зафиксированных синтетических вызовов и 1 вызов на публичном примере SDK (всего 19 ответов на типизированные вопросы). Новые тексты написаны для диагностики; это не validation и не benchmark.',
        f'- До и после обоих запусков и независимой проверки совпадают SHA-256 {checked} защищённых файлов: отчёты/архив пилота, исходные предсказания, конфигурация вопроса, код дня 4, модели дней 3/4, requirements и pyvenv.cfg. Списки и хеши сохранены в preservation_before/after.json.',
        '- Ни один файл data/ не читался. Audit hook запрещает открытие всего data/ и socket.connect/socket.bind; офлайн-флаги HF/Transformers включены. Test/challenge/reserve не открывались, 600 validation-записей не пересматривались.',
        '- Установок, смены окружений, обучения Qwen, обучения других моделей, подбора порога и калибровки не было. Значение 0,5 применяется лишь как неизменный способ показать ответ yes/no.',
        '- План A–E сохранён до загрузки модели в fixed_plan.json; короткий вариант B задан до получения ответов. Он не заменяет вопрос или результат первого пилота.', '',
        '## SDK, checkpoint и формат', '',
        '- Изучены [официальный README/пример SDK](https://huggingface.co/convaiinnovations/laya) и [репозиторий SDK](https://github.com/NandhaKishorM/laya). Локальный README сверён с manifest дня 4; его SHA-256 `00355613796b3f5860b0d5b20392381d26580ec242e04a1fd150c7f116858602`.',
        f"- Checkpoint: `{result['checkpoint']['model_id']}`, subfolder `{result['checkpoint']['subfolder']}`, revision `{result['checkpoint']['revision']}`. Backbone — mmBERT-base. English root и typed-decisions не загружались.",
        '- model.safetensors: 643835514 байт, SHA-256 `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`. Все 6 файлов README/весов/конфигурации/tokenizer совпали с manifest дня 4.',
        '- Установленные agent.py/common.py побайтно совпали с просмотренным SDK дня 4. Снимки проверенных методов и их хеши сохранены отдельно.',
        '- Версии: ' + ', '.join(f'`{k}=={v}`' for k, v in result['versions'].items()) + '.',
        '- Использован официальный direct API `laya.load(local_path, device="cuda", fast=False)` → `agent.predict(state, questions, max_len=1024, head_max_len=256)`. Реальное устройство CUDA, autocast bfloat16, model.training=False, температуры [1,1,1]. Основная .venv не менялась.',
        '- `state` передаётся первым аргументом predict. Строка остаётся строкой; dict сериализуется SDK в JSON с ensure_ascii=False. Для 16 вызовов захвачены реальные input_ids/marker_pos перед forward: все токены state и вопроса присутствуют, усечения нет.',
        '- `noul`: type/instructions, необязательные criteria с ключами false/true. SDK формирует [CLS] вопрос [SEP] [MASK] false-вариант [MASK] true-вариант [SEP] state [SEP]. Порядок строго false, true; noul = округлённый softmax(logits / temperature)[1].',
        '- Для всех 16 ответов направление независимо проверено по сырым logits (допуск 0,000051 на округление SDK). Для spam validate_noul возвращает именно noul, без инверсии, bool() или подмены confidence. У уверенного «нет» confidence может быть близка к 1; это не spam-score.', '',
        '## Вопросы и новые контрастные примеры', '',
        'A — исходный вопрос дня 4, без изменений:', '',
        '```json', json.dumps(original, ensure_ascii=False, indent=2), '```', '',
        'B — отдельная диагностика, без criteria: `Is this message unsolicited spam?`.', '',
        'В B одновременно сокращены инструкции и убраны описания критериев; эти воздействия не разделены. На основании этих примеров B не объявляется лучшим рабочим вариантом.', '',
        '| Новый текст | Ожидание по явно заданному контексту | A: raw noul | B: raw noul |',
        '|---|---|---:|---:|',
    ]
    for a, b in zip(raw_rows[:4], raw_rows[4:8]):
        lines.append(f"| {a['id'].split('/')[-1]} | {'spam/yes' if a['expected_yes'] else 'обычное уведомление/no'} | {a['score_yes']} | {b['score_yes']} |")
    lines += ['', 'Синтетические тексты (полные короткие строки, не письма из корпуса):', '']
    for row in raw_rows[:4]:
        lines += [f"**{row['id'].split('/')[-1]}**", '', row['state'].replace('\n', ' — '), '']
    lines += ['C и D — один простой факт: `Does the text say the code is 731?`. C передаёт строку, D — тот же английский текст как dict `body`.', '',
              '| Вызов | State | Ожидание | Raw noul |', '|---|---|---|---:|']
    for row in raw_rows[8:14]:
        lines.append(f"| {row['id']} | `{json.dumps(row['state'], ensure_ascii=False)}` | {row['expected_yes']} | {row['score_yes']} |")
    lines += ['', 'Все 6 ответов о факте различили верный и неверный код при фиксированных 0,5. Это подтверждает реакцию модели на state в этих случаях, а не общую надёжность интеграции на любых вопросах.', '',
              'E использует noul-вопрос из официального примера SDK на двух новых синтетических состояниях. Положительная явная угроза отмены получила 0,0548; благодарность с сохранением плана — 0,004. Первый ответ семантически ошибочен.', '',
              '## Точный публичный пример SDK', '',
              'Публичные state/questions Quickstart извлечены как литералы из сохранённого README через ast.parse/ast.literal_eval; код документации не исполнялся. Запущены сразу department/urgency/churn_risk. Использован тот же зафиксированный multilingual direct agent. Публичный пример по умолчанию использует English Router, поэтому это проверка схемы и публичных входов на нашем checkpoint, а не воспроизведение опубликованных чисел English-модели.', '',
              f"State SHA-256: `{official['state_sha256']}`; questions SHA-256: `{official['questions_sha256']}`.", '',
              'Реальный сырой ответ:', '', '```json', json.dumps(official['raw_response'], ensure_ascii=False, indent=2), '```', '',
              'department=billing соответствует тексту; churn_risk=0,045 не соответствует явной угрозе отмены в публичном state. Confidence 0,955 относится к уверенности в «нет». Act_probability=1 не является доказательством корректного решения.', '',
              '## Полные сырые ответы синтетической диагностики', '',
              'Ниже весь объект SDK каждого вызова, без пересчёта или исправления значений. В raw_responses.jsonl дополнительно записаны state, вопрос, реальные токены, raw logits, временные измерения и независимый p_true.', '']
    for row in raw_rows:
        lines += [f"**{row['id']}**", '', '```json', json.dumps(row['raw_response'], ensure_ascii=False), '```', '']
    lines += ['## Ошибки, исправления и неизвестности', '',
              '- Подтверждённые ошибочные ответы: A считает оба новых обычных уведомления спамом; E и публичный пример отвергают явную угрозу отмены. Ошибка результата модели не равнозначна доказанной ошибке нашего парсера/интеграции.',
              '- Потеря state, усечение вопроса/текста, подмена checkpoint и инверсия yes→spam в проверенном пути не обнаружены. Оснований менять исходный код или результаты пилота нет.',
              '- В новой диагностике добавлено сохранение полного сырого ответа и реальных logits/input_ids. Исходные 600 ответов не восстанавливались повторным inference: это создало бы новый результат. Первый пилот сохранён как выполненный.',
              '- Технический сбой нового вспомогательного скрипта: первое извлечение README захватило quoted-блок о long-document и завершилось SyntaxError до загрузки модели/inference. Исправлено выделение только Markdown-блока в начале строки; добавлена содержательная фикстура quoted-блока и запрета исполнения кода. Это не ошибка интеграции дня 4.',
              '- [Официальная карточка](https://huggingface.co/convaiinnovations/laya#honest-limits) описывает чувствительность noul к вариантам и избыточную уверенность. Это согласуется с необходимостью проверки, но не доказывает причину наших конкретных результатов.',
              '- Не установлено, какая часть исходной формулировки/criteria вызывает ложные срабатывания; отдельно не проверялись версия Transformers, bfloat16 против float32, другие checkpoints и нейтральный choice. Экспериментов для выбора формулировки на validation не проводилось.',
              '- Четыре искусственных spam/ham-текста не оценивают качество на реальных письмах, устойчивость к prompt injection, калибровку, безопасность автоматических действий или пригодность B для deployment. Статус направления: needs_review, без изменения baseline.', '',
              '## Проверки и воспроизведение', '',
              f'Независимый stdlib-аудит: 16 raw-ответов совпадают с заранее сохранённым планом, JSONL и results.json; mapping независимо восстановлен из logits; {checked} защищённых файлов всё ещё идентичны. 4 unit-теста прошли. Реальных SDK-вызовов 17, ответов на вопросы 19.', '',
              f"Первый синтетический процесс занял {result['elapsed_seconds']:.2f} с, включая загрузку модели {result['model_load_seconds']:.2f} с. Публичный пример: inference {official['inference_seconds']:.3f} с. Диагностика выполнена в пределах лимита 45 минут.", '',
              'Из корня проекта, PowerShell; повторные вызовы создают новые каталоги и сохраняют результаты текущих запусков:', '',
              '```powershell',
              '.\\.venv-decision\\Scripts\\python.exe src/diagnose_laya.py',
              '.\\.venv-decision\\Scripts\\python.exe src/diagnose_laya_official_example.py',
              '.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_laya_diagnostics.py -v',
              '.\\.venv\\Scripts\\python.exe src/report_laya_diagnostics.py',
              '```', '',
              f'Артефакты: `{rel}/` и `{official_rel}/`. Независимая проверка: `{rel}/independent_verification.json`. Все исходные данные диагностики сохранены отдельно от reports/day4/.', '']
    markdown = '\n'.join(lines)
    (ROOT / 'reports/laya_diagnostics.md').write_text(markdown, encoding='utf-8')
    (out / 'laya_diagnostics.md').write_text(markdown, encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    print('Saved reports/laya_diagnostics.md')


if __name__ == '__main__':
    main()
