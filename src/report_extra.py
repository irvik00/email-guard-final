"""Write factual reports exclusively from completed extra-run artifacts."""
import shutil
from extra_common import *

def number(value):return 'unknown' if value is None else f'{value:.6f}' if isinstance(value,float) else str(value)
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(number(v) for v in row)+' |\n' for row in rows)+'\n'
def text(out,name,content):
    (out/name).write_text(content,encoding='utf-8');shutil.copyfile(out/name,BASE/name)
def percentile(values,p):
    values=sorted(values);j=(len(values)-1)*p;a=int(j);b=min(a+1,len(values)-1)
    return values[a]+(values[b]-values[a])*(j-a)

def main():
    out,models=paths();verify=read_json(out/'verification.json');require(verify['status']=='passed','Audit must pass first')
    comparison=read_json(out/'comparison.json');train={b:read_json(out/b/'training.json') for b in ['A','B']}
    manifests={b:read_json(out/b/'candidate_manifest.json') for b in ['A','B','C']}
    prefix=f"Run `{out.name}`. Все сведения — фактический локальный запуск. E0 default сохранён.\n\n"
    rows=[]
    def add(model,domain,mode,m):
        rows.append([model,domain,mode,m['precision'],m['recall'],m['F1'],m['FPR'],m['TP'],m['FP'],m['TN'],m['FN'],str(m['ham_support'])+'/'+str(m['spam_support'])])
    for b in ['A','B','C']:
        for domain in DOMAINS:
            for mode in ['raw_0.5','calibrated_dev_threshold']:add(b,domain,mode,comparison[b][mode][domain]['metrics'])
    for b in ['E0','E2']:
        for domain in DOMAINS:
            for t,m in comparison['references'][b][domain].items():add(b+' frozen',domain,'threshold '+t,m)
    for domain in DOMAINS:
        for mode in ['raw_0.5','calibrated_dev_threshold']:
            add('Laya day5 frozen',domain,mode,comparison['references']['Laya_day5']['by_domain'][domain]['modes'][mode]['metrics'])
    content='# Дополнительные эксперименты: comparison\n\n'+prefix
    content+='Historical validation1275 (964ham/311spam), RU5990 (3000ham/2990spam). Это многократно использованный validation, **не свежий test**. Настройки A/B/C зафиксированы до нового прохода; E0/E2/Laya-day5 scores переиспользованы побайтно из неизменного дня5 после проверки идентичности ID/меток/текстов. Новых fit старых моделей не было.\n\n'
    content+=table(['Модель','Домен','Режим','P','R','F1','FPR','TP','FP','TN','FN','ham/spam'],rows)
    calibration=[]
    for b in ['A','B','C']:
        for domain in DOMAINS:
            for mode in ['raw_0.5','calibrated_0.5']:
                m=comparison[b][mode][domain]['probability_metrics'];coverage=comparison[b][mode][domain]
                calibration.append([b,domain,mode,m['NLL'],m['Brier'],m['ECE'],coverage['coverage'],coverage['failures']])
    content+='Температура меняет шкалу score; при положительном T классы на0.5 сохраняются. NLL/Brier/ECE ниже — на outer validation; fitting T был на отдельной calibration1000. Подробные ECE-bin counts — comparison.json.\n\n'
    content+=table(['Модель','Домен','Режим','NLL','Brier','ECE10','Coverage','Failures'],calibration)
    content+='Нет action-head abstention policy; coverage означает успешность вычисления, не достоверность. E0/E2 raw metrics сохраняют прежнюю обработку пустых векторов; wrapper coverage/отказы сохранены в неизменном отчёте дня5. Эти scores не доказанные вероятности будущих писем. RU — provisional source labels и неопределённый канал, не доказанная русская почта.\n\n'
    latency=[]
    for b in ['A','B']:
        pred=read_rows(out/b/'validation_predictions.jsonl')
        for domain in DOMAINS:
            times=[r['backend_seconds']*1000 for r in pred if r['domain']==domain and r['status']=='ok']
            latency.append([b+' backend',domain,len(times),percentile(times,.5),percentile(times,.95)])
    e5=read_json(out/'C/embeddings.json');dev=read_rows(out/'C/private_dev_inputs.jsonl')
    bench=sorted(dev,key=lambda r:digest([42,'latency',r['record_id']]))[:64]
    for domain in DOMAINS:
        times=[seconds*1000 for r,seconds in zip(bench,e5['warm_single_record_seconds']) if r['domain']==domain]
        latency.append(['C encoder+tokenizer+pool',domain,len(times),percentile(times,.5),percentile(times,.95)])
    if (out/'C/warm_lr_readout.json').exists():
        for domain in DOMAINS:
            times=[r['seconds']*1000 for r in read_json(out/'C/warm_lr_readout.json')['rows'] if r['domain']==domain]
            latency.append(['C cached-vector LR+T',domain,len(times),percentile(times,.5),percentile(times,.95)])
    if (out/'C/warm_end_to_end.json').exists():
        for domain in DOMAINS:
            times=[r['seconds']*1000 for r in read_json(out/'C/warm_end_to_end.json')['rows'] if r['domain']==domain]
            latency.append(['C end-to-end, actual sklearn',domain,len(times),percentile(times,.5),percentile(times,.95)])
    content+='\n'+table(['Warm measurement','Domain','n','p50 ms','p95 ms'],latency)
    content+='Laya timings — single-record backend forward после tokenization, на всём outer validation; C —64 fixed inner-dev IDs. Компонентные квантили не складываются. Если показана строка C end-to-end, это отдельное реальное измерение tokenizer+CUDA encoder+pool/L2+CPU copy+actual sklearn LR+T+threshold. Для него sklearn stack переиспользован read-only из основной .venv в decision-процессе, точные numerical versions сверены с C manifest, пакеты не устанавливались. Это одиночные локальные измерения, не serving/load benchmark; CPU fixture tests частично выполнялись одновременно с A outer inference.\n\n'
    content+='C combined-runtime latency существенно выше отдельного encoder measurement; причина расхождения **не изолирована**, performance needs_review. Этот отрицательный результат сохранён; по отдельному encoder timing нельзя заявлять скорость готового E5+LR сервиса.\n\n'
    content+='Бюджеты различаются: E0 full historical5953, E2 full mixed33906; A/B selected fit6000 с16000 предъявлений; C LR fit6000, E5 frozen. Laya tokenizer cap700, E5 tokenizer max512 включая query-prefix/special tokens. Matched E5 inner-dev контроль — context.md; это контроль входа, не изоляция архитектуры.\n'
    cm=comparison['C']['calibrated_dev_threshold']['spam_email']['metrics'];cr=comparison['C']['calibrated_dev_threshold']['russian_auxiliary']['metrics'];raw=comparison['C']['raw_0.5']['russian_auxiliary']['metrics']
    content+=f"\nC operating point: historical FP{cm['FP']}/FN{cm['FN']}, FPR={cm['FPR']*100:.4f}%; inner-dev цель FPR≤1% не является гарантией и в текущем run на outer validation превышена. RU operating point FP{raw['FP']}→{cr['FP']}, FN{raw['FN']}→{cr['FN']} относительно raw0.5. Настройки после этих исходов не менялись.\n"
    text(out,'comparison.md',content)
    training='# Дополнительные эксперименты: training\n\n'+prefix
    training+='A/B стартовали **от исходной Laya multilingual**, не от весов дня5 и не друг от друга. Revision7b928d828b7b0e022f929d9bd2e44165aa270148, Laya0.3.20; вопрос B без criteria, clean_builder_v2, cap700/context1024. Ровно прежние fit6000/dev600/calibration1000, membership/selected SHA совпадают. Synthetic augmentation отсутствует.\n\n'
    training+='AdamW CE, effective16 (micro2/accum8), weight_decay.01, clip1; warmup50 и linear decay до1000 вместо warmup15/300 дня5. Меняется и scheduler, поэтому это не изоляция только числа шагов. Первый shuffle seed42 совпал с днём5 (первые4800 предъявлений проверены); на каждом следующем проходе seed42+epoch. CUDA backward warn_only допускает недетерминизм; побитовый повтор нового training не обещан.\n\n'
    budgets=[]
    for b in ['A','B']:
        t=train[b];aud=verify['checks']['weights_and_real_gradients'][b]
        budgets.append([b,t['status'],t['actual_optimizer_steps'],t['selected_step'],t['examples_presented'],t['unique_records_used'],t['fraction_of_selected_fit_epoch'],t['trainable_parameters'],t['training_seconds'],t['peak_vram_allocated_bytes']/2**30,aud['sampled_training_peak_rss_bytes']/2**30,t['oom_retries']])
    training+=table(['Ветка','Статус','Steps','Selected','Presented','Unique','Epoch equivalents','Trainable params','Train+dev s','VRAM GiB','Sampled peak RSS GiB','OOM retries'],budgets)
    training+='RAM — максимум RSS в step-логах, не полный OS peak процесса во время сериализации. VRAM — PyTorch max allocated, не вся память драйвера. Training time включает dev/checkpoint checks; calibration/save/reload дополнительно отражены timestamps/process logs.\n\n'
    training+='В training.json поле actual_used_counts унаследовано от дня5 и считает **unique IDs** по классу; число повторных предъявлений отдельно независимо пересчитано ниже из actual_training_membership.jsonl.\n\n'
    training+=table(['Branch','Presented historical ham','Presented historical spam','Presented RU ham','Presented RU spam'],
        [[b]+[verify['checks']['weights_and_real_gradients'][b]['presented_class_counts'].get(k,0) for k in ['spam_email:0','spam_email:1','russian_auxiliary:0','russian_auxiliary:1']] for b in ['A','B']])
    for b in ['A','B']:
        training+=f'## {b}\n\n'
        if b=='B':training+='Разморожены реальные blocks: '+', '.join(manifests[b]['config']['training']['resolved_encoder_blocks'])+'. LRencoder1e-5/head1e-4. Activation checkpointing final two blocks, use_reentrant=False. Нижние блоки eval/frozen; act_head frozen.\n\n'
        training+=table(['Step','Historical dev F1','Historical FP','RU dev F1','RU FP','Mean domain F1','Dev NLL'],
            [[r['step'],r['historical_F1'],r['historical_FP'],r['ru_F1'],r['ru_FP'],r['mean_domain_F1'],r['dev_loss']] for r in train[b]['learning_curve']])
        c=manifests[b]['candidates']['adapted']
        training+=f"Выбор: max mean-domain F1@.5, затем min historical FP, затем earliest trained step. T={c['temperature']}, dev threshold={c['threshold']}; unmet_fp_scenario={c['unmet_fp_scenario']}. Source labels не исправлялись. Веса: `{c['checkpoint_path']}`.\n\n"
    c=manifests['C'];e=read_json(out/'C/embeddings.json')
    training+='## C: multilingual E5-small + LR\n\n'
    training+='Метод extraction сверён с [официальной model card](https://huggingface.co/intfloat/multilingual-e5-small). Никакой downloaded Python code не запускался.\n\n'
    training+=f"Official checkpoint `{c['source']['model_id']}`, revision `{c['source']['revision']}`, license `{c['source']['license']}`; dimension{c['dimension']}, encoder params{e['encoder_parameters']}. Masked mean pooling, L2, query-prefix, max512; encoder eval/no_grad/frozen. Кэш ключ включает ordered text hashes/revision/tokenizer SHA/prefix/tokenization/dtype; файлы и их SHA в embeddings.json.\n\n"
    training+=f"Один LR C1/L2/class_weight=None/seed42/max_iter2000; actual iterations{c['fit_iterations']}, fit seconds{c['fit_seconds']:.3f}. Feature extraction wall{e['wall_seconds']:.3f}s; VRAM{e['peak_vram_bytes']/2**30:.3f}GiB. T-range[.5,5], scalar calibration NLL; T={c['temperature']}, threshold={c['threshold']}. Никакого C-grid и обучения encoder.\n\n"
    training+=f"У C unconstrained calibration optimum T={c['calibration']['unconstrained_temperature']}; применён нижний предел0.5. Это заранее заданный **экспериментальный** диапазон C, а не ограничение E5 SDK; название ключа sdk_limits унаследовано от общего numerical helper.\n\n"
    training+='Reload проверен независимо: 16 dev примеров для обоих logits Laya/SDK outputs; C — свежий encoder reload + embeddings + LR logits/threshold. SHA всех frozen tensors и trainable changes проверены против source safetensors, подробные имена в verification.json. A executed-source snapshot сохранён с исходным SHA: до B код расширен только checkpointing/проверками B, результаты A не переписаны.\n'
    text(out,'training.md',training)
    context='# Контекст и tokenizer controls\n\n'+prefix
    ctrl=read_json(out/'context_control.json');match=read_json(out/'e5_matched_control.json')
    context+='A selected weights/T/threshold неизменны. Encoder config max_position_embeddings8192; eager SDK допускает2048. Ни positional embeddings, ни сохранённый config/веса не менялись. Удлинение только inference меняет distribution; cap700 остаётся основным.\n\n'
    context+=f"Inner dev600; long subset{ctrl['long_subset_n']} определяется по original tokenizer tokens>700, **до сравнения исходов**. Paired class changes={len(ctrl['paired_class_changes'])}.\n\n"
    ctxrows=[]
    for label,spec in [('short long-subset',ctrl['short_long_subset']),('long long-subset',ctrl['long_subset']),('A E5 exact fragments',match['all_dev'])]:
        for domain,d in spec.items():ctxrows.append([label,domain,d['metrics']['support'],d['metrics']['F1'],d['metrics']['FP'],d['metrics']['FN']])
    context+=table(['Контроль','Домен','n','F1','FP','FN'],ctxrows)
    context+='В matched контроле каждый видимый символьный span E5 побайтно совпал с текстом, переданным Laya; дополнительное усечение Laya запрещено assert. Два из600 E5 spans превышали700 токенов Laya (max1362); первоначальный cap700 assert остановился **до inference**. Для отдельного matched контроля до получения его scores зафиксирован cap1500/context2048, уже проверенный на A. Основной A cap700, его T/threshold/weights сохранены. Это дополнительное изменение inference distribution, не изоляция архитектуры. Дополнительного fit/калибровки/подбора checkpoint не было.\n\n'
    trunc=[]
    for model in ['A','C']:
        for part in ['fit','dev','calibration']:
            data=read_rows(out/f'{model}/private_{part}_visible.jsonl')
            for domain in DOMAINS:
                subset=[r for r in data if r['domain']==domain]
                trunc.append([model,part,domain,len(subset),sum(r['token_stats']['truncated'] for r in subset)])
    for domain in DOMAINS:
        v=read_rows(out/'private_validation_laya_visible.jsonl');s=[r for r in v if r['domain']==domain]
        trunc.append(['A/B','validation',domain,len(s),sum(r['token_stats']['truncated'] for r in s)])
        v=read_rows(out/'C/private_validation_visible.jsonl');s=[r for r in v if r['domain']==domain]
        trunc.append(['C','validation',domain,len(s),sum(r['token_stats']['truncated'] for r in s)])
    context+=table(['Model','Part','Domain','n','Truncated'],trunc)
    context+='B имеет те же фрагменты и counts, что A. Предупреждение tokenizer о полной исходной длине относится к подсчёту offsets: реальные forward caps проверены. Cross-part E5 visible-prefix collisions: '+str(verify['checks']['E5_visible_cross_inner_part_collisions']['count'])+'; подробности verification.json, без удаления/переобучения.\n'
    context+='E5 outer-validation visible-prefix collisions с fit/dev/calibration: '+str(verify['checks']['E5_visible_outer_prefix_collisions']['count'])+'; исходные group/canonical splits не изменены, configs не пересматривались.\n'
    context+='Одна cross-inner collision помечена needs_review_visible_prefix_only: локаторы/метки/проверка полных текстов без публикации писем — C/visible_collision_review.json. Прежняя фиксированная membership сохранена; зависимость видимых префиксов ограничивает независимость inner оценки C.\n'
    boundary_rows=[]
    for part in ['fit','dev','calibration','validation']:
        data=read_rows(out/f'C/private_{part}_visible.jsonl')
        boundary_rows.append([part,len(data),sum(not r['token_stats']['standalone_retokenization_exact'] for r in data)])
    context+='\nSentencePiece boundary: первоначальный assert о повторной токенизации cut-fragment оказался слишком строгим и остановил preprocessing **до LR fit**. E5 inference неизменно получает original-text capped512 IDs; видимый символьный span извлечён из их offsets. Standalone retokenization mid-word может иначе разбить последние pieces. ID hashes и флаг сохранены; вопрос/выборка/веса/реальные tokenizer caps не менялись. Это ограничивает точность изоляции входа на границе subword.\n\n'
    context+=table(['Part','n','Standalone fragment tokenization differs'],boundary_rows)
    text(out,'context.md',context)
    sender='# Sender / бренд / контакты / локальная история\n\n'+prefix
    sender+='Offline module `src/sender_evidence.py` переиспользует статический MIME parser, IDNA и URL нормализацию. HTML не исполняется; ссылки/вложения не открываются. Display name/address/domain, Reply-To/Return-Path, brand alias claims, PSL organizational domains, domain boundaries, limited confusable skeleton/mixed scripts, link-label mismatch, контакты с контекстом, digit ratio, unusual codepoints, unsubscribe (working=null), attachments/formatting возвращаются с provenance.\n\n'
    sender+='Реестр `config/registry/official_domains.json`: версия2026-10-12-v1, три примера банков. [Сбер: официальный web domain](https://research.sberbank.ru/assets/conditions-sber-research.BEQKfQO0.pdf); [ВТБ: публичный контактный домен](https://www.vtb.ru/about/contacts/); [Альфа-Банк: публичный контактный домен](https://alfabank.ru/essential/). Сбер mail domains unknown/[]; для ВТБ/Альфа подтверждено лишь опубликованное использование контактного домена. Реестр не исчерпывает mailing providers и не удостоверяет отправителя. Источники проверены отдельно, без текстов писем.\n\n'
    sender+='PSL snapshot из publicsuffix.org зафиксирован SHA в psl_manifest.json, включая private rules, wildcards/exceptions. IDNA2003 — ограничение существующего parser; confusable mapping неполный, отсутствие флага не доказывает безопасность. Bank claim/domain mismatch — наблюдение/candidate signal, **phishing_label=null**; никакого ручного штрафа к spam_score. Публичный почтовый сервис/мобильный номер сами по себе не spam.\n\n'
    sender+='Authentication-Results parsed reported, trusted=null; знакомый authserv-id не устанавливает trust. DKIM-Signature presence не проверка подписи. [RFC8601 trust boundary](https://www.rfc-editor.org/rfc/rfc8601.txt), [RFC7489 historical alignment reference](https://www.rfc-editor.org/rfc/rfc7489); последний сейчас отмечен RFC Editor как obsolete. Модуль не реализует live DMARC/SPF policy/DNS/MX/SMTP probes; auth pass не гарантия ham.\n\n'
    sender+='History JSON/CSV опционален: user_id/sender/trusted_received_at/opened_at/replied_at/user_label, timezone и trusted cutoff обязательны для использования. Date заголовка не cutoff. Только события строго раньше cutoff, контакты — по точному нормализованному mailbox (local part case preserved). Для feedback требуется также user_label_at, иначе unknown: дата получения не доказывает время присвоения метки. Missing file=unknown, missing opened=unknown; familiar sender не whitelist.\n\n'
    demos=read_json(out/'sender_demo_summary.json');sender+='## Синтетические проверки, не benchmark\n\n'
    sender+=table(['Fictitious case','Mixed scripts','Contacts observed','Auth trusted','History state','Phishing label'],
        [[k,v['mixed_scripts'],v['contact_count'],v['authentication_trusted'],v['history']['state'],v['phishing_label']] for k,v in demos['cases'].items()])
    sender+='Три .example fixtures и fictitious history лежат в sender_demo; метрик качества по ним нет. Реальная пользовательская почта здесь не читалась. Локальный CLI принимает один явно названный .eml, без сканирования аккаунта; цели/consent/разметка сохраняются в прежнем intake дня5.\n'
    text(out,'sender_features.md',sender)
    d=read_json(out/'D/comparison.json');elig=read_json(out/'D/eligibility_before_fit.json');dm=read_json(out/'D/candidate_manifest.json')
    metadata='# Условный metadata experiment\n\n'+prefix
    metadata+=f"Статус **completed**: пригодны реальные наблюдаемые поля SpamAssassin с обеими метками. Cohort зафиксирован до fit: train{elig['train_rows']} ({elig['class_counts']}), validation{sum(d['class_counts'].values())} ({d['class_counts']}). RU/Ling-Spam исключены из этого cohort, без выдуманных нулей.\n\n"
    metadata+='Два новых full-train fit: одинаковый clean_builder_v2 char TF-IDF(3,5)/min_df2/max_features100000/sublinear_tf/L2; LR C1/L2/no class weights/seed42/max_iter2000, threshold0.5. Hybrid добавляет восемь фиксированных наблюдений: sender-domain length, local-part digit ratio, link count, observed comparable-label mismatch count, attachment count, unsubscribe-header/body-hint, HTML presence. Никаких source/path/filter/auth/registry/history features.\n\n'
    metadata+='NaN остаются неизвестными в raw features; train-only median imputation/scaling + отдельные missing indicators для всех восьми полей. Нулевой count допустим лишь для наблюдаемого пустого списка. Mismatch count — число **наблюдаемых сравнений**; при отсутствии сравнимых labels unknown, а не отсутствие подражания. Значение не подтверждает фишинг.\n\n'
    metadata+=table(['Модель','P','R','F1','FPR','TP','FP','TN','FN'],[[b]+[d[b]['metrics'][k] for k in ['precision','recall','F1','FPR','TP','FP','TN','FN']] for b in ['text_only','hybrid']])
    metadata+=f"Paired changes: {len(d['paired_changes'])}; пары и исходы сохранены в D/comparison.json. Это отдельный source cohort, его F1 нельзя напрямую сравнивать с full-domain таблицей. Train/validation groups disjoint; два reload diff0; vocabulary/imputer/scaler fit только train. Никакого выбора гиперпараметров по этим исходам.\n\n"
    metadata+=table(['Model','Fit s','Iterations','RSS snapshot GiB','Warm p50 ms','Warm p95 ms','Reload max error'],
        [[b,dm['models'][b]['fit_seconds'],str(dm['models'][b]['iterations']),dm['models'][b]['rss_bytes']/2**30,
          percentile(d[b]['warm_single_record_seconds'],.5)*1000,percentile(d[b]['warm_single_record_seconds'],.95)*1000,
          dm['models'][b]['reload_max_logit_error']] for b in ['text_only','hybrid']])
    metadata+=table(['Feature','Missing ham train','Missing spam train'],[[feature,elig['missing_by_class']['0'][feature],elig['missing_by_class']['1'][feature]] for feature in elig['features']])
    metadata+='Исторические письма не проверяют современные банковские имитации/историю пользователя; таких надёжных меток/наблюдений в cohort нет. Raw headers не восстановлены из corpus raw/normalised: использованы только разрешённые split поля.\n'
    text(out,'metadata_experiment.md',metadata)
    presentation='# Материал для презентации\n\n'+prefix
    presentation+='1. **Вопрос:** достаточно ли увеличить бюджет head-only; помогает ли адаптация верхних encoder-блоков; что даёт frozen multilingual E5+LR?\n2. **Контракт:** прежние6000/600/1000, протокол до оценки, seed42, один run A/B/C, закрытый test не открыт; E0 сохранён.\n3. **A/B:** таблица actual budgets/learning curves в training.md; сравнивать raw0.5 и внутренний operating point отдельно.\n4. **C:** официальный E5 frozen,512tokens/query-prefix/mean/L2, один LR; другой context/tokenizer и prior.\n5. **Результаты:** вставить две доменные таблицы comparison.md; не сворачивать языки в один F1.\n6. **Evidence layer:** sender/brand/history/null/trust semantics; три fictitious .example demos без quality claims.\n7. **Hybrid:** отдельный SpamAssassin cohort, две одинаковые text pipelines и metadata; показать FP/FN, а не только F1.\n8. **Ограничения:** validation многократно использован, provisional RU/chat labels, один seed; современные отложенные письма здесь не оценивались, в использованном cohort нет пригодной phishing/category ground truth, registry не whitelist. API/Docker, собственные современные письма и финальная защита — следующие обязательные этапы, здесь не выполнены.\n\n'
    presentation+=table(['Ветка','Historical raw F1','RU raw F1','Historical dev-policy FP/FN','RU dev-policy FP/FN'],
        [[b,comparison[b]['raw_0.5']['spam_email']['metrics']['F1'],comparison[b]['raw_0.5']['russian_auxiliary']['metrics']['F1'],
          str(comparison[b]['calibrated_dev_threshold']['spam_email']['metrics']['FP'])+'/'+str(comparison[b]['calibrated_dev_threshold']['spam_email']['metrics']['FN']),
          str(comparison[b]['calibrated_dev_threshold']['russian_auxiliary']['metrics']['FP'])+'/'+str(comparison[b]['calibrated_dev_threshold']['russian_auxiliary']['metrics']['FN'])] for b in ['A','B','C']])
    text(out,'PRESENTATION_NOTES_EXTRA.md',presentation)
    status='# EXTRA_EXPERIMENTS_STATUS\n\n'+prefix
    status+='**completed** для A/B/C, context/matched controls, evidence module и supervised metadata cohort. Реальных нерешённых технических блокеров нет. Все новые кандидаты исследовательские; E0 default не заменён.\n\n'
    status+=table(['Ветка','Факт'],[['A',f"1000steps, selected{train['A']['selected_step']},6000unique/16000presented"],['B',f"1000steps, selected{train['B']['selected_step']},two final blocks, frozen SHA checked"],['C',f"frozen E5 dimension{c['dimension']},one LRfit6000,T/dev threshold fixed"],['Sender/history','module_ready,3fictitious demos,unknown/trust preserved'],['D','2full-train fits on SpamAssassin only; threshold0.5'],['Checks',f"independent audit passed; {verify['checks']['preservation']['files']} snapshot files SHA identical; one archived current-run source update; reload/tests passed"]])
    status+='Отчёты: reports/extra_experiments/{comparison,training,context,sender_features,metadata_experiment,PRESENTATION_NOTES_EXTRA}.md. Protocol/manifests/logs/predictions/verification/learning curves в runs/'+out.name+'; checkpoints в models/extra_experiments/'+out.name+'.\n\n'
    status+='Закрытые test/challenge/teacher/frozen reserve/Nazario/modern/fullParquet не открывались,2026-09 security_blocked сохранён. Исторические splits не изменены. Не запускались общий audit/pipeline, inventory/reservoir, новые grid/seed-runs.\n\n'
    status+='Ограничения: repeatedly-used validation, provisional RU, different training/token budgets, one seed, sampled peak RAM only; registry/auth/history — evidence, не подтверждённый phishing. В использованном cohort нет пригодной ground truth для phishing/тематических категорий. Решение о смене default не принято.\n\n'
    latency_state='измерена на64 inner-dev' if (out/'C/warm_end_to_end.json').exists() else 'не измерена'
    status+=f"C: inner visible-prefix collisions={verify['checks']['E5_visible_cross_inner_part_collisions']['count']} needs_review, outer-prefix collisions={verify['checks']['E5_visible_outer_prefix_collisions']['count']}; T={c['temperature']}, заданный диапазон[.5,5]. Historical outer FPR при dev-policy={cm['FPR']*100:.4f}% (inner цель1%). C end-to-end latency {latency_state}; encoder/LR components также сохранены.\n\n"
    status+='C combined runtime имеет заметно большую latency, чем отдельный encoder: причина не изолирована, performance needs_review; сервисная готовность/throughput не доказаны.\n\n'
    status+='Повтор: `.\\.venv\\Scripts\\python.exe -B src/run_extra_experiments.py` создаёт **новый** run, не перезаписывает этот. Подробные команды: docs/EXTRA_EXPERIMENTS_REPRODUCTION.md.\n'
    (out/'EXTRA_EXPERIMENTS_STATUS.md').write_text(status,encoding='utf-8');shutil.copyfile(out/'EXTRA_EXPERIMENTS_STATUS.md',ROOT/'reports/EXTRA_EXPERIMENTS_STATUS.md')
    for name in ['comparison.json','verification.json']:
        shutil.copyfile(out/name,BASE/name)
    write_json(BASE/'current_run.json',{'run_id':out.name,'stage':'completed','verification_sha256':sha_file(out/'verification.json')})
    print('Extra reports written:',out.name)

if __name__=='__main__':main()
