import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from day5_common import (ROOT,Boundary,group_and_partition,audit_parts,fit_temperature,sigmoid,
                         probability_metrics,choose_threshold,clean_text,digest)
from personal_email_review import resolve_entries,parse_static,inspect_entries,FIELDS,MAX_BYTES


class Day5Tests(unittest.TestCase):
    def record(self,i,text,group=None,domain='spam_email',label=0):
        return {'record_id':str(i),'duplicate_group_id':group or str(i),'domain':domain,'source':domain,
                'spam_label':label,'text':text,'original_model_text':text}

    def test_transitive_groups_across_domains_and_normalization(self):
        rows=[self.record(0,'One',group='shared'),self.record(1,'Two',group='shared',domain='russian_auxiliary'),
              self.record(2,' TWO ',domain='spam_email')]
        rows += [self.record(i,'Unique record '+str(i),label=i%2) for i in range(3,50)]
        parts,conflicts=group_and_partition(rows)
        self.assertEqual(len({r['inner_part'] for r in rows[:3]}),1)
        self.assertTrue(audit_parts(parts)['passed'])
        self.assertEqual(conflicts,[])

    def test_leakage_audit_rejects_cross_role_copy(self):
        a=self.record(1,'A');a['inner_group_id']='a'
        b=self.record(2,'A');b['inner_group_id']='b'
        with self.assertRaises(ValueError):audit_parts({'fit':[a],'dev':[b],'calibration':[]})

    def test_boundary_no_validation_during_training_or_closed_during_final(self):
        train=Boundary('train');final=Boundary('final')
        train.check('open',(str(ROOT/'data/processed/splits/manifest.json'),'r',0))
        with self.assertRaises(PermissionError):train.check('open',(str(ROOT/'data/processed/splits/spam_email/validation.jsonl'),'r',0))
        for phase in [train,final]:
            with self.assertRaises(PermissionError):phase.check('open',(str(ROOT/'data/processed/splits/spam_email/test.jsonl'),'r',0))
            with self.assertRaises(PermissionError):phase.check('socket.connect',())
        with self.assertRaises(PermissionError):final.check('open',(str(ROOT/'data/processed/splits/spam_email/train.jsonl'),'w',0))

    def test_temperature_limits_and_same_half_decisions(self):
        labels=[0,1,0,1];logits=[-3.,3.,-5.,5.]
        result=fit_temperature(labels,logits)
        self.assertTrue(result['optimum_outside_sdk_range'])
        self.assertAlmostEqual(result['applied_temperature'],.5)
        self.assertEqual([sigmoid(x)>=.5 for x in logits],[sigmoid(x/result['applied_temperature'])>=.5 for x in logits])
        self.assertEqual(sum(b['count'] for b in probability_metrics(labels,logits)['ECE_bins']),4)

    def test_threshold_cannot_choose_all_negative(self):
        rows=[{'domain':domain,'spam_label':y,'logit_difference':2. if y==0 else -2.}
              for domain in ['spam_email','russian_auxiliary'] for y in [0,1]]
        selected=choose_threshold(rows,1.)
        self.assertEqual(selected['threshold'],.5)
        self.assertTrue(selected['unmet_fp_scenario'])

    def test_dependency_free_builder_matches_day4(self):
        from day4_common import clean_text as original
        cfg=json.loads((ROOT/'config/day4_experiments.json').read_bytes())
        for availability in ['observable','unavailable']:
            record={'subject':'(SPAM? 4.5) appointment','body_text':'Body\nX-Spam-Score: 9\nreal body',
                    'feature_availability':{'subject':availability}}
            self.assertEqual(clean_text(record,cfg),original(record,cfg))

    def test_personal_manifest_paths_and_unknown_are_explicit(self):
        row=dict.fromkeys(FIELDS,'');row.update(local_path='data/personal_review/a.eml',spam_label='unknown',consent='unknown',review_status='pending')
        self.assertEqual(resolve_entries([row])[0][1]['spam_label'],'unknown')
        for value in ['false','true','', '2']:
            with self.assertRaises(ValueError):resolve_entries([dict(row,spam_label=value)])
        with self.assertRaises(ValueError):resolve_entries([dict(row,local_path='data/personal_review/../../processed/test.eml')])
        with self.assertRaises(ValueError):resolve_entries([dict(row,spam_label='1',label_reason='')])

    def test_static_personal_parser_html_attachments_and_duplicates(self):
        raw=(b'Subject: Synthetic fixture\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=z\r\n\r\n'
             b'--z\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>Visible notice</p><script>SECRET_SCRIPT</script>\r\n'
             b'--z\r\nContent-Type: text/plain\r\nContent-Disposition: attachment; filename="do_not_run.txt"\r\n\r\nSECRET_ATTACHMENT\r\n--z--\r\n')
        # Only locally generated fixtures; no personal or corpus content.
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as directory:
            path=Path(directory)/'fixture.eml';path.write_bytes(raw)
            metadata=parse_static(path)
            self.assertTrue(metadata['usable_text']);self.assertEqual(metadata['attachment_count'],1)
            self.assertEqual(metadata['model_text_hash'],digest('Synthetic fixture\n\nVisible notice'))
            result=inspect_entries([(path,{'spam_label':'0','review_status':'reviewed','consent':'yes'}),
                                    (path,{'spam_label':'1','review_status':'needs_review','consent':'unknown'})])
            self.assertEqual(result['duplicate_groups'],1)
            self.assertEqual(len(result['conflicting_label_groups']),1)
            self.assertFalse(result['model_inference_performed'])
            self.assertNotIn('body_text',result['metadata_only'][0])

    def test_personal_size_limit_before_mime_parse(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as directory:
            path=Path(directory)/'oversize.eml'
            with path.open('wb') as handle:handle.truncate(MAX_BYTES+1)
            with self.assertRaisesRegex(ValueError,'20 MiB'):parse_static(path)


if __name__=='__main__':unittest.main()
