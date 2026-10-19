"""Meaningful app-boundary checks. All test stubs are confined to this file."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.contract import compose, MAX_BYTES, unavailable
from app.inference import Engine, E0, parse_upload, build_texts, InputError
from app.web import create_app
from fastapi.testclient import TestClient


def part(name, decision):
    return {'model_id':name, 'status':'ok', 'decision':decision, 'score':0.9 if decision else 0.1}


class VotingTests(unittest.TestCase):
    def test_all_binary_votes_and_review_disagreement(self):
        for a in (0,1):
            for b in (0,1):
                p={'E0':part('E0',a),'Laya_day5':part('Laya_day5',b)}
                self.assertEqual(compose('baseline',p)['decision'],a)
                self.assertEqual(compose('recall',p)['decision'],int(a or b))
                review=compose('review',p)
                self.assertEqual(review['decision'],a if a==b else None)
                self.assertEqual(review['status'],'ok' if a==b else 'needs_review')

    def test_either_component_failure_never_ham_even_when_other_says_spam(self):
        for name in ('E0','Laya_day5'):
            for status in ('error','unavailable','insufficient_features','busy','initializing'):
                p={'E0':part('E0',1),'Laya_day5':part('Laya_day5',1)}
                p[name]=unavailable(name,'test_failure',status)
                self.assertIsNone(compose('recall',p)['decision'])
                self.assertEqual(compose('recall',p)['status'],'component_unavailable')
                self.assertEqual(compose('review',p)['status'],'needs_review')
                self.assertIn(name,compose('review',p)['unavailable_components'])

    def test_boolean_or_missing_prediction_not_a_binary_integer(self):
        for bad in (None, {'status':'ok','decision':False}, {'status':'ok','decision':None}):
            self.assertIsNone(compose('recall',{'E0':part('E0',0),'Laya_day5':bad})['decision'])

    def test_engine_baseline_does_not_call_missing_neural(self):
        engine=Engine.__new__(Engine);engine.e0=Mock();engine.e0.predict.return_value=part('E0',0)
        engine.laya=Mock();engine.laya.predict.side_effect=AssertionError('not required')
        self.assertEqual(engine.predict_texts('body','body','baseline')['decision'],0)
        engine.laya.predict.assert_not_called()

    def test_engine_component_exception_is_explicit(self):
        engine=Engine.__new__(Engine);engine.e0=Mock();engine.e0.predict.side_effect=RuntimeError('private data')
        engine.laya=Mock();engine.laya.predict.return_value=part('Laya_day5',0)
        result=engine.predict_texts('body','body','review')
        self.assertEqual(result['status'],'needs_review');self.assertIsNone(result['decision'])
        self.assertTrue(result['infrastructure_failure'])
        self.assertNotIn('private data',json.dumps(result))


class ParserTests(unittest.TestCase):
    def test_empty_binary_invalid_and_recoverable_warning(self):
        for raw in (b'',b'  \r\n',b'\x00\x01broken\x00'):
            with self.assertRaises(InputError):parse_upload(raw)
        parsed=parse_upload(b'Subject: broken multipart\r\nContent-Type: multipart/mixed\r\n\r\nrecoverable body')
        self.assertTrue(parsed['parse_warnings'])

    def test_exact_two_builders_and_filter_header_removal(self):
        p=parse_upload(b'Subject: (SPAM? 8.4) Meeting\r\n\r\nX-Spam-Flag: YES\r\n\tfilter detail\r\nThe requested meeting.')
        original,clean,changes=build_texts(p)
        self.assertEqual(original,'(SPAM? 8.4) Meeting\n\nThe requested meeting.')
        self.assertEqual(clean,'Meeting\n\nThe requested meeting.')
        self.assertIn('remove_leading_numeric_filter_marker',changes['Laya_day5'])

    def test_empty_features_produce_no_decision_on_real_e0(self):
        model=E0()
        result=model.predict('')
        self.assertEqual(result['active_features'],0)
        self.assertEqual(result['status'],'insufficient_features')
        self.assertIsNone(result['decision'])


class NeuralBoundaryTests(unittest.TestCase):
    def test_busy_neural_slot_is_a_refusal(self):
        from app.worker_client import LayaWorker
        worker=LayaWorker(enabled=False);worker.state='ready'
        worker.lock.acquire()
        try:
            result=worker.predict('ordinary text')
            self.assertEqual(result['status'],'busy');self.assertIsNone(result['decision'])
        finally:worker.lock.release()

    def test_worker_timeout_stops_own_process_and_never_falls_back(self):
        import queue
        from app.worker_client import LayaWorker
        worker=LayaWorker(enabled=False);worker.state='ready';worker.process=Mock()
        worker.process.poll.return_value=None
        worker.responses=Mock();worker.responses.get.side_effect=queue.Empty()
        result=worker.predict('ordinary text')
        self.assertEqual(result['status'],'unavailable');self.assertIsNone(result['decision'])
        worker.process.terminate.assert_called_once()
        self.assertEqual(worker.readiness()['status'],'unavailable')


class APIBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.engine=Mock()
        self.engine.readiness.return_value={'modes':{'baseline':True,'recall':False,'review':False}}
        self.engine.analyze.return_value={'status':'ok','decision':0,'infrastructure_failure':False}
        self.client=TestClient(create_app(self.engine));self.client.__enter__()

    def tearDown(self):self.client.__exit__(None,None,None)

    def post(self,raw=b'Subject: a\r\n\r\nbody',**kwargs):
        return self.client.post('/api/analyze',content=raw,headers={'Content-Type':'message/rfc822'},**kwargs)

    def test_unknown_mode_and_arbitrary_model_path_rejected(self):
        self.assertEqual(self.post(params={'mode':'Qwen'}).status_code,422)
        self.assertEqual(self.post(params={'model_path':'evil.joblib'}).status_code,422)
        self.engine.analyze.assert_not_called()

    def test_filename_header_cannot_leak_into_model_or_storage(self):
        raw=b'Subject: hello\r\n\r\nmeeting'
        for filename in ('spam/answer.eml','ham/answer.eml','../../evil.eml'):
            response=self.client.post('/api/analyze?mode=baseline',content=raw,
                headers={'Content-Type':'message/rfc822','Content-Disposition':'attachment; filename="'+filename+'"'})
            self.assertEqual(response.status_code,200)
            self.assertEqual(self.engine.analyze.call_args.args,(raw,'baseline'))

    def test_stream_limit_without_content_length(self):
        def chunks():
            for _ in range(11):yield b'x'*(1024*1024)
        response=self.client.post('/api/analyze',content=chunks(),headers={'Content-Type':'message/rfc822'})
        self.assertEqual(response.status_code,413);self.engine.analyze.assert_not_called()

    def test_declared_oversize_unsupported_mime_empty_error(self):
        self.assertEqual(self.post(b'x'*(MAX_BYTES+1)).status_code,413)
        self.assertEqual(self.client.post('/api/analyze',json={'email':'fake'}).status_code,415)
        self.engine.analyze.side_effect=InputError('empty_file')
        self.assertEqual(self.post(b'').status_code,400)

    def test_readiness_and_component_unavailability(self):
        self.assertEqual(self.client.get('/ready').status_code,200)
        self.assertEqual(self.client.get('/ready?mode=recall').status_code,503)
        self.engine.analyze.return_value={'status':'needs_review','decision':None,'infrastructure_failure':True}
        response=self.post(params={'mode':'review'})
        self.assertEqual(response.status_code,503);self.assertIsNone(response.json()['decision'])

    def test_cross_origin_upload_forbidden_and_csp(self):
        response=self.client.post('/api/analyze',content=b'private',headers={'Origin':'https://evil.example','Content-Type':'message/rfc822'})
        self.assertEqual(response.status_code,403);self.engine.analyze.assert_not_called()
        self.assertIn("default-src 'none'",response.headers['content-security-policy'])
        self.assertEqual(response.headers['cache-control'],'no-store')

    def test_html_output_is_inert_and_raw_body_not_html_response(self):
        raw=b'Subject: <script>alert(1)</script>\r\nContent-Type: text/html\r\n\r\n<script>alert(2)</script><a href="javascript:alert(3)">click</a>'
        parsed=parse_upload(raw)
        self.assertNotIn('alert(2)',parsed['body_text'])
        self.assertTrue(parsed['links'][0]['scheme_high_risk'])
        response=self.post(raw)
        self.assertEqual(response.headers['content-type'],'application/json')
        js=self.client.get('/assets/app.js').text
        self.assertIn('textContent',js);self.assertNotIn('innerHTML',js)
        self.assertNotIn('href =',js)

    def test_documented_raw_binary_endpoint(self):
        schema=self.client.get('/openapi.json').json()
        self.assertIn('message/rfc822',schema['paths']['/api/analyze']['post']['requestBody']['content'])


if __name__=='__main__':unittest.main()
