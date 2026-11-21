import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from next_eval.common import ROOT, ARMS, PROTOCOL, object_sha, read, write_new, validate_inputs
from next_eval.backend import EvaluationBackend, ReplayBackend, JUDGE, payload
from next_eval.data import boundaries, spans, validate_bundle
from next_eval.annotations import (check_facts, readiness, semantic_template, semantic_metrics)
from next_eval.runner import check_prefix, freeze, order, selective_policy
from next_eval.metrics import wilson, paired_interval, auroc, acceptance

BUNDLE=ROOT/'data/next_eval_v2'


def generation(system, request, role, text, eos=True):
    config=read(ROOT/'config.json')
    params={k:config[k] for k in ('max_new_tokens','do_sample','num_beams','repetition_penalty','temperature','top_p','top_k','use_cache')}
    params['max_new_tokens']=PROTOCOL['caps']['decision' if role=='verify' else role]
    ids=[ord(c)+1 for c in text]+([0] if eos else [])
    g={'raw_response':text,'raw_response_with_special_tokens':text+('<eos>' if eos else ''),
       'generated_token_ids':ids,'prompt_input_ids':[2,3],
       'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(request,ensure_ascii=False)}],
       'generation_params':params,'ended_with_eos':eos}
    if role in ('decision','verify'):
        g['token_trace']=[{'step':i,'token_id':t,'decoded_piece':'' if t==0 else chr(t-1),
                           'special':t==0,'label_token':None,'raw_logprob':-.1,'full_vocab_entropy_nats':.2} for i,t in enumerate(ids)]
        g['mean_nonspecial_generated_logprob']=-.1
    return g


class Scripted(EvaluationBackend):
    def __init__(self, repair=False, truncated=False, bad=False):
        super().__init__(); self.requests=[]; self.repair=repair; self.truncated=truncated; self.bad=bad

    def generate(self, system, request, role):
        self.requests.append((system,copy.deepcopy(request),role))
        text='TRUE'
        if role=='extract':
            text='not JSON' if self.bad else json.dumps([{'quote':request['source'],'proposition':None,'context_quotes':[]}])
        if role=='feedback': text='Retain the stated date.' if self.repair else 'NO_ISSUE'
        return generation(system,request,role,text,not(self.truncated and role in ('feedback','extract')))


def row():
    return {'id':'SECRET-ID','text':'Mira joined the clinic.',
            'context':{'claim_date':'2020-01-01','speaker':None,'location_ISO_code':'GB'},
            'evidence':[{'question':'Who joined?','answer':'Mira joined.','url':'https://example.org/false-verdict'}]}


class BackendTests(unittest.TestCase):
    def test_payload_information_ladder_and_common_final_prompt(self):
        for arm,keys in [('B',{'statement'}),('C',{'statement','context'}),('D',{'statement','context','evidence'})]:
            backend=Scripted(); result=backend.run(row(),arm)
            self.assertEqual(set(backend.requests[-1][1]),keys)
            self.assertEqual(backend.requests[-1][0],JUDGE)
            self.assertEqual(result['verdict'],'SUPPORTED')
            self.assertNotIn('SECRET-ID',json.dumps(backend.requests))
            self.assertNotIn('false-verdict',json.dumps(backend.requests))

    def test_single_round_and_two_round_feedback(self):
        e,f=Scripted(repair=True),Scripted(repair=True)
        er,fr=e.run(row(),'E'),f.run(row(),'F')
        self.assertEqual(len(er['rounds']),1); self.assertEqual(len(fr['rounds']),2)
        self.assertFalse(er['rounds'][0]['feedback_used']); self.assertTrue(fr['rounds'][0]['feedback_used'])
        req=[r for _,r,role in f.requests if role=='extract']
        self.assertNotIn('untrusted_verifier_feedback',req[0])
        self.assertIn('untrusted_verifier_feedback',req[1])
        self.assertEqual(e.requests[0],f.requests[0])
        self.assertEqual(e.requests[-1][0],f.requests[-1][0])

    def test_no_issue_stops_without_fabricated_second_round(self):
        self.assertEqual(len(Scripted().run(row(),'F')['rounds']),1)

    def test_truncated_extraction_and_feedback_not_consumed(self):
        b=Scripted(repair=True,truncated=True); r=b.run(row(),'F')
        self.assertFalse(r['extraction']['accepted']); self.assertEqual(len(r['rounds']),1)
        self.assertNotIn('verify',[role for _,_,role in b.requests])
        self.assertEqual(b.requests[-1][1]['statement'],row()['text'])

    def test_malformed_extraction_keeps_original_and_failure(self):
        r=Scripted(bad=True).run(row(),'E')
        self.assertTrue(r['extraction']['errors']); self.assertEqual(r['generation']['messages'][1]['content'].count(row()['text']),1)

    def test_replay_rejects_evidence_or_verdict_tampering(self):
        r=Scripted().run(row(),'F'); ReplayBackend(r).check(row(),'F')
        bad=copy.deepcopy(r); bad['verdict']='REFUTED'
        with self.assertRaises(ValueError): ReplayBackend(bad).check(row(),'F')
        bad=copy.deepcopy(r); bad['calls'][-1]['generation']['messages'][1]['content']='{"gold":"TRUE"}'
        with self.assertRaises(ValueError): ReplayBackend(bad).check(row(),'F')

    def test_nonliteral_reconstruction_has_no_fake_offsets(self):
        from accuracy_claims import parse_extraction
        text='Mira owns a boat and a car.'
        parsed=parse_extraction(text,json.dumps([{'quote':'a car','proposition':'Mira owns a car','context_quotes':['Mira owns']}]))
        claim=parsed['accepted'][0]
        self.assertEqual(text[claim['start_idx']:claim['end_idx']],'a car')
        self.assertEqual(claim['proposition_status'],'unverified_model_reconstruction')


class DataTests(unittest.TestCase):
    def test_shipped_candidate_bundle_and_blinding(self):
        dataset,rows=validate_bundle(BUNDLE)
        self.assertEqual(len(rows),200); self.assertEqual(len(dataset['semantic_ids']),60)
        a,b=read(BUNDLE/'reviewer_a.json'),read(BUNDLE/'reviewer_b.json')
        self.assertIsNone(a['reviewer_id']); self.assertIsNone(b['reviewer_id'])
        self.assertTrue(all(set(i)=={'id','text','status','facts'} for i in a['items']))

    def test_missing_human_review_blocks_formal_freeze_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'freeze.json'
            with self.assertRaises(ValueError): freeze('news',BUNDLE,output)
            self.assertFalse(output.exists())

    def test_input_gold_injection_rejected(self):
        r=row(); r['gold']='SUPPORTED'
        with self.assertRaises(ValueError): validate_inputs([r])

    def test_boundary_balance_insufficiency_and_all_occurrences(self):
        cases=boundaries()['samples']
        self.assertEqual(len(cases),30)
        self.assertEqual(sum(s['gold']=='NOT_ENOUGH_INFO' for s in cases),6)
        for s in cases:
            check_facts({'text':s['text'],'status':'complete','facts':s['facts']},s['text'])
        self.assertEqual(len(cases[0]['facts'][0]['occurrences']),2)

    def test_unicode_offsets_are_codepoints(self):
        self.assertEqual(spans('😀 猫。猫。','猫'),[{'start_idx':2,'end_idx':3},{'start_idx':4,'end_idx':5}])

    def test_reference_omitting_repeated_occurrence_rejected(self):
        s=boundaries()['samples'][0]
        facts=copy.deepcopy(s['facts']); facts[0]['occurrences'].pop()
        with self.assertRaises(ValueError): check_facts({'text':s['text'],'status':'complete','facts':facts},s['text'])

    def test_new_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.json'; write_new(p,{'a':1})
            with self.assertRaises(FileExistsError): write_new(p,{'a':2})
            self.assertEqual(read(p),{'a':1})


class StatisticsTests(unittest.TestCase):
    def test_wilson_boundaries(self):
        self.assertIsNone(wilson(0,0)); self.assertAlmostEqual(wilson(0,100)[0],0)
        self.assertAlmostEqual(wilson(100,100)[1],1)
        self.assertLess(wilson(80,100)[0],.8); self.assertGreater(wilson(80,100)[1],.8)

    def test_paired_ci_uses_pairing_and_is_deterministic(self):
        gold=[{'id':str(i),'label':'SUPPORTED' if i<5 else 'REFUTED'} for i in range(10)]
        pred=[{'id':g['id'],'verdict':g['label']} for g in gold]
        self.assertEqual(paired_interval(gold,pred,pred)['ci95'],[0,0])
        wrong=[{'id':g['id'],'verdict':None} for g in gold]
        self.assertEqual(paired_interval(gold,wrong,pred)['ci95'],[1,1])

    def test_auroc_ties_and_missing_classes(self):
        self.assertEqual(auroc([(1,True),(0,False)]),1)
        self.assertEqual(auroc([(1,True),(1,False)]),.5)
        self.assertIsNone(auroc([(1,True)]))

    def test_abstentions_stay_in_denominators(self):
        from phase3_metrics import evaluate
        g=[{'id':'r','label':'REFUTED'},{'id':'s','label':'SUPPORTED'}]
        p=[{'id':'r','verdict':'NOT_ENOUGH_INFO'},{'id':'s','verdict':None}]
        m=evaluate(g,p)
        self.assertEqual(m['accuracy'],0); self.assertEqual(m['false_negative_rate'],0)
        self.assertEqual(m['missed_error_rate_including_abstentions'],1)

    def test_upgrade_cannot_pass_without_semantics_or_with_risk_regression(self):
        f={'accuracy':.9,'false_negative_rate':.05,'missed_error_rate_including_abstentions':.10,
           'false_positive_rate':.15,'cost_and_extraction':{'accepted_span_precision':1}}
        d={**f,'accuracy':.80}
        p={'bootstrap':{'ci95':[.03,.17]}}
        sem={'F':{k:1.0 for k in ('fact_recall','atomicity','faithfulness','qualifier_retention')}}
        self.assertFalse(acceptance({'D':d,'F':f},None,p,True)['upgrade_default_method'])
        self.assertTrue(acceptance({'D':d,'F':f},sem,p,True)['upgrade_default_method'])
        d['false_positive_rate']=.1
        self.assertFalse(acceptance({'D':d,'F':f},sem,p,True)['upgrade_default_method'])
        self.assertTrue(acceptance({'D':d,'F':f},sem,p,True)['evaluation_complete'])

    def test_development_thresholds_require_complete_32(self):
        with self.assertRaises(ValueError): selective_policy({'completed':True,'stage':'news','results':[]})


class CheckpointTests(unittest.TestCase):
    def test_partial_last_arm_is_resumable_but_completed_missing_arm_is_not(self):
        rows=[{'id':'x','text':'one'},{'id':'y','text':'two'}]
        run={'results':[{'id':'x','source':'one','arms':{a:{} for a in ARMS}},
                        {'id':'y','source':'two','arms':{'B':{}}}]}
        check_prefix(run,rows,list(ARMS),False)
        with self.assertRaises(ValueError): check_prefix(run,rows,list(ARMS),True)
        run['results'][1]['source']='changed'
        with self.assertRaises(ValueError): check_prefix(run,rows,list(ARMS),False)

    def test_arm_order_rotates(self):
        self.assertEqual(order(list(ARMS),1),['B','C','D','E','F','A'])


class SemanticTests(unittest.TestCase):
    def test_missing_fact_lowers_recall_not_conditional_qualifier_denominator(self):
        source='A opened. B closed in June.'
        ex={'raw_items':[{'quote':'A opened','proposition':None,'context_quotes':[]}],
            'accepted':[{'quote_index':0}]}
        run={'completed_at_utc':'2026-11-22T10:00:00+00:00','results':[{'id':'x','source':source,'arms':{a:{'extraction':ex} for a in ('E','F')}}]}
        gold={'items':[{'id':'x','facts':[{'fact_id':'a','qualifiers':[]},
                                        {'fact_id':'b','qualifiers':[{'qualifier_id':'time'}]}]}]}
        alignment=semantic_template(run,gold)
        alignment.update(status='complete',reviewer_ids=['reader1','reader2'],completed_at_utc='2026-11-22T11:00:00+00:00')
        for i in alignment['items']:
            i['status']='complete'
            i['predictions'][0].update(fact_ids=['a'],atomic=True,faithful=True)
        result=semantic_metrics(run,gold,alignment)['F']
        self.assertEqual(result['fact_recall'],.5)
        self.assertEqual(result['end_to_end_qualifier_retention'],0)
        self.assertIsNone(result['qualifier_retention'])

    def test_tampered_alignment_or_missing_prediction_rejected(self):
        ex={'raw_items':[{'quote':'A'}],'accepted':[{'quote_index':0}]}
        run={'completed_at_utc':'2026-11-22T10:00:00+00:00','results':[{'id':'x','source':'A','arms':{a:{'extraction':ex} for a in ('E','F')}}]}
        gold={'items':[{'id':'x','facts':[{'fact_id':'a','qualifiers':[]}]}]}
        alignment=semantic_template(run,gold)
        alignment.update(status='complete',reviewer_ids=['r1','r2'],completed_at_utc='2026-11-22T11:00:00+00:00')
        for i in alignment['items']: i.update(status='complete',predictions=[])
        with self.assertRaises(ValueError): semantic_metrics(run,gold,alignment)


class HumanReadinessTests(unittest.TestCase):
    """Synthetic in-memory review fixtures test guards; never real annotations."""
    def fixtures(self):
        dataset,inputs=validate_bundle(BUNDLE)
        selected={s['id']:s['text'] for s in inputs if s['id'] in dataset['semantic_ids']}
        from next_eval.data import history
        events={'reviewer':'fixture_event_reviewer','reviewer_type':'human','completed_at_utc':'2026-11-22T08:00:00+00:00',
                'historical_records':history(), 'items':[{'id':s['id'],'text':s['text'],'decision':'include',
                    'event_key':f'test-fixture-{i}','overlaps_history':False,'reason':'Synthetic test fixture, not a real review.'} for i,s in enumerate(inputs)]}
        items=[]
        for sid,text in selected.items():
            items.append({'id':sid,'text':text,'status':'complete','facts':[
                {'fact_id':'f1','proposition':text,'quote':text,'occurrences':spans(text,text),
                 'context_quotes':[],'qualifiers':[]}]})
        a={'reviewer_id':'fixture_a','reviewer_type':'human','completed_at_utc':'2026-11-22T09:00:00+00:00','items':copy.deepcopy(items)}
        b={**copy.deepcopy(a),'reviewer_id':'fixture_b'}
        final={'reviewer_ids':['fixture_a','fixture_b'],'status':'complete','completed_at_utc':'2026-11-22T10:00:00+00:00',
               'reviewer_a_sha256':'a'*64,'reviewer_b_sha256':'b'*64,'items':copy.deepcopy(items)}
        for item in final['items']:
            item.update(resolution='Synthetic aligned fixture.',rejected_reviewer_a_fact_ids=[],rejected_reviewer_b_fact_ids=[])
            item['facts'][0].update(reviewer_a_fact_ids=['f1'],reviewer_b_fact_ids=['f1'])
        return dataset,inputs,{'event_review.json':events,'reviewer_a.json':a,'reviewer_b.json':b,'adjudication.json':final}

    def call(self,objects,dataset,inputs):
        with patch('next_eval.annotations.validate_bundle',return_value=(dataset,inputs)), \
             patch('next_eval.annotations.read',side_effect=lambda p:objects[p.name]), \
             patch('next_eval.annotations.sha',side_effect=lambda p:'a'*64 if p.name=='reviewer_a.json' else 'b'*64):
            return readiness(Path('fixture'))

    def test_complete_two_reviewers_and_resolution_accepted(self):
        d,i,o=self.fixtures(); result=self.call(o,d,i)
        self.assertTrue(result['ready']); self.assertEqual(result['agreement']['samples'],60)

    def test_same_reviewer_rejected(self):
        d,i,o=self.fixtures(); o['reviewer_b.json']['reviewer_id']='fixture_a'
        with self.assertRaises(ValueError): self.call(o,d,i)

    def test_machine_semantic_reviewer_rejected(self):
        d,i,o=self.fixtures(); o['reviewer_b.json']['reviewer_type']='agent'
        with self.assertRaises(ValueError): self.call(o,d,i)

    def test_duplicate_event_and_history_overlap_rejected(self):
        d,i,o=self.fixtures(); o['event_review.json']['items'][1]['event_key']=o['event_review.json']['items'][0]['event_key']
        with self.assertRaises(ValueError): self.call(o,d,i)
        d,i,o=self.fixtures(); o['event_review.json']['items'][0]['overlaps_history']=True
        with self.assertRaises(ValueError): self.call(o,d,i)

    def test_changed_initial_review_hash_rejected(self):
        d,i,o=self.fixtures(); o['adjudication.json']['reviewer_a_sha256']='0'*64
        with self.assertRaises(ValueError): self.call(o,d,i)

    def test_unresolved_proposal_cannot_disappear(self):
        d,i,o=self.fixtures(); o['adjudication.json']['items'][0]['facts'][0]['reviewer_a_fact_ids']=[]
        with self.assertRaises(ValueError): self.call(o,d,i)

    def test_inverted_timeline_rejected(self):
        d,i,o=self.fixtures(); o['adjudication.json']['completed_at_utc']='2026-11-22T07:00:00+00:00'
        with self.assertRaises(ValueError): self.call(o,d,i)


if __name__=='__main__':
    unittest.main()
