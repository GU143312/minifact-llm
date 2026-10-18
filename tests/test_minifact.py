import copy
import unittest
from compare import compatibility_errors
from evaluate import evaluate_case, evaluate_run
from minifact import align_quotes, exact_span, read_json, ROOT


def row(source,quotes):
    aligned,rejected=align_quotes(source,quotes)
    return dict(source=source,raw_quotes=quotes,aligned_claims=aligned,
                alignment_rejections=rejected,ended_with_eos=True)


class AlignmentTests(unittest.TestCase):
    def test_unicode_and_exclusive_offsets(self):
        source='新楼有40床🛏，急诊关闭。'
        claims,errors=align_quotes(source,['新楼有40床🛏','急诊关闭'])
        self.assertFalse(errors)
        self.assertTrue(all(exact_span(source,x) for x in claims))
        self.assertFalse(exact_span(source,dict(claim='新',start_idx=False,end_idx=1)))

    def test_repetition_rejected_without_guessing(self):
        claims,errors=align_quotes('a a',['a'])
        self.assertEqual(claims,[])
        self.assertEqual(errors[0]['reason'],'ambiguous_repeated_quote')
        self.assertEqual(len(errors[0]['matches']),2)

    def test_rewrite_duplicate_and_malformed_rejected(self):
        for quotes in [['forty beds'],['40 beds','40 beds'],[None],[],{}]:
            with self.subTest(quotes=quotes):
                self.assertTrue(align_quotes('40 beds',quotes)[1])


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.dataset=read_json(ROOT/'data/cases.json')
        self.cases={c['id']:c for c in self.dataset['cases']}

    def test_missing_time_does_not_erase_core_coverage(self):
        case=self.cases['base_03']
        metrics=evaluate_case(case,row(case['source'],['Green Line ridership rose 12%','ticket prices fell by 5%']))
        self.assertEqual(metrics['counts']['facts_covered'],2)
        self.assertEqual(metrics['counts']['qualifiers_retained'],2)
        self.assertEqual(metrics['counts']['gold_qualifiers'],3)
        self.assertEqual(metrics['counts']['exact_accepted_spans'],2)

    def test_negation_loss_is_qualifier_loss(self):
        case=self.cases['shared_02']
        metrics=evaluate_case(case,row(case['source'],['Eastport Council approved the park on Tuesday','approve the road on Wednesday']))
        self.assertEqual(metrics['counts']['facts_covered'],2)
        self.assertFalse(metrics['facts'][1]['qualifiers'][0]['retained'])

    def test_repeated_mentions_are_one_fact_and_alignment_is_separate(self):
        case=self.cases['repeat_01']
        metrics=evaluate_case(case,row(case['source'],['North Hall has 80 seats']))
        self.assertEqual(metrics['counts']['gold_facts'],1)
        self.assertEqual(metrics['counts']['facts_covered'],1)
        self.assertEqual(metrics['counts']['qualifiers_retained'],1)
        self.assertEqual(metrics['counts']['aligned_quote_items'],0)
        self.assertEqual(metrics['counts']['rejected_quote_items'],1)

    def test_nonliteral_subject_completion_does_not_hide_fact_or_qualifier_coverage(self):
        case=self.cases['shared_01']
        metrics=evaluate_case(case,row(case['source'],['Meadow Clinic hired 12 nurses in March','Meadow Clinic opened a new ward in July']))
        self.assertEqual(metrics['counts']['facts_covered'],2)
        self.assertEqual(metrics['counts']['qualifiers_retained'],3)
        self.assertEqual(metrics['counts']['aligned_quote_items'],1)
        self.assertEqual(metrics['counts']['rejected_quote_items'],1)

    def test_composite_quote_is_flagged_separately(self):
        case=self.cases['base_01']
        metrics=evaluate_case(case,row(case['source'],[case['source']]))
        self.assertEqual(metrics['counts']['facts_covered'],2)
        self.assertEqual(metrics['counts']['multi_fact_quote_items'],1)

    def test_all_reference_quotes_have_declared_core_and_qualifiers(self):
        for case in self.dataset['cases']:
            for fact in case['facts']:
                quote=fact['reference_quote']
                self.assertIn(quote,case['source'])
                self.assertTrue(all(x in quote for x in fact['core']))
                self.assertTrue(all(q['text'] in quote for q in fact['qualifiers']))

    def test_incomplete_runs_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_run(dict(completed=False,results=[]),self.dataset)


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        return dict(completed=True, contract={'pinned':True},
                    runtime={'packages':{p:'1' for p in ['torch','transformers','accelerate','tokenizers','safetensors']}},
                    execution=dict(actual_dtype='float16',actual_revision='revision',attention_implementation='eager'),
                    results=[dict(case_id='one',source='source',prompt_input_ids_sha256='same')])

    def test_matching_control_passes(self):
        a=self.fixture()
        self.assertEqual(compatibility_errors(a,copy.deepcopy(a)),[])

    def test_precision_prompt_code_or_version_drift_rejected(self):
        a=self.fixture()
        for path,value in [(('contract','pinned'),False),
                           (('execution','actual_dtype'),'bfloat16'),
                           (('runtime','packages','transformers'),'other')]:
            b=copy.deepcopy(a)
            target=b
            for key in path[:-1]:
                target=target[key]
            target[path[-1]]=value
            self.assertTrue(compatibility_errors(a,b))

    def test_actual_prompt_token_drift_rejected(self):
        a=self.fixture()
        b=copy.deepcopy(a)
        b['results'][0]['prompt_input_ids_sha256']='different'
        self.assertTrue(compatibility_errors(a,b))


if __name__=='__main__':
    unittest.main()
