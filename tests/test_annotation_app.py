"""Blinded packaging and compatibility with the existing human-review schema."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from zipfile import ZipFile

from next_eval.annotations import check_facts
from next_eval.common import ROOT, read
from scripts.build_annotation_app import build


class AnnotationAppTests(unittest.TestCase):
    def test_packages_are_standalone_blind_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'pages'
            bundle=Path(temp)/'bundle'
            shutil.copytree(ROOT/'data/next_eval_v2',bundle)
            expected=read(bundle/'reviewer_b.json')
            expected={'reviewer_id':None,'reviewer_type':'human','completed_at_utc':None,
                      'items':[{'id':s['id'],'text':s['text'],'status':'pending','facts':[]} for s in expected['items']]}
            (bundle/'reviewer_b.json').write_text('{"reviewer_id":"DO_NOT_EMBED_B_ANSWER"}',encoding='utf-8')
            build(bundle,out)
            for role in ('A','B'):
                with ZipFile(out/f'MiniFact_标注网页分享包_{role}.zip') as z:
                    self.assertIsNone(z.testzip())
                    self.assertEqual(set(z.namelist()),{f'MiniFact_标注页面_{role}.html','先读我.txt','NOTICE.md'})
                    html=z.read(f'MiniFact_标注页面_{role}.html').decode('utf-8')
                data=json.loads(re.search(r'<script id="packet" type="application/json">(.*?)</script>',html,re.S)[1])
                self.assertEqual(data['role'],role)
                self.assertEqual(data['template'],expected)
                self.assertEqual(len(data['template']['items']),60)
                self.assertNotIn('gold',data['template'])
                self.assertNotRegex(html,r'<(?:script|link)[^>]+(?:src|href)=')
                self.assertNotIn('/*APP_',html)
                self.assertNotIn('DO_NOT_EMBED_B_ANSWER',html)
            with self.assertRaises(ValueError): build(ROOT/'data/next_eval_v2',out)

    @unittest.skipUnless(shutil.which('node'),'Node needed for browser-to-Python contract check')
    def test_js_export_passes_existing_python_validator_including_unicode(self):
        script="""const C=require('./annotation_app/core.js');
const s='😀 Mira said no. Mira said no.';
const r={reviewer_id:'synthetic_fixture',reviewer_type:'human',items:[{id:'TEST-ONLY',text:s,status:'complete',facts:[{fact_id:'b_f1',quote:'Mira said no',proposition:'Mira said no',context_quotes:[{quote:'😀'}],qualifiers:[{qualifier_id:'q1',kind:'negation',text:'no'}]}]}]};
process.stdout.write(JSON.stringify(C.finalReview(r,'2026-11-23T00:00:00Z')));"""
        raw=subprocess.check_output(['node','-e',script],cwd=ROOT)
        exported=json.loads(raw)
        self.assertEqual(len(check_facts(exported['items'][0],exported['items'][0]['text'])),1)


if __name__=='__main__': unittest.main()
