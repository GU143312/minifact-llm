const test=require('node:test');
const assert=require('node:assert/strict');
const C=require('../annotation_app/core.js');
const source='😀 Mira said no. Mira said no.';
const fact=()=>({fact_id:'b_f1',quote:'Mira said no',proposition:'Mira said no',occurrences:[],context_quotes:[],qualifiers:[{qualifier_id:'q1',kind:'negation',text:'no'}]});
const review=()=>({reviewer_id:'test_fixture_only',reviewer_type:'human',completed_at_utc:null,items:[{id:'fixture',text:source,status:'pending',facts:[fact()]}]});
test('Unicode offsets are Python codepoints and enumerate repeated/overlapping quotes',()=>{
 assert.deepEqual(C.spans(source,'Mira said no'),[{start_idx:2,end_idx:14},{start_idx:16,end_idx:28}]);
 assert.deepEqual(C.spans('aaaa','aa'),[{start_idx:0,end_idx:2},{start_idx:1,end_idx:3},{start_idx:2,end_idx:4}]);
 assert.deepEqual(C.spans(source,''),[]);
});
test('reject a paraphrase as literal source while allowing full proposition',()=>{
 const f=fact();f.quote='Mira rejected it';assert.ok(C.factErrors(f,source).length);
 f.quote='Mira said no';f.proposition='Mira refused';assert.equal(C.factErrors(f,source).length,0);
});
test('completed export requires explicit sample completion and a human pseudonym',()=>{
 const r=review();assert.throws(()=>C.finalReview(r,'2026-11-23T00:00:00Z'));
 r.items[0].status='complete';r.reviewer_id='agent';assert.throws(()=>C.finalReview(r,'2026-11-23T00:00:00Z'));
 r.reviewer_id='reader';const exported=C.finalReview(r,'2026-11-23T00:00:00Z');
 assert.equal(exported.completed_at_utc,'2026-11-23T00:00:00Z');assert.equal(exported.items[0].facts[0].occurrences.length,2);assert.deepEqual(r.items[0].facts[0].occurrences,[]);
});
test('missing qualifiers, duplicate IDs and missing facts cannot silently pass',()=>{
 const r=review();r.items[0].facts[0].qualifiers[0].text='';assert.ok(C.itemErrors(r.items[0]).length);
 r.items[0].facts=[fact(),fact()];assert.ok(C.itemErrors(r.items[0]).length);
 r.items[0].facts=[];assert.ok(C.itemErrors(r.items[0]).length);
});
test('import rejects tampered source, unknown IDs and duplicate samples',()=>{
 const t=review(),r=review();r.items[0].text='changed';assert.throws(()=>C.validateImport(r,t));
 r.items[0]=structuredClone(t.items[0]);r.items[0].id='unknown';assert.throws(()=>C.validateImport(r,t));
 r.items=[t.items[0],t.items[0]];assert.throws(()=>C.validateImport(r,t));
});
test('imported completion must be valid; incomplete drafts remain editable',()=>{
 const t=review(),r=review();r.items[0].facts[0].quote='missing';
 assert.equal(C.validateImport(r,t).items[0].status,'pending');r.items[0].status='complete';assert.throws(()=>C.validateImport(r,t));
});
test('context occurrences are rebuilt without altering independent text',()=>{
 const r=review();r.items[0].facts[0].context_quotes=[{quote:'😀',occurrences:[{start_idx:99,end_idx:100}]}];
 const n=C.normalize(r);assert.deepEqual(n.items[0].facts[0].context_quotes[0].occurrences,[{start_idx:0,end_idx:1}]);assert.equal(r.items[0].facts[0].context_quotes[0].occurrences[0].start_idx,99);
});
