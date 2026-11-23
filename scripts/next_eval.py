"""Prepare, review, freeze, execute and audit MiniFact's prospective evaluation."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from next_eval.common import read, require, write_new


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude',type=int,nargs='*',default=[],help='Reviewed upstream indices to exclude; creates a fresh bundle')
    for command in ('check','readiness'):
        p=sub.add_parser(command); p.add_argument('--bundle',type=Path,required=True)
        if command=='check': p.add_argument('--source',type=Path,help='Additionally rebuild selected data from pinned upstream bytes')
    p=sub.add_parser('freeze')
    p.add_argument('--stage',choices=('dev','boundary','news'),required=True)
    p.add_argument('--bundle',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dev-run',type=Path); p.add_argument('--limit',type=int)
    p=sub.add_parser('run')
    p.add_argument('--freeze',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--cache-dir',type=Path,default=Path(os.environ.get('MINIFACT_MODEL_CACHE','E:/MiniFact-LLM-runtime/hf-cache')))
    p=sub.add_parser('audit')
    p.add_argument('--run',type=Path,required=True); p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--cache-dir',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    p=sub.add_parser('evaluate')
    p.add_argument('--run',type=Path,required=True); p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--alignment',type=Path)
    p.add_argument('--boundary-run',type=Path)
    p=sub.add_parser('alignment-template')
    p.add_argument('--run',type=Path,required=True); p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('report')
    p.add_argument('--metrics',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command not in ('check','readiness','run'):
        require(not args.output.exists(),'Output exists; choose a new filename/version')
    if args.command=='prepare':
        from next_eval.data import prepare
        result=prepare(args.source,args.output,args.exclude)
    elif args.command=='check':
        from next_eval.data import validate_bundle, build
        d,rows=validate_bundle(args.bundle)
        if args.source:
            rebuilt,projected,_=build(args.source,d['excluded_source_indices'])
            require(rebuilt==d and projected==rows,'Pinned-source reconstruction differs')
        result={'valid_candidate_bundle':True,'news':len(rows),'semantic':len(d['semantic_ids']),'human_readiness_checked':False}
    elif args.command=='readiness':
        from next_eval.annotations import readiness
        result=readiness(args.bundle)
    elif args.command=='freeze':
        from next_eval.runner import freeze
        value=freeze(args.stage,args.bundle,args.output,args.dev_run,args.limit)
        result={'stage':value['stage'],'samples':len(value['ids']),'freeze':str(args.output)}
    elif args.command=='run':
        from next_eval.runner import execute
        value=execute(args.freeze,args.output,args.cache_dir,args.resume)
        result={'stage':value['stage'],'samples':len(value['results']),'completed':value['completed']}
    elif args.command=='audit':
        from next_eval.runner import audit
        frozen=read(args.freeze); tokenizer=None
        if args.cache_dir:
            from transformers import AutoTokenizer
            config=frozen['model']
            snapshot=args.cache_dir/('models--'+config['model_id'].replace('/','--'))/'snapshots'/config['model_revision']
            tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,use_fast=True)
        result=audit(read(args.run),frozen,args.freeze,tokenizer,args.allow_partial)
        write_new(args.output,result)
    elif args.command=='evaluate':
        from next_eval.metrics import summarize
        result=summarize(read(args.run),read(args.freeze),args.freeze,
                         read(args.alignment) if args.alignment else None,
                         read(args.boundary_run) if args.boundary_run else None)
        write_new(args.output,result)
        result={'stage':result['stage'],'acceptance':result['acceptance']}
    elif args.command=='alignment-template':
        from next_eval.runner import audit
        from next_eval.annotations import semantic_template
        run,frozen=read(args.run),read(args.freeze)
        require(run['stage']=='news','Only the complete formal news run has formal semantic alignment')
        audit(run,frozen,args.freeze)
        result=semantic_template(run,read(ROOT/frozen['bundle']/'adjudication.json'))
        write_new(args.output,result)
        result={'alignment_rows':len(result['items']),'status':'pending_human_review'}
    else:
        from next_eval.report import publish
        result=publish(args.metrics,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
