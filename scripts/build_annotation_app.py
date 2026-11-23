"""Package standalone blinded reviewer pages; never embeds labels or other reviews."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from next_eval.data import validate_bundle


def build(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; use a new directory so saved work is not overwritten')
    dataset, inputs = validate_bundle(bundle)
    by_id = {s['id']: s['text'] for s in inputs}
    # Reconstruct empty blind templates. Never package a filled reviewer file.
    blind = {'reviewer_id': None, 'reviewer_type': 'human', 'completed_at_utc': None,
             'items': [{'id': sid, 'text': by_id[sid], 'status': 'pending', 'facts': []}
                       for sid in dataset['semantic_ids']]}
    template_hash = hashlib.sha256(json.dumps(blind, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    app = ROOT/'annotation_app'
    shell = (app/'index.html').read_text(encoding='utf-8')
    assets = {marker: (app/file).read_text(encoding='utf-8') for marker, file in
              [('/*APP_STYLE*/','style.css'), ('/*APP_CORE*/','core.js'),
               ('/*APP_UI*/','app.js'), ('/*APP_GUIDE*/','guide.html')]}
    plain_guide = (app/'guide.txt').read_text(encoding='utf-8')
    output.mkdir(parents=True)
    for role in ('A', 'B'):
        page = output/f'MiniFact_标注页面_{role}.html'
        data = json.dumps({'role': role, 'bundle_sha256': template_hash, 'template': blind}, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
        html = shell
        for marker, content in {**assets, '/*APP_DATA*/': data}.items():
            html = html.replace(marker, content)
        page.write_text(html, encoding='utf-8', newline='\n')
        guide = f'''MiniFact 标注者 {role} 使用说明

这是给标注者 {role} 的包。先解压，再打开 MiniFact_标注页面_{role}.html。
全部完成后，发回网页导出的 reviewer_{role.lower()}.json 文件。
如果已经在旧网页开始填写，请先在旧页“下载草稿”，再在新版“恢复进度”，不要只复制 HTML。

{plain_guide}

页面不上传内容，没有外部服务或模型调用。自动保存依赖浏览器设置；文件下载才是可携带的备份。不要只转发填过的 HTML 来回传结果：HTML 不包含浏览器保存的进度。
独立初审完成后，协调者会组织分歧讨论并记录仲裁。此页面用于初审，不会自动把两人答案合并为金标准。
数据版本：{bundle.name}，60 条语义子集。包内无真假标签、证据、另一位的答案或共享计划文档。
样本来源与许可见 NOTICE.md；网页里的“填写指南”也包含以上讲解。
'''
        with ZipFile(output/f'MiniFact_标注网页分享包_{role}.zip', 'x', ZIP_DEFLATED) as z:
            z.writestr(page.name, page.read_bytes())
            z.writestr('先读我.txt', guide.encode('utf-8'))
            z.writestr('NOTICE.md', (ROOT/'data/improvement/NOTICE.md').read_bytes())
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'data/next_eval_v2')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(build(args.bundle, args.output))
