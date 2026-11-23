/* Browser-independent annotation rules. No inference or semantic auto-labeling. */
(function (root) {
  'use strict';
  const kinds = {time:'时间', quantity:'数量', negation:'否定', condition:'条件', attribution:'归因 / 谁说的', scope:'范围 / 仅 / 至少', causality:'因果', location:'地点'};
  const clone = value => JSON.parse(JSON.stringify(value));
  function spans(text, quote) {
    if (!quote) return [];
    const s = Array.from(text), q = Array.from(quote), found = [];
    for (let i=0; i<=s.length-q.length; i++) {
      if (q.every((c,j) => c===s[i+j])) found.push({start_idx:i, end_idx:i+q.length});
    }
    return found;
  }
  const nonempty = x => typeof x==='string' && !!x.trim();
  const person = x => nonempty(x) && !['pending','todo','agent','model'].includes(x.trim().toLowerCase());
  function factErrors(f, source) {
    const errors=[];
    if (!nonempty(f.fact_id)) errors.push('缺少事实编号');
    if (!nonempty(f.proposition)) errors.push('请写出完整主张（含必要的主语与宾语）');
    if (!nonempty(f.quote) || !spans(source,f.quote).length) errors.push('引用必须逐字出现在原文中，请重新选择或粘贴');
    if (!Array.isArray(f.context_quotes)) errors.push('上下文格式错误');
    else for(const c of f.context_quotes) if(!nonempty(c.quote)||!spans(source,c.quote).length) errors.push('上下文引用未在原文找到');
    if (!Array.isArray(f.qualifiers)) errors.push('限定条件格式错误');
    else {
      const ids=new Set();
      for(const q of f.qualifiers) {
        if(!nonempty(q.qualifier_id)||ids.has(q.qualifier_id)||!Object.hasOwn(kinds,q.kind)||!nonempty(q.text)) errors.push('请补全限定条件的类型与内容');
        ids.add(q.qualifier_id);
      }
    }
    return errors;
  }
  function itemErrors(item) {
    if(!Array.isArray(item.facts)||!item.facts.length) return ['至少记录一条主张；暂不能确定时保留待办'];
    const errors=[], ids=new Set();
    item.facts.forEach((f,i)=>{
      if(ids.has(f.fact_id)) errors.push('事实编号重复');
      ids.add(f.fact_id);
      errors.push(...factErrors(f,item.text).map(e=>`主张 ${i+1}：${e}`));
    });
    return errors;
  }
  function normalizedFact(f, source) {
    return {fact_id:f.fact_id, proposition:f.proposition, quote:f.quote, occurrences:spans(source,f.quote),
      context_quotes:f.context_quotes.map(c=>({quote:c.quote,occurrences:spans(source,c.quote)})),
      qualifiers:clone(f.qualifiers)};
  }
  function normalize(review) {
    const d=clone(review);
    d.items=d.items.map(item=>({...item,facts:item.facts.map(f=>normalizedFact(f,item.text))}));
    return d;
  }
  function validateImport(input, template) {
    if(!input || input.reviewer_type!=='human' || !Array.isArray(input.items) || input.items.length!==template.items.length) throw Error('不是当前标注模板：需要 human 类型及完整的同一批样本');
    const map=new Map(input.items.map(s=>[s.id,s]));
    if(map.size!==template.items.length) throw Error('样本编号重复');
    for(const t of template.items) {
      const s=map.get(t.id);
      if(!s||s.text!==t.text||!['pending','complete'].includes(s.status)||!Array.isArray(s.facts)) throw Error('样本、原文或状态与本页面不一致');
      for(const f of s.facts) {
        if(!f || !nonempty(f.fact_id)||typeof f.quote!=='string'||typeof f.proposition!=='string'||!Array.isArray(f.context_quotes)||!Array.isArray(f.qualifiers)) throw Error('事实字段格式不正确');
        if(f.context_quotes.some(c=>!c||typeof c.quote!=='string') || f.qualifiers.some(q=>!q||typeof q.text!=='string'||!Object.hasOwn(kinds,q.kind)||!nonempty(q.qualifier_id))) throw Error('上下文或限定条件格式不正确');
      }
      if(s.status==='complete' && itemErrors(s).length) throw Error(`已完成样本 ${s.id} 存在漏填或无效引用`);
    }
    return normalize({...input,items:template.items.map(t=>map.get(t.id))});
  }
  function finalReview(review, time) {
    if(!person(review.reviewer_id)) throw Error('请填写有效的标注者化名');
    const remaining=review.items.filter(s=>s.status!=='complete'||itemErrors(s).length);
    if(remaining.length) throw Error(`还有 ${remaining.length} 条待完成，请先保存草稿`);
    const d=normalize(review); d.reviewer_id=d.reviewer_id.trim(); d.completed_at_utc=time;
    return d;
  }
  const api={kinds,clone,spans,person,factErrors,itemErrors,normalize,validateImport,finalReview};
  if(typeof module!=='undefined' && module.exports) module.exports=api;
  else root.MiniFactCore=api;
})(globalThis);
