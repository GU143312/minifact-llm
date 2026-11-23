/* A self-contained offline reviewer UI; data never leaves the browser. */
(() => {
  'use strict';
  const C=MiniFactCore, packet=JSON.parse(document.getElementById('packet').textContent);
  const $=id=>document.getElementById(id), key=`minifact-review-ui-v1:${packet.bundle_sha256}:${packet.role}`;
  let review=C.clone(packet.template), index=0, filter='all', selected='', localOK=true, conflict=false, restoreFailed=false;
  const el=(tag,cls,text)=>{const n=document.createElement(tag); if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  const button=(text,fn,cls='text-button')=>{const n=el('button',cls,text);n.type='button';n.addEventListener('click',fn);return n;};
  const item=()=>review.items[index];
  function notify(text,error=false){$('notice').hidden=false;$('notice').textContent=text;$('notice').className='notice'+(error?' error':'');}
  function confirmAction(title,text,action){$('confirmTitle').textContent=title;$('confirmText').textContent=text;$('acceptConfirm').onclick=()=>{$('confirmDialog').close();action();};$('confirmDialog').showModal();}
  $('cancelConfirm').onclick=()=>$('confirmDialog').close();
  function store(){
    if(conflict || restoreFailed) return;
    try {localStorage.setItem(key,JSON.stringify({version:1,role:packet.role,bundle_sha256:packet.bundle_sha256,review,index}));localOK=true;$('saveState').textContent='已本地保存';}
    catch {localOK=false;$('saveState').textContent='请下载草稿';notify('浏览器未允许本地保存。当前填写仍在页面中，请点击“下载草稿”保存，离开页面前务必备份。',true);}
  }
  try {
    const saved=JSON.parse(localStorage.getItem(key)||'null');
    if(saved){review=C.validateImport(saved.review,packet.template);index=Math.min(Math.max(saved.index||0,0),review.items.length-1);}
  } catch {localOK=false;restoreFailed=true;$('saveState').textContent='请恢复或下载草稿';notify('未能读取浏览器进度，已暂停自动保存以保留原记录。可用“恢复进度”导入之前下载的草稿；当前新填写请下载备份。',true);}
  window.addEventListener('storage',e=>{if(e.key===key){conflict=true;$('saveState').textContent='另一窗口有修改';notify('另一个页面修改了同一份进度，本页自动保存已暂停。请先下载本页草稿，再选择需要保留的版本并重新打开页面。',true);}});
  window.addEventListener('beforeunload',e=>{if(!localOK||conflict){e.preventDefault();e.returnValue='';}});
  function changed(){item().status='pending';review.completed_at_utc=null;$('reviewCheck').checked=false;store();refreshProgress();$('itemStatus').textContent='待完成';}
  function refreshProgress(){
    const done=review.items.filter(s=>s.status==='complete').length;
    $('progressText').textContent=`已完成 ${done} / ${review.items.length}`;$('progress').max=review.items.length;$('progress').value=done;
    $('sampleList').replaceChildren();
    const query=$('search').value.toLowerCase();
    review.items.forEach((s,i)=>{
      if(filter==='pending'&&s.status==='complete'||filter==='done'&&s.status!=='complete'||!`${s.id} ${s.text}`.toLowerCase().includes(query))return;
      const b=button('',()=>go(i),'sample-button'+(i===index?' active':''));
      b.setAttribute('aria-current',i===index?'true':'false');
      b.append(el('small','',`${s.status==='complete'?'✓ 已完成':'待办'} · ${i+1} · ${s.id}`),el('span','',s.text));$('sampleList').append(b);
    });
  }
  function setSelection(text){selected=text;$('addSelection').disabled=!text;$('selectionHint').textContent=text?`已选择：“${text}”`:'先选中一句或一个片段，再点击“用选中文字新增主张”。';}
  document.addEventListener('selectionchange',()=>{
    const s=window.getSelection(), source=$('source');
    if(s&&source.contains(s.anchorNode)&&source.contains(s.focusNode)&&s.toString())setSelection(s.toString());
  });
  function spanText(quote){const matches=C.spans(item().text,quote);return matches.length?`${matches.length===1?'引用已匹配':`引用重复 ${matches.length} 次，全部位置已记录`} · ${matches.map(s=>`[${s.start_idx}, ${s.end_idx})`).join('、')}`:'尚未匹配原文：请逐字选择或复制';}
  function newId(prefix,values,field){let n=1;while(values.some(v=>v[field]===prefix+n))n++;return prefix+n;}
  function addFact(quote=''){
    item().facts.push({fact_id:newId(packet.role.toLowerCase()+'_f',item().facts,'fact_id'),proposition:quote,quote,occurrences:C.spans(item().text,quote),context_quotes:[],qualifiers:[]});
    changed();renderFacts();$('facts').lastElementChild.querySelector('textarea').focus();
  }
  function field(container,label,value,oninput,id){
    const l=el('label','',label), t=el('textarea');l.htmlFor=id;t.id=id;t.value=value;t.addEventListener('input',()=>oninput(t.value));container.append(l,t);return t;
  }
  function renderFacts(){
    $('facts').replaceChildren();$('factCount').textContent=`${item().facts.length} 条`;$('emptyState').hidden=!!item().facts.length;
    item().facts.forEach((f,fi)=>{
      const card=el('section','fact');card.setAttribute('aria-label',`主张 ${fi+1}`);
      const top=el('div','fact-top');top.append(el('strong','',`主张 ${fi+1}`),button('删除主张',()=>confirmAction('删除这条主张？','这只删除当前主张，不会修改原文。请先确认它不是漏提项。',()=>{item().facts.splice(fi,1);changed();renderFacts();}),'text-button danger'));card.append(top);
      const grid=el('div','field-grid'), left=el('div'),right=el('div');grid.append(left,right);card.append(grid);
      const status=el('p','span-status',spanText(f.quote));
      function quoteChange(v){f.quote=v;f.occurrences=C.spans(item().text,v);status.textContent=spanText(v);status.classList.toggle('invalid',!f.occurrences.length);changed();}
      const quote=field(left,'原文引用（逐字复制）',f.quote,quoteChange,`quote-${fi}`);
      status.classList.toggle('invalid',!C.spans(item().text,f.quote).length);left.append(status);
      const qt=el('div','quote-tools');qt.append(button('使用选中的原文',()=>{if(!selected){notify('请先在上方原文中选择一个片段。',true);return;}quote.value=selected;quoteChange(selected);}));left.append(qt);
      field(right,'完整主张（可补全主语、指代）',f.proposition,v=>{f.proposition=v;changed();},`proposition-${fi}`);
      right.append(el('p','hint','补全必须有原文依据；保留否定、归因和必要宾语。'));
      const contexts=el('div','subsection'), ch=el('div','subhead'), list=el('div');
      ch.append(el('strong','','上下文依据（按需添加）'),button('添加选中的上下文',()=>{
        if(!selected){notify('请先在原文中选择代词或共享主语所依据的片段。',true);return;}
        if(f.context_quotes.some(c=>c.quote===selected)){notify('这段上下文已添加。');return;}
        f.context_quotes.push({quote:selected,occurrences:C.spans(item().text,selected)});changed();renderFacts();
      }));contexts.append(ch,el('p','hint','例如原文用了 She，可选中前文的人名作为补全依据。'),list);
      f.context_quotes.forEach((c,ci)=>{const row=el('div','context-row');const wrap=el('div');wrap.append(el('div','',c.quote),el('p','span-status',spanText(c.quote)));row.append(wrap,button('移除',()=>{f.context_quotes.splice(ci,1);changed();renderFacts();}));list.append(row);});card.append(contexts);
      const quals=el('div','subsection'), qh=el('div','subhead');qh.append(el('strong','','限定条件（没有则留空）'),button('添加限定条件',()=>{f.qualifiers.push({qualifier_id:newId('q',f.qualifiers,'qualifier_id'),kind:'time',text:''});changed();renderFacts();$('facts').children[fi].querySelector('.qualifier-row:last-child input').focus();}));quals.append(qh);
      f.qualifiers.forEach((q,qi)=>{
        const row=el('div','qualifier-row'), select=el('select'), input=el('input');select.setAttribute('aria-label',`主张 ${fi+1} 限定 ${qi+1} 类型`);
        for(const [value,label] of Object.entries(C.kinds)){const opt=el('option','',label);opt.value=value;select.append(opt);}select.value=q.kind;
        select.onchange=()=>{q.kind=select.value;changed();};input.value=q.text;input.placeholder='如：in June / exactly two / not';input.setAttribute('aria-label',`主张 ${fi+1} 限定 ${qi+1} 内容`);input.oninput=()=>{q.text=input.value;changed();};
        row.append(select,input,button('移除',()=>{f.qualifiers.splice(qi,1);changed();renderFacts();}));quals.append(row);
      });card.append(quals);$('facts').append(card);
    });
  }
  function render(){
    const s=item();$('source').textContent=s.text;$('sampleMeta').textContent=`样本 ${index+1} / ${review.items.length} · ${s.id}`;$('itemStatus').textContent=s.status==='complete'?'已完成':'待完成';$('reviewCheck').checked=s.status==='complete';$('positionText').textContent=`${index+1} / ${review.items.length}`;$('previousButton').disabled=index===0;$('nextButton').disabled=index===review.items.length-1;setSelection('');refreshProgress();renderFacts();
  }
  function go(i){index=i;store();render();document.querySelector('main').scrollTo({top:0});}
  function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json;charset=utf-8'}));const a=el('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);}
  $('roleBadge').textContent=`标注者 ${packet.role} · 独立初审`;$('reviewerName').value=review.reviewer_id||'';
  $('reviewerName').oninput=()=>{review.reviewer_id=$('reviewerName').value;review.completed_at_utc=null;store();};
  $('search').oninput=refreshProgress;
  for(const [id,value] of [['allFilter','all'],['pendingFilter','pending'],['doneFilter','done']])$(id).onclick=()=>{filter=value;for(const b of document.querySelectorAll('.filters button'))b.classList.toggle('active',b.id===id);refreshProgress();};
  $('previousButton').onclick=()=>go(index-1);$('nextButton').onclick=()=>go(index+1);$('addFact').onclick=()=>addFact();$('addSelection').onclick=()=>addFact(selected);
  $('completeButton').onclick=()=>{
    const errors=C.itemErrors(item());
    if(errors.length){notify(errors.join('\n'),true);return;}
    if(!$('reviewCheck').checked){notify('请完成本条内容复核，再勾选确认项。',true);return;}
    item().facts=C.normalize({items:[item()]}).items[0].facts;item().status='complete';review.completed_at_utc=null;store();render();notify('本条已完成。你可以继续下一条，也可以回来修改。');
  };
  $('pendingButton').onclick=()=>{changed();render();};
  $('draftButton').onclick=()=>{download({app:'minifact-review-ui',version:1,role:packet.role,bundle_sha256:packet.bundle_sha256,index,review:C.normalize(review)},`MiniFact_${packet.role}_草稿_${new Date().toISOString().replace(/[:.]/g,'-')}.json`);notify('已请求下载草稿，请确认浏览器下载成功。下次使用“恢复进度”导入它。');};
  $('exportButton').onclick=()=>{
    try{C.finalReview(review,new Date().toISOString());}catch(e){notify(e.message,true);return;}
    confirmAction('确认独立人工标注完成',`你以化名“${review.reviewer_id}”完成了全部 ${review.items.length} 条，且未参考另一位标注者的初稿。确认后生成回传文件。`,()=>{
      review=C.finalReview(review,new Date().toISOString());store();download(review,`reviewer_${packet.role.toLowerCase()}.json`);notify('已请求下载完成稿，请确认下载成功后，将文件回传协调者。页面不会自动发送文件。');
    });
  };
  $('importButton').onclick=()=>$('importFile').click();
  $('importFile').onchange=async()=>{
    const file=$('importFile').files[0];if(!file)return;
    try{
      if(file.size>5*1024*1024)throw Error('文件超过大小限制，请选择本页面导出的进度文件');
      const input=JSON.parse(await file.text());let candidate=input, next=0;
      if(input.app==='minifact-review-ui'){
        if(input.version!==1||input.role!==packet.role||input.bundle_sha256!==packet.bundle_sha256)throw Error('草稿属于另一位标注者或另一版本，请使用对应页面');
        candidate=input.review;next=Number.isInteger(input.index)?input.index:0;
      } else if(new RegExp(`reviewer_${packet.role==='A'?'b':'a'}`, 'i').test(file.name)) throw Error('这是另一位标注者的文件，请独立填写自己的页面');
      const incoming=C.validateImport(candidate,packet.template);
      confirmAction('恢复这份进度？','这会替换当前页面的进度。如果当前填写需要保留，请先取消并下载草稿。',()=>{
        review=incoming;index=Math.min(Math.max(next,0),review.items.length-1);conflict=false;restoreFailed=false;$('reviewerName').value=review.reviewer_id||'';store();render();notify('进度已恢复。');
      });
    }catch(e){notify('恢复失败：'+e.message,true);}finally{$('importFile').value='';}
  };
  $('guideButton').onclick=()=>$('guide').showModal();$('closeGuide').onclick=()=>$('guide').close();
  render();store();
})();
