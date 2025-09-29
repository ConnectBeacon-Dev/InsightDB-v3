// static/askme-modal.js
(function () {
  const S = document.currentScript || (function(){const s=document.getElementsByTagName('script');return s[s.length-1];})();
  const CHAT_URL = S.getAttribute('data-chat-url') || '/chat';
  const LABEL    = S.getAttribute('data-label') || 'Ask Me';
  const THEME    = S.getAttribute('data-theme') || '#12a150';
  const POS      = S.getAttribute('data-position') || 'right';

  const HEADER_MODE = (S.getAttribute('data-header') || 'compact').toLowerCase(); // 'compact' | 'normal'
  const MODAL_W     = parseInt(S.getAttribute('data-modal-width')  || '1000', 10);
  const MODAL_H     = parseInt(S.getAttribute('data-modal-height') || '720', 10);

  const isCompact   = HEADER_MODE === 'compact';
  const headerPad   = isCompact ? 6  : 10;
  const headerH     = isCompact ? 36 : 44;
  const fontSize    = isCompact ? 13 : 14;
  const btnPad      = isCompact ? '0 6px' : '2px 8px';
  const btnFont     = isCompact ? 12 : 14;

  const css = document.createElement('style');
  css.textContent = `
  .askme-fab { position:fixed; ${POS}:20px; bottom:20px; z-index:2147483647;
    display:inline-flex; align-items:center; gap:8px; background:${THEME}; color:#fff;
    border:none; border-radius:999px; padding:12px 14px; font:600 14px/1.1 system-ui; box-shadow:0 8px 24px rgba(0,0,0,.18); cursor:pointer; }
  .askme-fab:hover { filter:brightness(1.05); }

  .askme-backdrop { position:fixed; inset:0; background:rgba(15,23,42,.5); z-index:2147483646; display:none; }
  .askme-modal { position:fixed; inset:auto; ${POS}:24px; bottom:90px;
    width:min(100vw - 48px, ${MODAL_W}px);
    height:min(100vh - 140px, ${MODAL_H}px);
    background:#fff; border-radius:14px; overflow:hidden;
    box-shadow:0 16px 48px rgba(0,0,0,.35); display:none; z-index:2147483647; }

  .askme-modal header { background:${THEME}; color:#fff; padding:${headerPad}px 12px;
    display:flex; justify-content:space-between; align-items:center; }
  .askme-modal header .left { display:flex; align-items:center; gap:8px; }
  .askme-modal header h3 { margin:0; font:700 ${fontSize}px/1.2 system-ui; }

  .askme-modal header .mic-ind {
    width:10px; height:10px; border-radius:999px; background:rgba(255,255,255,.35);
    box-shadow:0 0 0 0 rgba(239,68,68,0); transition:background .15s ease, box-shadow .15s ease;
  }
  .askme-modal header .mic-ind.live {
    background:#ef4444;
    animation: micpulse 1.2s infinite;
  }
  @keyframes micpulse {
    0% { box-shadow:0 0 0 0 rgba(239,68,68,.6) }
    70%{ box-shadow:0 0 0 10px rgba(239,68,68,0) }
    100%{ box-shadow:0 0 0 0 rgba(239,68,68,0) }
  }

  .askme-modal header .askme-actions { display:flex; align-items:center; }
  .askme-modal header .askme-actions button {
    background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25); color:#fff;
    border-radius:8px; margin-left:8px; padding:${btnPad}; font:${btnFont}px/1.2 system-ui; cursor:pointer; }
  .askme-modal header .askme-close { background:transparent; border:none; color:#fff; font-size:${isCompact?18:20}px; cursor:pointer; margin-left:8px; }

  .askme-iframe { width:100%; height:calc(100% - ${headerH}px); border:none; }
  `;
  document.head.appendChild(css);

  const btn = document.createElement('button'); btn.className = 'askme-fab'; btn.textContent = LABEL;
  const backdrop = document.createElement('div'); backdrop.className = 'askme-backdrop';
  const modal = document.createElement('div'); modal.className = 'askme-modal';

  modal.innerHTML = `
    <header>
      <div class="left">
        <h3>${LABEL}</h3>
        <span class="mic-ind" title="Mic status"></span>
      </div>
      <div class="askme-actions">
        <button type="button" class="askme-mic-toggle" title="Toggle voice input">🎤</button>
        <button class="askme-close" aria-label="Close">✕</button>
      </div>
    </header>
    <iframe class="askme-iframe" src="${CHAT_URL}" title="Ask Me"></iframe>`;

  const closeBtn = modal.querySelector('.askme-close');
  const micToggleBtn = modal.querySelector('.askme-mic-toggle');
  const micInd = modal.querySelector('.mic-ind');
  const iframe = modal.querySelector('.askme-iframe');

  const close = () => { modal.style.display='none'; backdrop.style.display='none'; };
  closeBtn.addEventListener('click', close);
  backdrop.addEventListener('click', close);
  btn.addEventListener('click', () => { backdrop.style.display='block'; modal.style.display='block'; });

  // reflect mic state from iframe
  window.addEventListener('message', (ev)=>{
    try{
      const src = new URL(iframe.src, window.location.href).origin;
      if (ev.origin !== src) return;
    }catch {}
    const d = ev.data || {};
    if (d.type === 'stt' && d.status){
      if (d.status === 'start') micInd.classList.add('live');
      if (d.status === 'stop' || d.status === 'error') micInd.classList.remove('live');
    }
  });

  // header mic button toggles STT in iframe
  micToggleBtn.addEventListener('click', ()=>{
    try { iframe.contentWindow.postMessage({ type:'stt:toggle' }, '*'); } catch {}
  });

  document.body.appendChild(btn);
  document.body.appendChild(backdrop);
  document.body.appendChild(modal);
})();
