"use strict";

// ---- APIエンドポイント -----------------------------------------------------
const API = {
  SESSION: "/api/session",
  SAVE: "/api/save",
  OPEN: "/api/open",
  HISTORY: "/api/history",
};

// サーバが index.html に埋め込んだ CSRF トークン（保存リクエストに付与する）
const CSRF_TOKEN = (() => {
  const meta = document.querySelector('meta[name="csrf-token"]');
  return meta ? meta.getAttribute("content") : "";
})();

// ---- 状態 -----------------------------------------------------------------
const state = {
  labels: [],
  segments: [],   // {id, phase_name, start_sec, end_sec}
  selectedId: null,
  dirty: false,
  procedure: "",
  procedures: [],
};

const video = document.getElementById("video");
if (!video) {
  throw new Error("必須要素 #video が見つかりません（HTMLの構造を確認してください）");
}

// ---- 時刻ユーティリティ（SRT: HH:MM:SS,mmm） -------------------------------
function fmt(sec) {
  sec = Math.max(0, sec || 0);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.round((sec - Math.floor(sec)) * 1000);
  const p = (n, w) => String(n).padStart(w, "0");
  return `${p(h, 2)}:${p(m, 2)}:${p(s, 2)},${p(Math.min(ms, 999), 3)}`;
}
function parseTime(str) {
  const m = String(str).trim().match(/^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})$/);
  if (!m) return null;
  return (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]) + (+m[4]) / 1000;
}
function randHex() {
  return Array.from({ length: 8 }, () =>
    Math.floor(Math.random() * 16).toString(16)).join("");
}

function setStatus(msg) { document.getElementById("status").textContent = msg; }
function markDirty() { state.dirty = true; setStatus("未保存の変更あり"); }

// ---- 初期ロード -----------------------------------------------------------
async function load() {
  let s;
  try {
    const r = await fetch(API.SESSION);
    if (!r.ok) { setStatus("セッション読込失敗: " + r.status); return; }
    s = await r.json();
  } catch (err) {
    setStatus("セッション読込エラー: " + (err && err.message ? err.message : err));
    return;
  }
  state.labels = s.labels;
  state.procedure = s.procedure || "";
  state.procedures = s.procedures || [];
  state.segments = s.segments.sort((a, b) => a.start_sec - b.start_sec);
  state.selectedId = null;
  state.dirty = false;
  document.getElementById("video-name").textContent = s.video_name;
  document.getElementById("mode-badge").textContent =
    s.is_new ? "新規作成" : "修正";
  video.src = s.video_url;
  renderPalette();
  renderAll();
  setStatus(`${state.segments.length} セグメント / 保存先: ${s.save_target}`);
}

// ---- ラベルパレット -------------------------------------------------------
function renderPalette() {
  const box = document.getElementById("label-buttons");
  box.innerHTML = "";
  state.labels.forEach((name, i) => {
    const b = document.createElement("button");
    b.className = "label-btn";
    // ラベル名はサーバ由来の任意文字列。innerHTML 連結を避け、
    // 番号は要素として、ラベル名は textContent で安全に挿入する（XSS対策）。
    if (i < 9) {
      const num = document.createElement("span");
      num.className = "num";
      num.textContent = String(i + 1);
      b.appendChild(num);
    }
    b.appendChild(document.createTextNode(name));
    b.onclick = () => assignLabel(name);
    box.appendChild(b);
  });
}

// ---- 描画 -----------------------------------------------------------------
function renderAll() { renderList(); renderTimeline(); renderEditor(); }

function renderList() {
  const ol = document.getElementById("seg-list");
  ol.innerHTML = "";
  state.segments.forEach((seg, i) => {
    const li = document.createElement("li");
    li.className = "seg-row" + (seg.id === state.selectedId ? " sel" : "");
    // phase_name はサーバ由来の任意文字列なので textContent で安全に挿入する。
    const mk = (cls, text) => {
      const sp = document.createElement("span");
      sp.className = cls;
      sp.textContent = text;
      return sp;
    };
    li.appendChild(mk("idx", String(i + 1)));
    li.appendChild(mk("name", seg.phase_name || "(未設定)"));
    li.appendChild(mk("time", `${fmt(seg.start_sec)} → ${fmt(seg.end_sec)}`));
    // キーボード操作で行を選択できるようフォーカス可能にする（アクセシビリティ）
    li.tabIndex = 0;
    li.setAttribute("role", "button");
    li.setAttribute("aria-selected", seg.id === state.selectedId ? "true" : "false");
    const activate = () => { select(seg.id); video.currentTime = seg.start_sec; };
    li.onclick = activate;
    li.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        e.stopPropagation();  // グローバルの再生/停止ハンドラとの二重発火を防ぐ
        activate();
      }
    };
    ol.appendChild(li);
  });
}

function renderTimeline() {
  const box = document.getElementById("timeline-segments");
  box.innerHTML = "";
  const dur = video.duration || lastEnd() || 1;
  state.segments.forEach(seg => {
    const d = document.createElement("div");
    d.className = "tl-seg" + (seg.id === state.selectedId ? " sel" : "");
    d.style.left = (100 * seg.start_sec / dur) + "%";
    d.style.width = Math.max(0.4, 100 * (seg.end_sec - seg.start_sec) / dur) + "%";
    d.textContent = seg.phase_name || "";
    d.title = `${seg.phase_name}  ${fmt(seg.start_sec)}→${fmt(seg.end_sec)}`;
    // キーボード/スクリーンリーダーでも区間を選択・シークできるようにする
    d.tabIndex = 0;
    d.setAttribute("role", "listitem");
    d.setAttribute("aria-label",
      `${seg.phase_name || "(未設定)"} ${fmt(seg.start_sec)} から ${fmt(seg.end_sec)}`);
    d.setAttribute("aria-selected", seg.id === state.selectedId ? "true" : "false");
    const activate = (e) => { e.stopPropagation(); select(seg.id); video.currentTime = seg.start_sec; };
    d.onclick = activate;
    d.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(e); }
    };
    box.appendChild(d);
  });
}

function renderEditor() {
  const seg = selected();
  document.getElementById("sel-label").textContent = seg ? (seg.phase_name || "—") : "—";
  const start = document.getElementById("sel-start");
  const end = document.getElementById("sel-end");
  start.value = seg ? fmt(seg.start_sec) : "";
  end.value = seg ? fmt(seg.end_sec) : "";
  clearFieldError(start);
  clearFieldError(end);
}

// 不正入力の視覚的フィードバック（赤枠＋ステータス＋aria-invalid）
function fieldError(input, msg) {
  input.classList.add("invalid");
  input.setAttribute("aria-invalid", "true");
  setStatus("入力エラー: " + msg);
}
function clearFieldError(input) {
  input.classList.remove("invalid");
  input.removeAttribute("aria-invalid");
}

function lastEnd() {
  return state.segments.reduce((m, s) => Math.max(m, s.end_sec), 0);
}

// ---- 選択 / 編集 ----------------------------------------------------------
function selected() { return state.segments.find(s => s.id === state.selectedId) || null; }
function select(id) { state.selectedId = id; renderAll(); }

// 再生ヘッドを前/後ろのセグメント開始位置へジャンプし、その区間を選択する。
// dir > 0: 次の区間、dir < 0: 前の区間。境界の取りこぼしを防ぐため小さなεを使う。
function jumpSegment(dir) {
  if (!state.segments.length) return;
  const eps = 0.05;
  const t = video.currentTime;
  const sorted = state.segments.slice().sort((a, b) => a.start_sec - b.start_sec);
  let target = null;
  if (dir > 0) {
    target = sorted.find(s => s.start_sec > t + eps) || null;
  } else {
    const before = sorted.filter(s => s.start_sec < t - eps);
    target = before.length ? before[before.length - 1] : null;
  }
  if (!target) { setStatus(dir > 0 ? "最後の区間です" : "最初の区間です"); return; }
  video.currentTime = target.start_sec;
  select(target.id);
}

function assignLabel(name) {
  const seg = selected();
  if (!seg) { setStatus("セグメント未選択"); return; }
  seg.phase_name = name;
  markDirty();
  renderAll();
}

function snap(field) {
  const seg = selected();
  if (!seg) return;
  const t = video.currentTime;
  if (field === "start") seg.start_sec = Math.min(t, seg.end_sec);
  else seg.end_sec = Math.max(t, seg.start_sec);
  markDirty();
  renderAll();
}

function insertAtPlayhead() {
  const t = video.currentTime;
  // 次のセグメント開始（または動画末尾）まで、なければ +5秒
  const nextStart = state.segments
    .map(s => s.start_sec).filter(x => x > t).sort((a, b) => a - b)[0];
  const end = nextStart !== undefined ? nextStart : Math.min(t + 5, video.duration || t + 5);
  const seg = {
    id: randHex(),
    phase_name: state.labels[0] || "",
    start_sec: t,
    end_sec: Math.max(end, t + 0.1),
  };
  state.segments.push(seg);
  state.segments.sort((a, b) => a.start_sec - b.start_sec);
  state.selectedId = seg.id;
  markDirty();
  renderAll();
}

function deleteSelected() {
  if (!state.selectedId) return;
  state.segments = state.segments.filter(s => s.id !== state.selectedId);
  state.selectedId = null;
  markDirty();
  renderAll();
}

// ---- 保存 -----------------------------------------------------------------
async function save() {
  setStatus("保存中…");
  const payload = {
    segments: state.segments.map(s => ({
      id: s.id,
      phase_name: s.phase_name || "",
      start_sec: s.start_sec,
      end_sec: s.end_sec,
    })),
  };
  let res;
  try {
    const r = await fetch(API.SAVE, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": CSRF_TOKEN,
      },
      body: JSON.stringify(payload),
    });
    if (!r.ok) { setStatus("保存失敗: " + r.status); return; }
    res = await r.json();
  } catch (err) {
    setStatus("保存エラー: " + (err && err.message ? err.message : err));
    return;
  }
  state.dirty = false;
  setStatus(
    `保存完了: ${res.saved} セグメント / 今回の変更 ${res.changes} 件 / DPOペア ${res.pairs} 件`);
  await loadHistory();  // 履歴ダイアログが開いていれば最新化（順序を保つため await）
}

// ---- フィールド入力 -------------------------------------------------------
document.getElementById("sel-start").addEventListener("change", (e) => {
  const seg = selected(); const v = parseTime(e.target.value);
  if (!seg) { renderEditor(); return; }
  if (v === null) {
    fieldError(e.target, "形式が不正です（HH:MM:SS,mmm）");
    return;
  }
  if (v > seg.end_sec) {
    fieldError(e.target, "開始時刻が終了時刻を超えています");
    return;
  }
  clearFieldError(e.target);
  seg.start_sec = v; markDirty(); renderAll();
});
document.getElementById("sel-end").addEventListener("change", (e) => {
  const seg = selected(); const v = parseTime(e.target.value);
  if (!seg) { renderEditor(); return; }
  if (v === null) {
    fieldError(e.target, "形式が不正です（HH:MM:SS,mmm）");
    return;
  }
  if (v < seg.start_sec) {
    fieldError(e.target, "終了時刻が開始時刻を下回っています");
    return;
  }
  clearFieldError(e.target);
  seg.end_sec = v; markDirty(); renderAll();
});
document.querySelectorAll(".snap").forEach(b =>
  b.addEventListener("click", () => snap(b.dataset.field)));
document.getElementById("insert-btn").onclick = insertAtPlayhead;
document.getElementById("delete-btn").onclick = deleteSelected;
document.getElementById("save-btn").onclick = save;

// ---- タイムライン / 再生位置 ----------------------------------------------
document.getElementById("timeline").addEventListener("click", (e) => {
  const rect = e.currentTarget.getBoundingClientRect();
  const ratio = (e.clientX - rect.left) / rect.width;
  video.currentTime = ratio * (video.duration || lastEnd());
});
video.addEventListener("timeupdate", () => {
  const dur = video.duration || lastEnd() || 1;
  document.getElementById("time-cur").textContent = fmt(video.currentTime);
  document.getElementById("timeline-cursor").style.left =
    (100 * video.currentTime / dur) + "%";
  const tl = document.getElementById("timeline");
  tl.setAttribute("aria-valuenow", String(Math.round(video.currentTime)));
  tl.setAttribute("aria-valuetext", fmt(video.currentTime));
});
video.addEventListener("loadedmetadata", () => {
  document.getElementById("time-dur").textContent = "/ " + fmt(video.duration);
  const dur = video.duration || 0;
  document.getElementById("timeline").setAttribute(
    "aria-valuemax", String(Math.round(dur)));
  renderTimeline();
});

// ---- 動画/SRT を開く ------------------------------------------------------
const openDialog = document.getElementById("open-dialog");

function showOpenDialog() {
  // 術式ドロップダウンを最新のセッション情報から構築
  const sel = document.getElementById("open-procedure");
  sel.innerHTML = "";
  (state.procedures.length ? state.procedures : [state.procedure]).forEach(p => {
    const o = document.createElement("option");
    o.value = p; o.textContent = p;
    if (p === state.procedure) o.selected = true;
    sel.appendChild(o);
  });
  if (typeof openDialog.showModal === "function") openDialog.showModal();
  else openDialog.setAttribute("open", "");
}

async function openSession(e) {
  if (e) e.preventDefault();
  const videoPath = document.getElementById("open-video").value.trim();
  if (!videoPath) { setStatus("動画パスを入力してください"); return; }
  if (state.dirty &&
      !confirm("未保存の変更があります。破棄して別ファイルを開きますか？")) {
    return;
  }
  const srtPath = document.getElementById("open-srt").value.trim();
  const procedure = document.getElementById("open-procedure").value;
  setStatus("読み込み中…（プロキシ生成に時間がかかる場合があります）");
  try {
    const r = await fetch(API.OPEN, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": CSRF_TOKEN },
      body: JSON.stringify({
        video: videoPath,
        srt: srtPath || null,
        procedure: procedure || null,
      }),
    });
    if (!r.ok) {
      let detail = r.status;
      try { detail = (await r.json()).detail || r.status; } catch (_) {}
      setStatus("読込失敗: " + detail);
      return;
    }
  } catch (err) {
    setStatus("読込エラー: " + (err && err.message ? err.message : err));
    return;
  }
  openDialog.close();
  await load();        // 新しい編集対象でセッションを再構築
  await loadHistory(); // 履歴も対象に合わせて更新
}

document.getElementById("open-btn").onclick = showOpenDialog;
document.getElementById("open-cancel").onclick = () => openDialog.close();
document.getElementById("open-form").addEventListener("submit", openSession);

// ---- 修正履歴 -------------------------------------------------------------
const historyDialog = document.getElementById("history-dialog");

const CHANGE_KINDS = ["edited", "inserted", "deleted"];

function summarizeChanges(changes) {
  // プロトタイプ汚染を避けるため null プロトタイプの辞書で集計する。
  const c = Object.create(null);
  CHANGE_KINDS.forEach(k => { c[k] = 0; });
  // hasOwnProperty.call は ES2022 の Object.hasOwn 非対応環境でも動く。
  const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  (changes || []).forEach(p => {
    if (has(c, p.change)) c[p.change]++;
  });
  return c;
}

async function loadHistory() {
  let data;
  try {
    // /api/history は CSRF トークン必須（動画名・差分などを含むため）
    const r = await fetch(API.HISTORY, { headers: { "X-CSRF-Token": CSRF_TOKEN } });
    if (!r.ok) {
      setStatus(`履歴の取得に失敗しました (HTTP ${r.status})`);
      return;
    }
    data = await r.json();
  } catch (err) {
    setStatus("履歴の取得エラー: " + (err && err.message ? err.message : err));
    return;
  }
  document.getElementById("history-file").textContent =
    `${data.history_file}（${data.entries.length} 回保存）`;
  const ol = document.getElementById("history-list");
  ol.innerHTML = "";
  // 新しい保存を上に表示
  data.entries.slice().reverse().forEach(ent => {
    const c = summarizeChanges(ent.changes);
    const li = document.createElement("li");
    li.className = "hist-row";
    const head = document.createElement("div");
    head.className = "hist-head";
    head.textContent =
      `${ent.timestamp}  ・ 変更${ent.n_changes}件（編集${c.edited}/追加${c.inserted}/削除${c.deleted}）`;
    li.appendChild(head);
    // 各変更の詳細（ラベル・時刻の前後）
    (ent.changes || []).forEach(p => {
      const d = document.createElement("div");
      d.className = "hist-change " + p.change;
      d.textContent = describeChange(p);
      li.appendChild(d);
    });
    ol.appendChild(li);
  });
}

function describeChange(p) {
  const r = p.rejected, c = p.chosen;
  const seg = s => s ? `${s.phase_name} [${fmt(s.start_sec)}→${fmt(s.end_sec)}]` : "—";
  // 既知タイプを明示的に分岐し、未知タイプは「編集」に丸めず可視化する。
  switch (p.change) {
    case "inserted": return `＋追加  ${seg(c)}`;
    case "deleted":  return `－削除  ${seg(r)}`;
    case "edited":   return `±編集  ${seg(r)}  ⇒  ${seg(c)}`;
    default:         return `? 不明(${p.change})  ${seg(r)}  ⇒  ${seg(c)}`;
  }
}

async function showHistoryDialog() {
  await loadHistory();
  if (typeof historyDialog.showModal === "function") historyDialog.showModal();
  else historyDialog.setAttribute("open", "");
}

document.getElementById("history-btn").onclick = showHistoryDialog;
document.getElementById("history-close").onclick = () => historyDialog.close();

// ---- キーボード -----------------------------------------------------------
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;  // 数値入力中は無効
  if (document.querySelector("dialog[open]")) return;  // ダイアログ表示中は無効
  if (e.ctrlKey && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
  switch (e.key) {
    case " ": e.preventDefault(); video.paused ? video.play() : video.pause(); break;
    case "ArrowLeft":
      e.preventDefault();
      if (e.shiftKey) jumpSegment(-1);
      else video.currentTime = Math.max(0, video.currentTime - 1);
      break;
    case "ArrowRight":
      e.preventDefault();
      if (e.shiftKey) jumpSegment(1);
      else video.currentTime += 1;
      break;
    case "i": case "I": snap("start"); break;
    case "o": case "O": snap("end"); break;
    case "n": case "N": insertAtPlayhead(); break;
    case "Delete": case "Backspace": deleteSelected(); break;
    default:
      if (/^[1-9]$/.test(e.key)) {
        const idx = (+e.key) - 1;
        if (idx < state.labels.length) assignLabel(state.labels[idx]);
      }
  }
});

window.addEventListener("beforeunload", (e) => {
  if (state.dirty) { e.preventDefault(); e.returnValue = ""; }
});

load();
