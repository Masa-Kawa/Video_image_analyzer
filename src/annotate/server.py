"""
アノテーション編集Webツールのバックエンド（FastAPI）。

動画プロキシ（ブラウザ再生用）とフェーズ語彙、編集対象セグメントを
ブラウザUIに提供し、保存時に安定ID付きSRTとDPOペアJSONLを書き出す。

起動例:
    python -m src.annotate.server \
        --video case001.mp4 \
        --srt out/case001_phase.srt \
        --procedure cholecystectomy \
        --port 8000

--srt を省略すると新規作成モード（空セグメント）。
既定の保存先は <video_stem>_gold.srt（元SRTは保持）、
ペアは <video_stem>_dpo_pairs.jsonl。
"""

import argparse
import datetime
import json
import secrets
import threading
import webbrowser
from collections import deque
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.annotate.label_sets import (
    DEFAULT_PROCEDURE,
    available_procedures,
    get_label_set,
)
from src.annotate.pairing import make_pairs
from src.annotate.srt_io import load_segments, save_segments
from src.tools.proxy_manager import ProxyManager

WEB_DIR = Path(__file__).parent / "web"
# ブラウザ再生用プロキシ解像度（ProxyManager.RESOLUTIONS のキー）
PROXY_RESOLUTION = "480p"

# プロキシ動画は元動画の拡張子を引き継ぐため、拡張子から MIME を引く。
_VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}


def _video_media_type(path: Path) -> str:
    """プロキシ動画の拡張子に応じた MIME タイプ（不明時は mp4 にフォールバック）。"""
    return _VIDEO_MIME.get(path.suffix.lower(), "video/mp4")


class AppState:
    """1セッション分の編集対象（動画・元SRT・保存先）を保持する。"""

    def __init__(
        self,
        video: Path,
        srt: Optional[Path],
        procedure: str,
        proxy: Path,
        save_target: Path,
        pairs_target: Path,
        history_target: Path,
    ):
        self.video = video
        self.srt = srt
        self.procedure = procedure
        self.proxy = proxy
        self.save_target = save_target
        self.pairs_target = pairs_target
        # 保存ごとの差分（前回保存→今回）を追記する修正履歴ログ
        self.history_target = history_target
        # 元セグメント（採番済み）をメモリに保持しDPOペアリングの基準にする
        self.original_segments = load_segments(srt) if srt else []
        # 修正履歴の基準＝このセッションで最後に保存した状態（初期値は開いた内容）。
        # 保存先に無関係な古い gold が残っていてもそれに引きずられないよう、
        # ディスクではなくメモリのスナップショットを基準にする。
        self.last_saved = list(self.original_segments)


class SegmentIn(BaseModel):
    id: Optional[str] = None
    phase_name: str
    start_sec: float
    end_sec: float


class SaveRequest(BaseModel):
    segments: List[SegmentIn]


class OpenRequest(BaseModel):
    """UIから別の動画/SRTを開くためのリクエスト（サーバ側パスを指定）。"""
    video: str
    srt: Optional[str] = None
    procedure: Optional[str] = None
    save_target: Optional[str] = None


def _to_dict(model: BaseModel) -> dict:
    """Pydantic v1/v2 両対応で BaseModel を dict 化する。

    v2 の ``model_dump`` を優先し、無ければ v1 の ``dict`` にフォールバック。
    依存解決で Pydantic v1 が入った環境でも保存機能が壊れないようにする。
    """
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _require_csrf(token: Optional[str], expected: str) -> None:
    """CSRFトークンを検証する。不正/欠落なら 403。"""
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="invalid or missing CSRF token")


def _read_history(path: Path, limit: Optional[int] = None) -> List[dict]:
    """修正履歴JSONLを読み込む（無ければ空、壊れた行はスキップ）。

    ``limit`` 指定時は末尾（最新）``limit`` 件のみを保持する。deque(maxlen) で
    メモリ上に抱える件数を上限に抑えるため、履歴が巨大化しても OOM を避けられる。
    """
    if not path.exists():
        return []
    dq: deque = deque(maxlen=limit) if limit and limit > 0 else deque()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dq.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(dq)


def _proxy_version(proxy: Path) -> str:
    """プロキシ動画のキャッシュ無効化トークン（更新時刻ベースで一意）。"""
    try:
        return str(proxy.stat().st_mtime_ns)
    except OSError:
        return "0"


def _resolve_within(base: Optional[Path], path: Path) -> Path:
    """``path`` を絶対パスに正規化して返す。

    ``base`` が指定された場合、正規化後のパスが ``base`` 配下に収まることを
    検証し、外側（例: ``../../../etc/passwd``）なら ValueError を送出する。
    パストラバーサルやベースディレクトリ外アクセスを防ぐための関門。
    """
    resolved = path.expanduser().resolve()
    if base is not None:
        base = base.resolve()
        if resolved != base and base not in resolved.parents:
            raise ValueError(f"許可ディレクトリ外のパスです: {path}")
    return resolved


def create_app(state: AppState, base_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="Surgical Annotation Editor")

    # /api/open で開けるファイルの許可ベースディレクトリ（None なら無制限）。
    # ブラウザ由来の任意パスでサーバ上の任意ファイルを読み書きされるのを防ぐ。
    allowed_base = base_dir.resolve() if base_dir is not None else None

    # 編集対象は実行中に /api/open で差し替えられるため、可変ホルダに包む。
    # 各エンドポイントは cur() で「現在の編集対象」を参照する。
    holder = {"state": state}

    def cur() -> AppState:
        return holder["state"]

    # サーバ起動ごとに生成する CSRF トークン。index.html に埋め込み、
    # /api/save・/api/open では同一トークンのヘッダ提示を必須にして
    # 同一オリジン由来のリクエストのみ許可する（外部サイトはトークンを読めない）。
    csrf_token = secrets.token_urlsafe(32)

    # 保存/読込（SRT/JSONL 書き出し・状態差し替え）の排他制御。複数タブからの
    # 同時操作によるファイル破損や状態競合を防ぐため IO を直列化する。
    save_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        return html.replace("__CSRF_TOKEN__", csrf_token)

    @app.get("/app.js")
    def app_js() -> FileResponse:
        return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")

    @app.get("/style.css")
    def style_css() -> FileResponse:
        return FileResponse(WEB_DIR / "style.css", media_type="text/css")

    @app.get("/api/session")
    def session() -> JSONResponse:
        st = cur()
        return JSONResponse({
            "procedure": st.procedure,
            "procedures": available_procedures(),
            "labels": get_label_set(st.procedure),
            "segments": [
                {
                    "id": s["id"],
                    "phase_name": s.get("phase_name", s.get("label", "")),
                    "start_sec": s.get("start_sec", 0.0),
                    "end_sec": s.get("end_sec", 0.0),
                }
                for s in st.original_segments
            ],
            # 同一動画でも /api/open 後にプロキシが変わりうるため、
            # キャッシュ無効化用のトークンを付けて再読込を促す。プロキシの更新時刻
            # (mtime_ns) を使い、同名・別ディレクトリや再生成でも確実に無効化する。
            "video_url": f"/media/video?v={_proxy_version(st.proxy)}",
            "video_name": st.video.name,
            "save_target": st.save_target.name,
            "is_new": st.srt is None,
        })

    @app.get("/media/video")
    def media_video() -> FileResponse:
        # FileResponse は Range リクエスト（206）に対応し、ブラウザのシークを可能にする。
        # media_type を明示し、拡張子推測に依存せず正しくシーク/バッファできるようにする。
        st = cur()
        if not st.proxy.exists():
            raise HTTPException(status_code=404, detail="proxy not found")
        return FileResponse(st.proxy, media_type=_video_media_type(st.proxy))

    @app.get("/api/history")
    def history(
        limit: int = 200,
        x_csrf_token: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        """現在の編集対象の修正履歴（保存ごとの差分）を最新 ``limit`` 件返す。

        動画名・保存先・編集差分を含むため、状態変更系と同様に CSRF トークンを
        必須にして同一オリジン由来の取得のみ許可する。
        """
        _require_csrf(x_csrf_token, csrf_token)
        st = cur()
        limit = max(1, min(limit, 1000))  # 異常値を無害な範囲に丸める
        return JSONResponse({
            "history_file": st.history_target.name,
            "limit": limit,
            "entries": _read_history(st.history_target, limit=limit),
        })

    @app.post("/api/open")
    def open_target(
        req: OpenRequest,
        x_csrf_token: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        """別の動画/SRTを開いて編集対象を差し替える。

        保存済みの ``_gold.srt`` を ``srt`` に指定すれば、安定IDを保ったまま
        過去の修正を再編集できる。ファイルパスは許可ベースディレクトリ配下に
        限定され、配下外（パストラバーサル）は 400 で拒否する。
        """
        _require_csrf(x_csrf_token, csrf_token)
        procedure = req.procedure or DEFAULT_PROCEDURE
        try:
            with save_lock:
                new_state = build_state(
                    req.video, req.srt, procedure, req.save_target,
                    base_dir=allowed_base,
                )
                holder["state"] = new_state
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse({
            "ok": True,
            "video_name": new_state.video.name,
            "save_target": new_state.save_target.name,
            "segments": len(new_state.original_segments),
            "is_new": new_state.srt is None,
        })

    @app.post("/api/save")
    def save(
        req: SaveRequest,
        x_csrf_token: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        _require_csrf(x_csrf_token, csrf_token)
        st = cur()

        segments = [_to_dict(s) for s in req.segments]

        # 重複 id を拒否（後勝ちでペアリング結果がセグメント数と不一致になるのを防ぐ）
        ids = [s["id"] for s in segments if s.get("id")]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate segment id(s): {', '.join(dupes)}",
            )

        for seg in segments:
            seg["type"] = "surgical_phase"
            seg["label"] = seg["phase_name"]

        # 複数タブからの同時保存による SRT/JSONL 破損を防ぐため直列化する
        with save_lock:
            # 履歴用の基準＝このセッションで最後に保存した状態（初期値は開いた内容）。
            prev_segments = st.last_saved

            n = save_segments(segments, st.save_target)

            # 保存後のセグメント（採番が確定したもの）を読み戻してペアリング
            corrected = load_segments(st.save_target)

            # DPOペア: 元（自動生成 = rejected）に対する最終形（chosen）。毎回上書き。
            pairs = make_pairs(
                st.original_segments,
                corrected,
                procedure=st.procedure,
                video=st.video.name,
            )
            with open(st.pairs_target, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            # 修正履歴: 前回保存→今回の差分を1エントリとして追記（append-only）。
            changes = make_pairs(
                prev_segments,
                corrected,
                procedure=st.procedure,
                video=st.video.name,
            )
            # 変更が無い保存（誤操作の Ctrl+S 等）は履歴を汚さないよう記録しない。
            if changes:
                history_entry = {
                    # タイムゾーン非依存の UTC で記録（DST/環境差で順序が乱れないよう）。
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(timespec="seconds"),
                    "video": st.video.name,
                    "save_target": st.save_target.name,
                    "saved": n,
                    "n_changes": len(changes),
                    "changes": changes,
                }
                with open(st.history_target, "a", encoding="utf-8") as f:
                    f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")

            # 次回保存の基準更新は全ファイル書き込みの成功後に行う。途中で I/O が
            # 失敗した場合は last_saved を進めず、次回保存で同じ差分を再記録できる
            # ようにして「SRTは保存済みだが履歴が欠損」の不整合を避ける。
            st.last_saved = corrected

        return JSONResponse({
            "saved": n,
            "srt": str(st.save_target),
            "pairs": len(pairs),
            "pairs_file": str(st.pairs_target),
            "changes": len(changes),
            "history_file": str(st.history_target),
        })

    return app


def build_state(
    video: str,
    srt: Optional[str],
    procedure: str,
    save_target: Optional[str],
    base_dir: Optional[Path] = None,
) -> AppState:
    # パスは正規化し、base_dir 指定時は配下に限定（パストラバーサル防止）。
    if not video:
        raise ValueError("動画パスが空です")
    video_path = _resolve_within(base_dir, Path(video))
    if not video_path.is_file():
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    # 語彙を先に検証（未登録術式なら即エラー）
    get_label_set(procedure)

    srt_path = _resolve_within(base_dir, Path(srt)) if srt else None
    if srt_path and not srt_path.is_file():
        raise FileNotFoundError(f"SRTが見つかりません: {srt_path}")

    if save_target:
        save_path = _resolve_within(base_dir, Path(save_target))
    else:
        save_path = video_path.with_name(f"{video_path.stem}_gold.srt")

    # ブラウザ再生用プロキシを取得/生成（入力検証を通過した後に実行）
    pm = ProxyManager()
    if not pm.proxy_exists(str(video_path), PROXY_RESOLUTION):
        print(f"[proxy] 生成中（{PROXY_RESOLUTION}）: {video_path.name} ...")
        proxy = pm.create_proxy(str(video_path), resolution=PROXY_RESOLUTION)
    else:
        proxy = pm.get_proxy_path(str(video_path), PROXY_RESOLUTION)
    print(f"[proxy] {proxy}")
    pairs_path = save_path.with_name(f"{save_path.stem}_dpo_pairs.jsonl")
    history_path = save_path.with_name(f"{save_path.stem}_history.jsonl")

    return AppState(
        video=video_path,
        srt=srt_path,
        procedure=procedure,
        proxy=proxy,
        save_target=save_path,
        pairs_target=pairs_path,
        history_target=history_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="手術動画アノテーション編集Webツール（フェーズSRT）"
    )
    parser.add_argument("--video", required=True, help="対象動画ファイル")
    parser.add_argument("--srt", default=None,
                        help="編集対象の元SRT（省略で新規作成モード）")
    parser.add_argument("--procedure", default=DEFAULT_PROCEDURE,
                        help="術式識別子（フェーズ語彙の選択）")
    parser.add_argument("--save-target", default=None,
                        help="保存先SRT（省略時は <video>_gold.srt）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="起動時にブラウザを開かない")
    parser.add_argument("--base-dir", default=".",
                        help="UI(/api/open)から開けるファイルの許可ベースディレクトリ"
                             "（既定: カレントディレクトリ）")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    # 起動時のCLI指定は明示的なので確認のみ。配下外なら base を広げて受け入れる。
    state = build_state(args.video, args.srt, args.procedure, args.save_target)
    app = create_app(state, base_dir=base_dir)

    url = f"http://{args.host}:{args.port}/"
    print(f"[server] {url}")
    print(f"[server] 保存先: {state.save_target}")
    print(f"[server] ペア : {state.pairs_target}")
    print(f"[server] 履歴 : {state.history_target}")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
