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
import json
import secrets
import threading
import webbrowser
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.annotate.label_sets import DEFAULT_PROCEDURE, get_label_set
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
    ):
        self.video = video
        self.srt = srt
        self.procedure = procedure
        self.proxy = proxy
        self.save_target = save_target
        self.pairs_target = pairs_target
        # 元セグメント（採番済み）をメモリに保持しペアリングの基準にする
        self.original_segments = load_segments(srt) if srt else []


class SegmentIn(BaseModel):
    id: Optional[str] = None
    phase_name: str
    start_sec: float
    end_sec: float


class SaveRequest(BaseModel):
    segments: List[SegmentIn]


def _to_dict(model: BaseModel) -> dict:
    """Pydantic v1/v2 両対応で BaseModel を dict 化する。

    v2 の ``model_dump`` を優先し、無ければ v1 の ``dict`` にフォールバック。
    依存解決で Pydantic v1 が入った環境でも保存機能が壊れないようにする。
    """
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Surgical Annotation Editor")

    # サーバ起動ごとに生成する CSRF トークン。index.html に埋め込み、
    # /api/save では同一トークンのヘッダ提示を必須にして同一オリジン由来の
    # リクエストのみ許可する（外部サイトはトークンを読めない）。
    csrf_token = secrets.token_urlsafe(32)

    # 保存処理（SRT/JSONL 書き出し）の排他制御。複数タブからの同時保存による
    # ファイル破損を防ぐためエンドポイント全体の IO を直列化する。
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
        return JSONResponse({
            "procedure": state.procedure,
            "labels": get_label_set(state.procedure),
            "segments": [
                {
                    "id": s["id"],
                    "phase_name": s.get("phase_name", s.get("label", "")),
                    "start_sec": s.get("start_sec", 0.0),
                    "end_sec": s.get("end_sec", 0.0),
                }
                for s in state.original_segments
            ],
            "video_url": "/media/video",
            "video_name": state.video.name,
            "save_target": state.save_target.name,
            "is_new": state.srt is None,
        })

    @app.get("/media/video")
    def media_video() -> FileResponse:
        # FileResponse は Range リクエスト（206）に対応し、ブラウザのシークを可能にする。
        # media_type を明示し、拡張子推測に依存せず正しくシーク/バッファできるようにする。
        if not state.proxy.exists():
            raise HTTPException(status_code=404, detail="proxy not found")
        return FileResponse(state.proxy, media_type=_video_media_type(state.proxy))

    @app.post("/api/save")
    def save(
        req: SaveRequest,
        x_csrf_token: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        # CSRF: 同一オリジンが埋め込みトークンを提示した場合のみ許可
        if not x_csrf_token or not secrets.compare_digest(x_csrf_token, csrf_token):
            raise HTTPException(status_code=403, detail="invalid or missing CSRF token")

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
            n = save_segments(segments, state.save_target)

            # 保存後のセグメント（採番が確定したもの）を読み戻してペアリング
            corrected = load_segments(state.save_target)
            pairs = make_pairs(
                state.original_segments,
                corrected,
                procedure=state.procedure,
                video=state.video.name,
            )
            with open(state.pairs_target, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

        return JSONResponse({
            "saved": n,
            "srt": str(state.save_target),
            "pairs": len(pairs),
            "pairs_file": str(state.pairs_target),
        })

    return app


def build_state(
    video: str,
    srt: Optional[str],
    procedure: str,
    save_target: Optional[str],
) -> AppState:
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    # 語彙を先に検証（未登録術式なら即エラー）
    get_label_set(procedure)

    # ブラウザ再生用プロキシを取得/生成
    pm = ProxyManager()
    if not pm.proxy_exists(str(video_path), PROXY_RESOLUTION):
        print(f"[proxy] 生成中（{PROXY_RESOLUTION}）: {video_path.name} ...")
        proxy = pm.create_proxy(str(video_path), resolution=PROXY_RESOLUTION)
    else:
        proxy = pm.get_proxy_path(str(video_path), PROXY_RESOLUTION)
    print(f"[proxy] {proxy}")

    srt_path = Path(srt) if srt else None
    if srt_path and not srt_path.exists():
        raise FileNotFoundError(f"SRTが見つかりません: {srt_path}")

    if save_target:
        save_path = Path(save_target)
    else:
        save_path = video_path.with_name(f"{video_path.stem}_gold.srt")
    pairs_path = save_path.with_name(f"{save_path.stem}_dpo_pairs.jsonl")

    return AppState(
        video=video_path,
        srt=srt_path,
        procedure=procedure,
        proxy=proxy,
        save_target=save_path,
        pairs_target=pairs_path,
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
    args = parser.parse_args()

    state = build_state(args.video, args.srt, args.procedure, args.save_target)
    app = create_app(state)

    url = f"http://{args.host}:{args.port}/"
    print(f"[server] {url}")
    print(f"[server] 保存先: {state.save_target}")
    print(f"[server] ペア : {state.pairs_target}")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
