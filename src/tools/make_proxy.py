"""
プロキシ動画変換ツール（make_proxy）

高解像度の動画ファイルを低解像度のプロキシ動画に変換する。
分析の前処理として使用し、処理速度とストレージを節約する。

依存: ffmpeg（システムにインストール済みであること）

使用例:
    # 単一ファイル
    python -m src.tools.make_proxy --video case001.mp4 --outdir proxy/

    # ディレクトリ一括
    python -m src.tools.make_proxy --video-dir ~/動画/ --outdir ~/proxy/

    # 解像度・品質指定
    python -m src.tools.make_proxy --video case001.mp4 --outdir proxy/ \\
        --size 800x600 --crf 23 --no-audio
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# サポートする動画拡張子
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mts", ".m2ts", ".wmv", ".flv"}


# ---------------------------------------------------------------------------
# ffmpegコマンド生成
# ---------------------------------------------------------------------------

def build_ffmpeg_command(
    input_path: str,
    output_path: str,
    size: str = "800x600",
    crf: int = 23,
    no_audio: bool = True,
) -> List[str]:
    """
    ffmpeg変換コマンドを組み立てる。

    Args:
        input_path: 入力動画ファイルパス
        output_path: 出力動画ファイルパス
        size: 出力解像度（"800x600" or "800:-1" 等）
        crf: 品質（0-51、低いほど高品質、デフォルト: 23）
        no_audio: Trueなら音声を除去

    Returns:
        ffmpegコマンドのリスト
    """
    # サイズ文字列をffmpegのscaleフィルター形式に変換
    # "800x600" → "800:600", "800:-1" はそのまま
    scale = size.replace("x", ":")

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", f"scale={scale}",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
    ]

    if no_audio:
        cmd.append("-an")
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])

    # 上書き確認なし
    cmd.extend(["-y", str(output_path)])
    return cmd


# ---------------------------------------------------------------------------
# 入力ファイル検索
# ---------------------------------------------------------------------------

def find_video_files(path: str) -> List[Path]:
    """
    指定パスから動画ファイルのリストを取得する。

    Args:
        path: ファイルまたはディレクトリのパス

    Returns:
        動画ファイルのPathリスト（ソート済み）
    """
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            return [p]
        return []
    elif p.is_dir():
        files = []
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(f)
        return files
    return []


# ---------------------------------------------------------------------------
# 変換実行
# ---------------------------------------------------------------------------

def make_output_path(video_path: Path, outdir: Path, suffix: str = "_proxy") -> Path:
    """変換後のファイルパスを生成する。"""
    return outdir / f"{video_path.stem}{suffix}.mp4"


def convert_one(
    video_path: Path,
    outdir: Path,
    size: str = "800x600",
    crf: int = 23,
    no_audio: bool = True,
    force: bool = False,
) -> Optional[str]:
    """
    1ファイルをプロキシ動画に変換する。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        size: 出力解像度
        crf: 品質
        no_audio: 音声除去
        force: 既存ファイルを上書き

    Returns:
        出力ファイルパス（変換成功時）、Noneはスキップまたはエラー
    """
    out_path = make_output_path(video_path, outdir)

    if out_path.exists() and not force:
        print(f"スキップ（変換済み）: {video_path.name} → {out_path.name}")
        return None

    cmd = build_ffmpeg_command(
        str(video_path), str(out_path),
        size=size, crf=crf, no_audio=no_audio,
    )

    print(f"変換中: {video_path.name} → {out_path.name} ({size}, crf={crf})")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1時間のタイムアウト
        )
        if result.returncode != 0:
            print(f"エラー: {video_path.name}", file=sys.stderr)
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
                  file=sys.stderr)
            return None

        # ファイルサイズ表示
        in_size = video_path.stat().st_size / (1024 * 1024)
        out_size = out_path.stat().st_size / (1024 * 1024)
        ratio = (out_size / in_size * 100) if in_size > 0 else 0
        print(f"完了: {in_size:.1f}MB → {out_size:.1f}MB ({ratio:.0f}%)")

        return str(out_path)

    except FileNotFoundError:
        print("エラー: ffmpegが見つかりません。インストールしてください。",
              file=sys.stderr)
        print("  sudo apt install ffmpeg", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"エラー: タイムアウト（1時間超過）: {video_path.name}",
              file=sys.stderr)
        return None


def convert_batch(
    videos: List[Path],
    outdir: Path,
    size: str = "800x600",
    crf: int = 23,
    no_audio: bool = True,
    force: bool = False,
) -> dict:
    """
    複数ファイルを一括変換する。

    Returns:
        {"converted": 変換数, "skipped": スキップ数, "failed": 失敗数,
         "outputs": [出力パスリスト]}
    """
    outdir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    failed = 0
    outputs: List[str] = []

    total = len(videos)
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{total}] ", end="")
        result = convert_one(
            video, outdir,
            size=size, crf=crf, no_audio=no_audio, force=force,
        )
        if result is not None:
            converted += 1
            outputs.append(result)
        elif make_output_path(video, outdir).exists() and not force:
            skipped += 1
        else:
            failed += 1

    print(f"\n==============================")
    print(f"変換完了: {converted}, スキップ: {skipped}, 失敗: {failed}")
    print(f"出力先: {outdir}")

    return {
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    """コマンドラインエントリポイント"""
    # ffmpegの存在チェック
    if shutil.which("ffmpeg") is None:
        print("エラー: ffmpegが見つかりません。", file=sys.stderr)
        print("  sudo apt install ffmpeg", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        description="動画をプロキシ（低解像度）に変換する"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="入力動画ファイル（単一）")
    group.add_argument("--video-dir", help="入力ディレクトリ（一括変換）")

    parser.add_argument("--outdir", required=True, help="出力ディレクトリ")
    parser.add_argument("--size", default="800x600",
                        help="出力解像度（デフォルト: 800x600、アスペクト比維持: 800:-1）")
    parser.add_argument("--crf", type=int, default=23,
                        help="品質（0-51、低いほど高品質、デフォルト: 23）")
    parser.add_argument("--no-audio", action="store_true", default=True,
                        help="音声を除去する（デフォルト: 有効）")
    parser.add_argument("--with-audio", action="store_true",
                        help="音声を残す")
    parser.add_argument("--force", action="store_true",
                        help="変換済みファイルを上書きする")

    args = parser.parse_args()

    # 音声設定
    no_audio = not args.with_audio

    # 入力ファイルの取得
    if args.video:
        videos = find_video_files(args.video)
    else:
        videos = find_video_files(args.video_dir)

    if not videos:
        print("エラー: 変換対象の動画ファイルが見つかりません。", file=sys.stderr)
        return 1

    print(f"対象ファイル: {len(videos)} 件")
    for v in videos:
        print(f"  - {v.name}")

    outdir = Path(args.outdir)
    result = convert_batch(
        videos, outdir,
        size=args.size, crf=args.crf, no_audio=no_audio, force=args.force,
    )

    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
