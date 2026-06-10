"""
統合手術動画解析パイプライン（surgical_pipeline）

腹腔鏡下胆嚢摘出術動画に対して、3つの解析を一括実行する:

1. 腹腔内外判定（cavity_detector）
2. 出血検出（bleed_detector + bleed_spread）
3. フェーズセグメンテーション（phase_segmenter）

出力:
  - 統合CSV: 全指標を含む時系列（timestamp, inside_abdomen, bleeding_score,
             phase_id, phase_name, red_area_ratio, ...）
  - 個別SRT: 各解析のSRT字幕（Shotcutで動画に重ねて確認）
  - 統合SRT: 全SRTをマージした統合字幕
  - JSONL: 各解析のイベント正本

使用例:
  python -m src.surgical_pipeline analyze \\
      --video surgery.mp4 --outdir output/
"""

import argparse
import bisect
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.core.time_utils import format_srt_time


def _safe_step(label: str, fn: Callable[[], dict]) -> Optional[dict]:
    """1解析ステップを実行する。

    例外が発生しても捕捉してそのステップのみスキップし、それまでの長時間
    実行結果（他ステップの出力）を失わないようにする。失敗時は None を返す。
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — 1ステップの失敗を全体に波及させない
        print(f"⚠ {label} でエラーが発生したためスキップします: {e}",
              file=sys.stderr)
        traceback.print_exc()
        return None


def run_surgical_analysis(
    video_path: str,
    outdir: str,
    # 共通パラメータ
    device: str = "cuda",
    # cavity パラメータ
    cavity_fps: float = 2.0,
    cavity_thr: float = 0.35,
    cavity_smooth_s: float = 3.0,
    cavity_min_duration: float = 3.0,
    roi_margin: float = 0.08,
    no_roi: bool = False,
    # bleeding パラメータ
    bleed_fps: float = 5.0,
    bleed_thr: float = 0.005,
    bleed_k_s: float = 1.0,
    bleed_smooth_s: float = 5.0,
    bleed_bg_norm: float = 30.0,
    bleed_s_min: int = 60,
    bleed_v_min: int = 40,
    # spread パラメータ
    spread_thr: float = 0.001,
    spread_grid: int = 8,
    # bleed_ai パラメータ
    bleed_ai_fps: float = 10.0,
    bleed_ai_thr: float = 0.10,
    bleed_ai_min_duration: float = 2.0,
    bleed_ai_smooth_s: float = 5.0,
    bleed_ai_use_deep: bool = False,
    bleed_ai_classifier_weights: Optional[str] = None,
    bleed_ai_segmenter_weights: Optional[str] = None,
    # phase パラメータ
    phase_fps: float = 1.0,
    phase_use_resnet: bool = True,
    phase_smooth_s: float = 3.0,
    phase_n_phases: int = 7,
    phase_sensitivity: float = 1.5,
    phase_min_s: float = 30.0,
    # 実行制御
    skip_cavity: bool = False,
    skip_bleed: bool = False,
    skip_bleed_ai: bool = False,
    skip_phase: bool = False,
) -> dict:
    """
    手術動画の統合解析を実行する。

    Args:
        video_path: 入力動画ファイルパス
        outdir: 出力ディレクトリ
        (各種パラメータ — 上記参照)

    Returns:
        全出力ファイルパスの辞書
    """
    # 入力検証: 動画パスが存在しなければ即座に失敗する（長時間実行の前に弾く）
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
    if not video_file.is_file():
        raise ValueError(f"動画パスがファイルではありません: {video_path}")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    results: Dict[str, dict] = {}
    srt_files: List[str] = []

    # ================================================================
    # 1. 腹腔内外判定
    # ================================================================
    if not skip_cavity:
        print("=" * 60)
        print("STEP 1: 腹腔内外判定 (Cavity Detection)")
        print("=" * 60)

        from src.cavity.cavity_detector import analyze_video as cavity_analyze

        cavity_result = _safe_step("STEP 1 (cavity)", lambda: cavity_analyze(
            video_path=video_path,
            outdir=outdir,
            fps=cavity_fps,
            roi_margin=roi_margin,
            no_roi=no_roi,
            smooth_s=cavity_smooth_s,
            thr=cavity_thr,
            min_duration_s=cavity_min_duration,
        ))
        if cavity_result:
            results["cavity"] = cavity_result
            if "srt" in cavity_result:
                srt_files.append(cavity_result["srt"])
        print()

    # ================================================================
    # 2. 出血検出
    # ================================================================
    if not skip_bleed:
        print("=" * 60)
        print("STEP 2: 出血検出 (Bleeding Detection)")
        print("=" * 60)

        # 2a. red_expansion ベース
        print("--- 2a. Red Expansion ---")
        from src.red.bleed_detector import analyze_video as bleed_expansion_analyze

        bleed_exp_result = _safe_step("STEP 2a (red expansion)", lambda: bleed_expansion_analyze(
            video_path=video_path,
            outdir=outdir,
            fps=bleed_fps,
            s_min=bleed_s_min,
            v_min=bleed_v_min,
            roi_margin=roi_margin,
            no_roi=no_roi,
            smooth_s=bleed_smooth_s,
            bg_norm_factor=bleed_bg_norm,
            thr=bleed_thr,
            k_s=bleed_k_s,
        ))
        if bleed_exp_result:
            results["bleed_expansion"] = bleed_exp_result
            if "srt" in bleed_exp_result:
                srt_files.append(bleed_exp_result["srt"])

        # 2b. spread_score ベース
        print("--- 2b. Spread Score ---")
        from src.red.bleed_spread import analyze_video as bleed_spread_analyze

        bleed_spread_result = _safe_step("STEP 2b (spread score)", lambda: bleed_spread_analyze(
            video_path=video_path,
            outdir=outdir,
            fps=bleed_fps,
            grid_size=spread_grid,
            s_min=bleed_s_min,
            v_min=bleed_v_min,
            roi_margin=roi_margin,
            no_roi=no_roi,
            smooth_s=bleed_smooth_s,
            thr=spread_thr,
            k_s=bleed_k_s,
        ))
        if bleed_spread_result:
            results["bleed_spread"] = bleed_spread_result
            if "srt" in bleed_spread_result:
                srt_files.append(bleed_spread_result["srt"])
        print()

    # ================================================================
    # 2c. AI出血検出
    # ================================================================
    if not skip_bleed_ai:
        print("--- 2c. AI Bleeding Detection ---")
        from src.bleed_ai.detector import analyze_video as bleed_ai_analyze

        bleed_ai_result = _safe_step("STEP 2c (AI bleeding)", lambda: bleed_ai_analyze(
            video_path=video_path,
            outdir=outdir,
            fps=bleed_ai_fps,
            roi_margin=roi_margin,
            no_roi=no_roi,
            smooth_s=bleed_ai_smooth_s,
            device=device,
            use_deep=bleed_ai_use_deep,
            classifier_weights=bleed_ai_classifier_weights,
            segmenter_weights=bleed_ai_segmenter_weights,
            thr=bleed_ai_thr,
            min_duration_s=bleed_ai_min_duration,
        ))
        if bleed_ai_result:
            results["bleed_ai"] = bleed_ai_result
            if "srt" in bleed_ai_result:
                srt_files.append(bleed_ai_result["srt"])
        print()

    # ================================================================
    # 3. フェーズセグメンテーション
    # ================================================================
    if not skip_phase:
        print("=" * 60)
        print("STEP 3: フェーズセグメンテーション (Phase Segmentation)")
        print("=" * 60)

        from src.phase.phase_segmenter import analyze_video as phase_analyze

        phase_result = _safe_step("STEP 3 (phase)", lambda: phase_analyze(
            video_path=video_path,
            outdir=outdir,
            fps=phase_fps,
            device=device,
            use_resnet=phase_use_resnet,
            smooth_s=phase_smooth_s,
            n_phases=phase_n_phases,
            sensitivity=phase_sensitivity,
            min_phase_s=phase_min_s,
        ))
        if phase_result:
            results["phase"] = phase_result
            if "srt" in phase_result:
                srt_files.append(phase_result["srt"])
        print()

    # ================================================================
    # 4. 統合CSV生成
    # ================================================================
    print("=" * 60)
    print("STEP 4: 統合CSV生成")
    print("=" * 60)

    combined_csv = _safe_step("STEP 4 (combined CSV)", lambda: _build_combined_csv(
        stem=stem,
        outdir=outdir,
        results=results,
    ))
    if combined_csv:
        results["combined_csv"] = combined_csv
        print(f"統合CSV: {combined_csv}")

    # ================================================================
    # 5. 統合SRT生成
    # ================================================================
    if len(srt_files) > 1:
        print("=" * 60)
        print("STEP 5: 統合SRTマージ")
        print("=" * 60)

        from src.tools.merge_srt import merge

        merged_srt = str(out_path / f"{stem}_all.srt")
        if _safe_step("STEP 5 (merge SRT)",
                      lambda: merge(merged_srt, srt_files) or {"ok": True}):
            results["merged_srt"] = merged_srt
            print(f"統合SRT: {merged_srt}")

    # ================================================================
    # サマリー
    # ================================================================
    print()
    print("=" * 60)
    print("解析完了")
    print("=" * 60)

    if "cavity" in results:
        ev_count = results["cavity"].get("events", 0)
        print(f"  腹腔内外: {ev_count} 区間")
    if "bleed_expansion" in results:
        ev_count = results["bleed_expansion"].get("events", 0)
        print(f"  出血(expansion): {ev_count} イベント")
    if "bleed_spread" in results:
        ev_count = results["bleed_spread"].get("events", 0)
        print(f"  出血(spread): {ev_count} イベント")
    if "bleed_ai" in results:
        ev_count = results["bleed_ai"].get("events", 0)
        print(f"  出血(AI): {ev_count} イベント")
    if "phase" in results:
        ph_count = results["phase"].get("phases", 0)
        print(f"  フェーズ: {ph_count} 区間")

    print(f"\n出力ディレクトリ: {outdir}")

    return results


# ---------------------------------------------------------------------------
# 統合CSV構築
# ---------------------------------------------------------------------------

def _build_combined_csv(
    stem: str,
    outdir: str,
    results: dict,
) -> Optional[str]:
    """
    各解析のCSVログを統合して1つのCSVにまとめる。

    統合CSVの列:
      timestamp, inside_abdomen, cavity_score,
      bleeding_score, red_area_ratio, spread_score,
      phase_id, phase_name, brightness, similarity
    """
    out_path = Path(outdir)

    # 各CSVの読み込み
    cavity_data = None
    bleed_data = None
    spread_data = None
    phase_data = None

    if "cavity" in results and "csv" in results["cavity"]:
        from src.cavity.cavity_detector import read_cavitylog_csv
        cavity_data = read_cavitylog_csv(results["cavity"]["csv"])

    if "bleed_expansion" in results and "csv" in results["bleed_expansion"]:
        from src.red.bleed_detector import read_bleedlog_csv
        bleed_data = read_bleedlog_csv(results["bleed_expansion"]["csv"])

    if "bleed_spread" in results and "csv" in results["bleed_spread"]:
        from src.red.bleed_spread import read_spreadlog_csv
        spread_data = read_spreadlog_csv(results["bleed_spread"]["csv"])

    if "phase" in results and "csv" in results["phase"]:
        from src.phase.phase_segmenter import read_phaselog_csv
        phase_data = read_phaselog_csv(results["phase"]["csv"])

    # フェーズイベントの読み込み
    phase_events: List[dict] = []
    if "phase" in results and "jsonl" in results["phase"]:
        jsonl_path = results["phase"]["jsonl"]
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    phase_events.append(json.loads(line))

    # cavity イベントの読み込み
    cavity_events: List[dict] = []
    if "cavity" in results and "jsonl" in results["cavity"]:
        jsonl_path = results["cavity"]["jsonl"]
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cavity_events.append(json.loads(line))

    # 時刻グリッドの決定（最も細かいサンプリングを基準）
    all_times = set()
    for data_source in [cavity_data, bleed_data, spread_data, phase_data]:
        if data_source:
            all_times.update(data_source["times"])

    if not all_times:
        return None

    sorted_times = sorted(all_times)

    # 各指標を時刻でルックアップする構造を作成。
    # ソート済みキー列を一度だけ事前計算し、bisect で最近傍探索する
    # （タイムスタンプ数 × 列数 回の再ソートを避ける）。
    def make_lookup(data, key) -> Tuple[dict, List[float]]:
        if data is None:
            return {}, []
        lookup = dict(zip(data["times"], data[key]))
        return lookup, sorted(lookup.keys())

    cavity_lookup = make_lookup(cavity_data, "smooth_cavity")
    red_ratio_lookup = make_lookup(bleed_data, "red_ratios")
    bleed_score_lookup = make_lookup(bleed_data, "smooth_expansions")
    spread_lookup = make_lookup(spread_data, "smooth_spreads")
    brightness_lookup = make_lookup(phase_data, "brightnesses")
    similarity_lookup = make_lookup(phase_data, "smooth_similarities")

    # イベントは start_sec 昇順の非重複区間。sorted_times を昇順に走査する
    # ため、前進のみのポインタで O(n + m) に解決できる（線形再走査を回避）。
    phase_events.sort(key=lambda e: e.get("start_sec", 0.0))
    cavity_events.sort(key=lambda e: e.get("start_sec", 0.0))

    phase_ptr = 0
    cavity_ptr = 0

    def find_phase(t: float) -> tuple:
        """時刻tが属するフェーズを返す（前進ポインタ方式）"""
        nonlocal phase_ptr
        while (phase_ptr < len(phase_events)
               and phase_events[phase_ptr]["end_sec"] < t):
            phase_ptr += 1
        if phase_ptr < len(phase_events):
            ev = phase_events[phase_ptr]
            if ev["start_sec"] <= t <= ev["end_sec"]:
                return ev.get("phase_id", -1), ev.get("phase_name", "")
        return -1, ""

    def find_cavity(t: float) -> int:
        """時刻tの腹腔内外判定（1=inside, 0=outside、前進ポインタ方式）"""
        nonlocal cavity_ptr
        while (cavity_ptr < len(cavity_events)
               and cavity_events[cavity_ptr]["end_sec"] < t):
            cavity_ptr += 1
        if cavity_ptr < len(cavity_events):
            ev = cavity_events[cavity_ptr]
            if ev["start_sec"] <= t <= ev["end_sec"]:
                return 1 if ev.get("label") == "inside" else 0
        return -1

    def closest_value(entry: Tuple[dict, List[float]], t: float) -> Optional[float]:
        lookup, keys = entry
        if not lookup:
            return None
        if t in lookup:
            return lookup[t]
        # 最近傍（事前ソート済みキーに対する二分探索 O(log n)）
        pos = bisect.bisect_left(keys, t)
        candidates = []
        if pos < len(keys):
            candidates.append(keys[pos])
        if pos > 0:
            candidates.append(keys[pos - 1])
        if not candidates:
            return None
        nearest = min(candidates, key=lambda k: abs(k - t))
        if abs(nearest - t) < 2.0:
            return lookup[nearest]
        return None

    # CSV書き出し
    csv_path = out_path / f"{stem}_combined.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "t_srt",
            "inside_abdomen", "cavity_score",
            "bleeding_score", "red_area_ratio", "spread_score",
            "phase_id", "phase_name",
            "brightness", "similarity",
        ])

        for t in sorted_times:
            phase_id, phase_name = find_phase(t)
            inside = find_cavity(t)

            writer.writerow([
                f"{t:.3f}",
                format_srt_time(t),
                inside,
                f"{closest_value(cavity_lookup, t) or 0.0:.6f}",
                f"{closest_value(bleed_score_lookup, t) or 0.0:.6f}",
                f"{closest_value(red_ratio_lookup, t) or 0.0:.6f}",
                f"{closest_value(spread_lookup, t) or 0.0:.6f}",
                phase_id,
                phase_name,
                f"{closest_value(brightness_lookup, t) or 0.0:.2f}",
                f"{closest_value(similarity_lookup, t) or 0.0:.6f}",
            ])

    return str(csv_path)


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="統合手術動画解析パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全解析を実行
  python -m src.surgical_pipeline analyze --video surgery.mp4 --outdir output/

  # 出血検出のみスキップ
  python -m src.surgical_pipeline analyze --video surgery.mp4 --outdir output/ --skip-bleed

  # ResNetなし（GPU不要）
  python -m src.surgical_pipeline analyze --video surgery.mp4 --outdir output/ --no-resnet --device cpu
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    ana = subparsers.add_parser("analyze", help="統合解析を実行")
    ana.add_argument("--video", required=True, help="入力動画ファイル")
    ana.add_argument("--outdir", required=True, help="出力ディレクトリ")
    ana.add_argument("--device", default="cuda", help="計算デバイス")

    # 実行制御
    ana.add_argument("--skip-cavity", action="store_true")
    ana.add_argument("--skip-bleed", action="store_true")
    ana.add_argument("--skip-bleed-ai", action="store_true")
    ana.add_argument("--skip-phase", action="store_true")

    # cavity
    ana.add_argument("--cavity-fps", type=float, default=2.0)
    ana.add_argument("--cavity-thr", type=float, default=0.35)

    # bleed
    ana.add_argument("--bleed-fps", type=float, default=5.0)
    ana.add_argument("--bleed-thr", type=float, default=0.005)
    ana.add_argument("--spread-thr", type=float, default=0.001)

    # bleed_ai
    ana.add_argument("--bleed-ai-fps", type=float, default=10.0)
    ana.add_argument("--bleed-ai-thr", type=float, default=0.10)
    ana.add_argument("--bleed-ai-use-deep", action="store_true")

    # phase
    ana.add_argument("--phase-fps", type=float, default=1.0)
    ana.add_argument("--no-resnet", action="store_true")
    ana.add_argument("--n-phases", type=int, default=7)
    ana.add_argument("--phase-sensitivity", type=float, default=1.5)
    ana.add_argument("--min-phase-s", type=float, default=30.0)

    # ROI
    ana.add_argument("--roi-margin", type=float, default=0.08)
    ana.add_argument("--no-roi", action="store_true")

    args = parser.parse_args()

    if args.command == "analyze":
        run_surgical_analysis(
            video_path=args.video,
            outdir=args.outdir,
            device=args.device,
            cavity_fps=args.cavity_fps,
            cavity_thr=args.cavity_thr,
            bleed_fps=args.bleed_fps,
            bleed_thr=args.bleed_thr,
            spread_thr=args.spread_thr,
            phase_fps=args.phase_fps,
            phase_use_resnet=not args.no_resnet,
            phase_n_phases=args.n_phases,
            phase_sensitivity=args.phase_sensitivity,
            phase_min_s=args.min_phase_s,
            roi_margin=args.roi_margin,
            no_roi=args.no_roi,
            bleed_ai_fps=args.bleed_ai_fps,
            bleed_ai_thr=args.bleed_ai_thr,
            bleed_ai_use_deep=args.bleed_ai_use_deep,
            skip_cavity=args.skip_cavity,
            skip_bleed=args.skip_bleed,
            skip_bleed_ai=args.skip_bleed_ai,
            skip_phase=args.skip_phase,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
