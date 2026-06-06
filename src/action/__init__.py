"""手技・動作認識「変換器」パッケージ（action_to_outputs）。

先行研究の動作/手技認識結果（CholecT50 triplet、汎用 action、clip 単位ラベル）を
リポジトリ標準の二層構造（SRT＋JSONL/CSV）へ変換する。推論モデル本体は含まない。
"""
