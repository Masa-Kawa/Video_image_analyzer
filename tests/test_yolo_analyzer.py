"""
yolo_analyzer のクラスIDマッピングとフェーズ推定のユニットテスト

ultralytics YOLOv8 は標準 COCO80（0始まり）でクラスIDを返すため、
SURGICAL_INSTRUMENTS が標準 COCO に整合していること、および knife/scissors の
ID が infer_phase で正しく機能することを検証する。
"""

import unittest

from src.yolo.yolo_analyzer import (
    SURGICAL_INSTRUMENTS,
    INSTRUMENT_COCONames,
    cluster_scenes,
    infer_phase,
)


class TestClassIdMapping(unittest.TestCase):
    """標準 COCO80 インデックスとの整合性。"""

    def test_knife_and_scissors_ids(self):
        # 標準 COCO80: knife=43, scissors=76
        self.assertEqual(SURGICAL_INSTRUMENTS[43], "knife")
        self.assertEqual(SURGICAL_INSTRUMENTS[76], "scissors")

    def test_surrounding_ids_aligned(self):
        self.assertEqual(SURGICAL_INSTRUMENTS[42], "fork")
        self.assertEqual(SURGICAL_INSTRUMENTS[44], "spoon")
        self.assertEqual(SURGICAL_INSTRUMENTS[75], "vase")
        self.assertEqual(SURGICAL_INSTRUMENTS[0], "person")

    def test_no_out_of_range_id(self):
        # COCO80 の最大IDは 79（旧コードの 80 は off-by-one の症状）
        self.assertEqual(max(SURGICAL_INSTRUMENTS), 79)
        self.assertNotIn(80, SURGICAL_INSTRUMENTS)

    def test_cocoNames_consistent_with_main_map(self):
        # 補助マップが主マップと矛盾しないこと
        for cid, name in INSTRUMENT_COCONames.items():
            self.assertEqual(SURGICAL_INSTRUMENTS[cid], name)


class TestInferPhase(unittest.TestCase):
    """knife/scissors の名称が正しく解決された前提でのフェーズ推定。"""

    def test_knife_to_dissection(self):
        # SURGICAL_INSTRUMENTS[43] == "knife" なので dissection になる
        phase, primary = infer_phase([SURGICAL_INSTRUMENTS[43], "person"])
        self.assertEqual(phase, "dissection")
        self.assertIn("knife", primary)

    def test_scissors_to_cutting(self):
        phase, primary = infer_phase([SURGICAL_INSTRUMENTS[76]])
        self.assertEqual(phase, "cutting")
        self.assertIn("scissors", primary)

    def test_person_only_manipulation(self):
        phase, primary = infer_phase(["person"])
        self.assertEqual(phase, "manipulation")

    def test_empty_neutral(self):
        phase, primary = infer_phase([])
        self.assertEqual(phase, "neutral")


class TestClusterScenes(unittest.TestCase):
    """連続フレームの器械セット変化に基づくシーン分割。"""

    def test_empty_input_returns_empty(self):
        self.assertEqual(cluster_scenes([], []), [])

    def test_single_long_scene(self):
        # 器械セットが一定 → 1シーンに集約。min_duration を満たす長さ。
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        sets = [["knife"]] * 5
        scenes = cluster_scenes(times, sets, min_duration=3.0)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["start"], 0.0)
        self.assertEqual(scenes[0]["end"], 4.0)
        self.assertEqual(scenes[0]["count"], 5)
        # infer_phase の結果が付与される
        self.assertEqual(scenes[0]["phase"], "dissection")

    def test_scene_split_on_instrument_change(self):
        # 前半 knife / 後半 scissors の2シーンに分かれること
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        sets = [["knife"]] * 4 + [["scissors"]] * 4
        scenes = cluster_scenes(times, sets, min_duration=3.0)
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0]["phase"], "dissection")
        self.assertEqual(scenes[1]["phase"], "cutting")

    def test_short_scene_below_min_duration_dropped(self):
        # 2秒しか続かないシーンは min_duration=3.0 未満で除去される
        times = [0.0, 1.0, 2.0]
        sets = [["knife"], ["knife"], ["knife"]]
        scenes = cluster_scenes(times, sets, min_duration=3.0)
        self.assertEqual(scenes, [])

    def test_instrument_order_does_not_split(self):
        # 集合として同一なら順序が違ってもシーンは分割されない
        times = [0.0, 1.0, 2.0, 3.0]
        sets = [["knife", "person"], ["person", "knife"],
                ["knife", "person"], ["person", "knife"]]
        scenes = cluster_scenes(times, sets, min_duration=3.0)
        self.assertEqual(len(scenes), 1)


if __name__ == "__main__":
    unittest.main()
