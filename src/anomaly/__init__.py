"""
Self-Supervised Video Anomaly Detection モジュール

正常フレームだけで学習し、出血・煙・カメラ衝突等を
「正常分布からの逸脱」として検出する。

ConvLSTM Autoencoder による次フレーム予測を使用。
予測誤差が大きい区間 = 異常（出血など）。
"""
