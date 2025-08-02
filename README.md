LC RVC
----

単純にHubertとRVCのテキストエンコーダーを拡張したものです。
これにより比較的短いレイテンシ(0.4 s/per iter)であってもロングトーンをキャプチャできるようになりました。
現状RVCv2にのみ対応しておりrealtime処理のためindexサーチは施しておりません。
sdpaによる速度向上のためhuggingface transformer形式を採用しております。
exporter.ipynbからRVCモデルとhubertを変換してください。
ライセンスに関しては追加したコードはMIT、フォーク元のhf-rvcのライセンスに関してはesnyaさんに依拠します。

#使い方
---
run.pyからモデルを編集してそのままpythonで実行してください。
