# Docker 操作チートシート

---

## 1. コンテナの起動・停止

### docker compose（複数コンテナをまとめて操作）

```bash
# 起動（バックグラウンド）
# -d は --detach の略。ターミナルとコンテナを切り離す（デタッチ）ことで
# バックグラウンド動作になる。-d なしだと Ctrl+C で止まってしまう。
docker compose up -d

# 停止（コンテナを削除）
# down はコンテナを止めて削除する。次の up で再作成される。
# stop はコンテナを止めるだけで削除しない（down との違い）。
docker compose down

# 再起動
docker compose restart

# 特定のサービスだけ起動
docker compose up -d annotator
```

### 動作確認

```bash
# 起動中のコンテナ一覧（状態・ポートも表示）
docker compose ps

# 全コンテナ一覧（停止中も含む）
# -a は --all の略。デフォルトは起動中のみ表示される。
docker ps -a
```

---

## 2. イメージのビルド

```bash
# 全サービスをビルドして起動
# && は「前のコマンドが成功したら次を実行」という意味
docker compose build && docker compose up -d

# 特定サービスだけビルド
docker compose build annotator

# ビルド後すぐ起動（build と up を一度に行う）
docker compose up -d --build annotator
```

---

## 3. ログの確認

```bash
# サービスのログを表示
docker compose logs annotator

# リアルタイムで流す
# -f は --follow の略。tail -f と同じ感覚。Ctrl+C で抜ける。
docker compose logs -f annotator
```

---

## 4. コンテナの中に入る

```bash
# コンテナ内でコマンド実行
# -i は --interactive（標準入力を有効にする）
# -t は --tty（ターミナルを割り当てる）
# -it をセットで使うことでターミナルとして対話操作できるようになる
# exit または Ctrl+D でコンテナから抜ける
docker exec -it annotator bash
docker exec -it ollama bash

# コンテナ内でコマンドを1回だけ実行して終了（-it 不要）
docker exec ollama ollama list
```

---

## 5. ディスク容量の確認

```bash
# Dockerが使っている容量の概要
docker system df

# 詳細（イメージ・コンテナ・ボリューム別）
# -v は --verbose の略。詳細表示によく使われるオプション。
docker system df -v

# ホストのディスク全体
# -h は --human-readable の略。1G や 500M など人間が読みやすい単位で表示。
df -h

# 特定ディレクトリのサイズ
# du = disk usage。-s は合計のみ表示（-s なしだとサブディレクトリも全部出る）
du -sh /home/docker/
```

---

## 6. 不要データの削除（容量を空ける）

```bash
# 使われていないイメージ（<none>タグ）を削除
# <none> はビルドのたびに古いレイヤーが残ったもの（dangling imageと呼ぶ）
# -f は --force の略。確認プロンプトをスキップして強制実行。
docker image prune -f

# ビルドキャッシュを削除
docker builder prune -f

# 上記すべてをまとめて削除（コンテナ・ネットワーク・イメージ・キャッシュ）
# ※ 起動中のコンテナには影響しない
docker system prune -f
```

---

## 7. ボリュームの操作

```bash
# ボリューム一覧とサイズ
docker system df -v | grep -A 10 "VOLUME NAME"

# ボリュームの詳細（保存先パスなど）
docker volume inspect ollama_ollama_data

# ボリュームを手動で作成
docker volume create --name ollama_ollama_data
```

---

## 8. Dockerのデータ置き場を変更する手順

Dockerのデフォルトの保存先は `/var/lib/docker`（rootパーティション）。
容量が大きい別ディスクに移動したい場合の手順。

### ① 現状確認

```bash
# Dockerのデータがどこにあるか確認
docker info | grep "Docker Root Dir"

# ディスクの空き容量確認
df -h

# ディスク構成確認
lsblk
```

### ② コンテナとDockerを止める

```bash
# コンテナを全停止
docker compose down

# Dockerサービス自体を停止
sudo systemctl stop docker
```

### ③ データを移動する

```bash
# 例：/home/docker に移動する場合
# （/home が別ディスクにある前提）
sudo mv /var/lib/docker /home/docker
```

> **注意**: 同じパーティション内の mv は一瞬だが、異なるパーティション間の mv は
> 実際にはコピー→削除になるので時間がかかる（数十GB規模だと数分〜十数分）。
>
> **落とし穴**: Docker を先に起動してしまうと移動先に空の `/home/docker` が
> 作られてしまい、その状態で mv すると `/home/docker/docker/` という
> 二重構造になってしまう。必ず Docker を止めてから mv すること。

### ④ Dockerに新しい場所を設定する

```bash
# 設定ファイルを作成
# mkdir -p は親ディレクトリも含めて一気に作る（すでにあってもエラーにならない）
sudo mkdir -p /etc/docker
# echo の出力を tee でファイルに書き込む。sudo tee にすることで
# root権限が必要なファイルにも書き込める（sudo echo > file はできないため）
echo '{"data-root": "/home/docker"}' | sudo tee /etc/docker/daemon.json
```

### ⑤ Dockerを再起動して確認

```bash
sudo systemctl start docker

# 設定が反映されているか確認
docker info | grep "Docker Root Dir"
# → Docker Root Dir: /home/docker  と表示されればOK

# ボリュームが復活しているか確認
docker volume ls
```

### ⑥ コンテナを起動

```bash
docker compose up -d
```

---

## 9. Dockerサービス自体の操作

```bash
# systemctl は Linux のサービス管理コマンド（systemd）
# Docker はバックグラウンドで動くサービス（デーモン）として動作している
# docker compose とは別物。compose のコンテナを動かす土台にあたる。

# 起動
sudo systemctl start docker

# 停止（docker compose down より先にやると compose も強制終了される）
sudo systemctl stop docker

# 再起動
sudo systemctl restart docker

# 状態確認（Active: running と出れば正常）
sudo systemctl status docker
```

---

## 10. よくあるトラブル

| 症状 | 確認コマンド | 対処 |
|---|---|---|
| コンテナが起動しない | `docker compose logs サービス名` | ログでエラー内容を確認 |
| ディスクがいっぱい | `docker system df` | `docker system prune -f` |
| ボリュームが見つからない | `docker volume ls` | `docker volume create --name 名前` |
| Dockerが起動しない | `sudo systemctl status docker` | `/etc/docker/daemon.json` の記述ミスを確認 |

---

## このプロジェクトのコンテナ構成

```
annotator  (port 8000)  FastAPI + React  ← アノテーションツール
ollama     (port 11434) Ollama           ← VLM推論サーバー
mlflow     (port 5000)  MLflow           ← 実験管理
python_dev              Python開発環境   ← スクリプト実行用
```

### データの保存場所

```
ホスト ./data/     ←→  コンテナ /app/data/     アノテーションJSON・画像
ホスト ./prompts/  ←→  コンテナ /app/prompts/  プロンプトファイル
/home/docker/volumes/ollama_ollama_data/  Ollamaモデル（gemma4:26b 約17GB）
```
