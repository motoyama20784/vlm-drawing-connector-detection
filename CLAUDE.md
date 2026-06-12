
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This repository is for **VLM Drawing Connector Detection** — a project that uses Vision Language Models to detect connectors (lines, arrows, edges) in technical drawings or diagrams.

The codebase has not been initialized yet. Update this file once the project structure, dependencies, and build tooling are established.

## アノテーションツール

`annotator/` ディレクトリに FastAPI + React 製のアノテーション Web アプリがある。

- 起動: `docker compose up annotator`
- アクセス: `http://localhost:8000`
- バックエンドテスト: `docker compose exec python pytest tests/annotator/ -v`
- フロントエンドビルド: `cd annotator/frontend && npm install && npm run build`
- フロントエンドを変更したら `docker compose build annotator` が必要
