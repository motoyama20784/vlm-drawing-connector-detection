MODEL ?= gemma4:27b

.PHONY: up down pull logs ps shell

up:
	docker compose up -d

down:
	docker compose down

pull:
	docker compose exec ollama ollama pull $(MODEL)

logs:
	docker compose logs -f

ps:
	docker compose ps

shell:
	docker compose exec ollama bash
