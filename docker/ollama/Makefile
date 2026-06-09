MODEL ?= gemma4:26b
ROOT := $(shell git rev-parse --show-toplevel)

.PHONY: up down pull logs ps shell

up:
	docker compose -f $(ROOT)/docker-compose.yml up -d

down:
	docker compose -f $(ROOT)/docker-compose.yml down

pull:
	docker compose -f $(ROOT)/docker-compose.yml exec ollama ollama pull $(MODEL)

logs:
	docker compose -f $(ROOT)/docker-compose.yml logs -f

ps:
	docker compose -f $(ROOT)/docker-compose.yml ps

shell:
	docker compose -f $(ROOT)/docker-compose.yml exec ollama bash
