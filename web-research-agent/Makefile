.PHONY: run test clean verify

run:
	docker-compose up --build

test:
	docker-compose -f docker-compose.yml run --rm app pytest tests/

verify:
	docker-compose -f docker-compose.yml run --rm app bash verify.sh

clean:
	docker-compose down -v
	rm -rf results/*
