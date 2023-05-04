build_wheel:
	docker build -t landing-ai-wheel:latest --output=type=docker -f ./Dockerfile.wheel .
	docker create -ti --name landing-ai-wheel-tmp landing-ai-wheel bash
	docker cp landing-ai-wheel-tmp:/tmp/dist/. .
	docker rm landing-ai-wheel-tmp
	docker image rm landing-ai-wheel

run_notebook:
	$(DOCKER_RUN) $(APP_NAME) jupyter nbconvert  --clear-output  --execute $(DEFAULT_NB)
	$(DOCKER_RUN) $(APP_NAME) jupyter nbconvert  --to markdown $(DEFAULT_NB)


clean_notebook: 
	$(DOCKER_RUN) $(APP_NAME) jupyter nbconvert --clear-output $(DEFAULT_NB)

clean: clean_notebook
	rm -rf .cache .config .ipynb_checkpoints .ipython .jupyter .local __pycache__ .npm
