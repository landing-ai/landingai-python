run_notebook:
	$(DOCKER_RUN) $(APP_NAME) jupyter nbconvert  --clear-output  --execute $(DEFAULT_NB)
	$(DOCKER_RUN) $(APP_NAME) jupyter nbconvert  --to markdown $(DEFAULT_NB)


clean_notebook: 
	$(DOCKER_RUN) $(APP_NAME) jupyter nbconvert --clear-output $(DEFAULT_NB)

clean: clean_notebook
	rm -rf .cache .config .ipynb_checkpoints .ipython .jupyter .local __pycache__ .npm
