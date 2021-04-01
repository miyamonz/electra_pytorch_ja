install:
	pip install -r requirements.txt

Pretrain.py: Pretrain.ipynb
	jupyter nbconvert --to script Pretrain.ipynb

clear:
	jupyter nbconvert --clear-output --inplace Pretrain.ipynb
	jupyter nbconvert --clear-output --inplace Finetune_GLUE.ipynb

jupyter:
	jupyter lab --no-browser --NotebookApp.token='' --NotebookApp.password=''
