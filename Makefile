init:
	pip install -r requirements.txt

test:
	python -m unittest tests.py

profile:
	python -m unittest profl.py
