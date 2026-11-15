.PHONY: install run run-loading run-features test
install:
	pip install -r requirements.txt
run:
	streamlit run app.py
run-loading:
	streamlit run modules/data_loading/app.py
run-features:
	streamlit run modules/feature_extraction/app.py
test:
	pytest -q
