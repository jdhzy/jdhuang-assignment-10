VENV_DIR := venv

install:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

run:
	$(VENV_DIR)/bin/python app.py

clean:
	rm -rf $(VENV_DIR)