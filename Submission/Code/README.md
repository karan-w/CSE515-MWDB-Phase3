# CSE515-MWDB-Phase3
This is the code repository for phase 3 of the project for CSE 515 - Multimedia and Web Databases. 

python3 -m venv /path/to/new/virtual/environment
Example: python3 -m venv venv

Linux/maxOS: source venv/bin/activate
Windows (cmd): C:\> <venv>\Scripts\activate.bat

pip install -r requirements.txt

sphinx-quickstart docs

sphinx-build -b html docs/source/ docs/build/html

https://www.sphinx-doc.org/en/master/tutorial/first-steps.html

https://betterprogramming.pub/auto-documenting-a-python-project-using-sphinx-8878f9ddc6e9


sphinx-apidoc -o docs/build/html src/

pip install -e .