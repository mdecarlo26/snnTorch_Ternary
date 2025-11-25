## PyPi Build Instructions
First delete the dist dir.
```
pip install build twine
python -m build .
twine upload dist/*
```