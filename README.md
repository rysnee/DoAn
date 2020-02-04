# Image retrieval Project

## Install newsest version of Pillow and Django

```bash
pip install Django==3.0.2
```

```bash
pip install pillow
```

## Download Oxford Building database and extract to 'CBIR/mysite/static/dataset/'

```bash
http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
```

## Running Locally

Move to folder 'CBIR/' and run 2 commands:

```bash
python manage.py migrate
```

```bash
python manage.py runserver
```
On your browser, go to:

```bash
127.0.0.1:8000
```

## Evaluate
Follow guide in README.md at:

```bash
CBIR/VLADdata/
```
