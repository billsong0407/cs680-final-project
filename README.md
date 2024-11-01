## Prerequisites
- Python 3.10 or higher

## Environment Setup
Run the following commands in the project root directory:

1. Create a virtual environment:
```bash
python -m venv myenv
```
2. Activate environment
```bash
source myenv/bin/activate
```
3. Install dependencies from requirements.txt
```bash
python -m pip install -r requirements.txt
```

## Web Crawler
```bash
cd /scrapper
python generator.py -q "healthy house plant" -b "bing" -n 500 -s "medium"
```