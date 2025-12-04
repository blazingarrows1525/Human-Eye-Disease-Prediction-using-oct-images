import os

BASE = os.path.dirname(os.path.dirname(__file__))

REQUIRED = [
    'app.py',
    'recommendation.py',
    'Trained_Model.keras',
    'requirements.txt',
    'README.md'
]


def test_required_files_exist():
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(BASE, f))]
    assert not missing, f"Missing required files: {missing}"
