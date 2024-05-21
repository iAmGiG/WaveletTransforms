import os

def ensure_directories_exist():
    required_dirs = ['results', 'results/confusion_matrices']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

if __name__ == '__main__':
    ensure_directories_exist()
