import os
import subprocess

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for i in range(4, 10):
    chapter_dir = os.path.join(parent_dir, f'Chapter_{i}')

    if os.path.isdir(chapter_dir):
        for filename in os.listdir(chapter_dir):
            if filename.endswith('.py'):
                file_path = os.path.join(chapter_dir, filename)

                print(f'Running {file_path}')
                subprocess.run(['python', file_path], check=True)
