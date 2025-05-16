brew install pyenv
pyenv install 3.10.13
pyenv global 3.10.13

python3 -m venv myvenv
source myvenv/bin/activate
python3 -m pip install opencv-python dlib mediapipe tensorflow
