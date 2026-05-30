#!/bin/bash

pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install -e .

mkdir -p models
mkdir -p src
python -c "import pickle; import numpy as np; from sklearn.preprocessing import StandardScaler; scaler = StandardScaler(); scaler.fit(np.array([[0.0], [1.0], [2.0]])); with open('models/estimator.pkl', 'wb') as f: pickle.dump(scaler, f)"
