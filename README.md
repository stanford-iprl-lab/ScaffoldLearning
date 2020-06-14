# ScaffoldLearning
Simulation Environments in "Learning to Scaffold the Development of Robotic Manipulation Skills"


1.Initialize repository
```
git submodule init && git submodule update
```

2.Compile bullet
```
cd external/bullet3.git; bash build_cmake_pybullet_double.sh
```

3.Install virtual environment
```
pip install pipenv
pipenv install
```

4.Install CUDA >= 9.2

## Activate virtual environment
```pipenv shell```

## Run the code
```
cd rl; python3 wrench_rl.py
