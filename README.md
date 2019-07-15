The code is based on [Curiosity-driven Exploration by Self-supervised Prediction](https://github.com/pathak22/noreward-rl) and the license comes from the project.

## Memory Replay with Trajectory for Side-Scrolling Video Games ##
Based on previous related works,to enhance the curiosity-driven exploration andutilize prior experience more effectively, we develop a new memory replay mechanism, whichconsists of two modules: Trajectory Replay Module (TRM) to record the agent moving trajectory information with much less space, and the Trajectory Optimization Module (TOM) to formulate the state information as reward.

### 1.  This code is based on [TensorFlow](https://www.tensorflow.org/). To install, run these commands:
  ```Shell
  # you might not need many of these, e.g., fceux is only for mario
  sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb \
  libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig python3-dev \
  python3-venv make golang libjpeg-turbo8-dev gcc wget unzip git fceux virtualenv \
  tmux

  # install the code
  cd superMarioBros
  virtualenv tom
  source $PWD/tom/bin/activate
  pip install numpy
  pip install pandas
  pip install -r src/requirements.txt
  python tom/src/go-vncdriver/build.py
  ```

### 2. Setting World and Level of Super Mario Bros.
  ```Shell
  cd tom/src/
  vim envs.py
    # revise the lines below
    env_id = 'ppaquette/SuperMarioBros-' + level + '-v0'
    env = env_wrapper.MarioEnv(env, tilesEnv=False)
    
    # if you want to train the agent in a effient version.
    # revise the lines below
    env_id = 'ppaquette/SuperMarioBros-' + level + '-Tiles-v0'
    env = env_wrapper.MarioEnv(env, tilesEnv=True)
  
  vim a3c_TOM.py
    LEVEL = "1-1" # revise the level to which you want
  ```


### 3. Training code
  ```Shell
  cd tom/src/
  python train.py --default --env-id mario --noReward --log-dir tmp/ac4_$LEVEL --num-worker 4 --pretrain ./model/dir
  ```
### Note
The log directory which is parameter of `log-dir` must be clean! Or the neural network will load the weight and bias from the dirty directory.

### 4. Testing model
  ```Shell
  cd tom/src/
  vim worker.py
    #from ICM_TOM import A3C # comment the line
    from test_dis import A3C # uncomment the line
  
  python train.py --default --env-id mario --noReward --log-dir tmp/testing_mario --num-worker 4 --pretrain ./model/testing/model/dir
  ```
