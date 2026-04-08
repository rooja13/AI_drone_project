# AI for Unmanned Systems Project
This project aims to implement a autonomous drone controlled by a neural network to follow a ground vehicle

## Project Structure
Here is the project structure so far and is subject to change:

- drone_tracker/
  - worlds/
    - flat_world.sdf --> The main simulation world
  - models/
    - quadrotor/
      - model.config
      - model.sdf --> drone
  - ground_vehicle/
      - model.config
      - model.sdf --> Ground vehicle
  - scripts/
      - teleop.py --> Keyboard control script

# Setup
This setup is only for Linux (so far). I am running this on a virtual machine and I recommend you also use a virtual machine if you don't have a Linux machine.

## Install Gazebo
Install Gazebo Harmonic (if not already installed) with the following command:
```
sudo apt-get install gz-harmonic
```

## Install Python
Install Python pip (if not already installed) with the following command:
```
sudo apt install python3-pip
```

Then install Gazebo for Python with:
```
pip install gz-python
```

# Running the Simulation
You need to run the following command to let Gazebo know where you project models are. Make sure you are in the root directory of the project when running this command.

This command needs to be done every terminal session
```
export GZ_SIM_RESOURCE_PATH=$PWD/models
```

Then run the simulation with:
```
gz sim worlds/flat_world.sdf
```

This should open the Gazebo GUI and you can start the simulation there.

To control the drone with WASD+QE keys, open a second terminal and run the `teleop.py` file:
```
python3 scripts/teleop.py
```

WASD: move the drone forwards, backwards, left, and right
Q/E: up and down
Arrow keys: control the yaw

If running in a virtual machine, ensure the world SDF uses <render_engine>ogre</render_engine> instead of ogre2, which requires a physical GPU.
