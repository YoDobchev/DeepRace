# DeepRace

A small RC car project that follows a taped line using a sim2real RL pipeline.

![demo](demo.gif)

## What it does
- Drives on a physical track
- Streams camera video from ESP32-CAM
- Extracts simple vision features on a laptop:
  - distance from the line (CTE)
  - curvature / heading
- Reads yaw rate (IMU gyro)
- Uses a discrete policy to pick actions like:
  - forward, full left, full right, brake
- Sends actions to the car via Wi-Fi

## Why sim2real
Instead of training directly on raw camera pixels, the policy trains in simulation on features (CTE, curvature, yaw rate).  
On the real car, the same features are computed from the camera + sensors, then the trained policy runs on the live data.

## Hardware
- 1 x ESP32 Dev board
- 1 x ESP32-CAM
- 4WD TT chassis + motor drivers (DRV8833)
- IMU
- Battery + buck converters

## Software
- Python 3.10+
- OpenCV
- Gymnasium custom env
- Stable-Baselines3 (DQN)

## Status
Work in progress. The car is currently slower and less accurate than a basic algorithm. Things I want to improve:
- A full car + track physics simulation (instead of the current simple environment) for more realistic training
- Continuous action space for smoother, more precise steering
- Adding a hall sensor for speed measurement
- Use a more stable and higher quality camera than the ESP32-CAM