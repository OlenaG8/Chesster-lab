#!/bin/bash
set -euo pipefail

conda run -n chesster --no-capture-output lerobot-find-port
conda run -n chesster --no-capture-output lerobot-find-port

sudo chmod 666 /dev/ttyACM*

conda run -n chesster lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{
        "front": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 360, "fps": 30, "fourcc": "MJPG"},
        "hand": {"type": "opencv", "index_or_path": 6, "width": 640, "height": 360, "fps": 30, "fourcc": "MJPG"}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true