windows



lerobot-record --robot.type=so101_follower --robot.port=COM5 --robot.cameras="{ cam_overhead: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}, cam_ego: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, cam_external: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" --policy.path=tinkerbuggy/le-pickup-pi05 --policy.device=cuda --dataset.repo_id=tinkerbuggy/eval_le-pickup-inference-test3 --dataset.single_task="Pick up the blue cube and place it in the cardboard box" --dataset.num_episodes=1 --dataset.push_to_hub=false --display_data=true


lerobot-record --robot.type=so101_follower --robot.port=COM5 --robot.cameras="{ cam_overhead: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}, cam_ego: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, cam_external: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" --policy.path=tinkerbuggy/le-pickup-pi05 --policy.device=cuda --dataset.repo_id=tinkerbuggy/eval_le-pickup-inference-test4 --dataset.single_task="Pick up the blue cube and place it in the cardboard box" --dataset.num_episodes=1 --dataset.fps=10 --dataset.episode_time_s=120 --dataset.vcodec=h264 --dataset.push_to_hub=false --display_data=true
