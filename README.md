# Human-in-the-loop learning with Diffusion Policy

## ⚙️ Installation

```
conda env create -f conda_environment.yaml
```

If you'd like to test on real robot, execute the following command:

```
conda env create -f conda_environment_real.yaml
```

## 🦾 Demo collection, Training, and Evaluation on Real Robot

```
# Teleoperation w/ sigma7
python record.py -o /path/to/data -res {Desired image resolution}

# Training Diffusion Policy in naive/reweighted manner
cd diffusion_policy
bash mytest.sh

# Evaluating Diffusion Policy in real world
python my_eval_real.py

# Rollout trained policy with human intervention and annotate human intervention, pre-intervention samples, and robot rollout samples
python rollout_sirius.py

# Rollout trained policy with baseline failure detection approach (FAIL-DETECT)
python rollout_FAIL_DETECT.py

# Rollout trained policy with our failure detection method based on OT matching and action inconsistency supervision 
python real_failure_detection.py

# Rollout trained policy with predicted trajectory visualized
cd diffusion_policy
python rollout_visualize.py

# Save the annotated predicted trajectory into a video
cd diffusion_policy
python visualize_traj.py
```

If you would like to verify the correctness of human teleoperation, please run the following command:

```
python test/replay_demo.py -p ${your demo path}
```