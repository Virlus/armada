# Human-in-the-loop learning with Diffusion Policy

## ⚙️ Installation

```
conda env create -f conda_environment.yaml
```

If you'd like to test on real robot, execute the following command:

```
conda env create -f conda_environment_real.yaml
```

## 📷 Demo collection on real robot on Flexiv Rizon4 robot and Sigma7 haptic teleoperation

```
python record.py -o /path/to/data -res {Desired image resolution}
```

## Training Diffusion Policy with Dino v2 visual encoder in a DAgger-style or reweighted manner
```
cd diffusion_policy
bash mytest.sh
```

## Evaluating trained policy in real world
```
python my_eval_real.py
```

## Rollout trained policy with human-in-the-loop intervention according to Sirius
```
python rollout_sirius.py
```

## Rollout trained policy with baseline failure detection approach (FAIL-DETECT)
```
python rollout_FAIL_DETECT.py
```

## 🚨 Rollout trained policy with our failure detection method based on OT matching and action inconsistency supervision
```
python real_failure_detection.py
```

## [Deprecated] Rollout trained policy with predicted trajectory visualized (need calibration)
```
cd diffusion_policy
python rollout_visualize.py
```

## [Deprecated] Save the annotated predicted trajectory into a video
```
cd diffusion_policy
python visualize_traj.py
```

If you would like to verify the stitched demonstrations by human intervention, please run the following command:

```
python test/replay_demo.py -p ${your demo path}
```