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
python run_real_rollout.py --config-name rollout_sirius
```

## Rollout trained policy with baseline failure detection approach (FAIL-DETECT)
```
python run_real_rollout.py --config-name rollout_FAIL_DETECT 
```

## 🚨 Rollout trained policy with our failure detection method based on OT matching and action inconsistency supervision
```
python run_real_rollout.py --config-name real_failure_detection
```

If you would like to verify the stitched demonstrations by human intervention, please run the following command:

```
python test/replay_demo.py -p ${your demo path}
```