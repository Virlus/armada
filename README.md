# Human-in-the-loop learning with Diffusion Policy

```
# Teleoperation w/ sigma7
python record.py -o /path/to/data -res {Desired image resolution}

# Training Diffusion Policy in naive/reweighed manner
cd third_party/diffusion_policy
bash mytest.sh

# Evaluating Diffusion Policy in real world
cd third_party/diffusion_policy
python my_eval_real.py

# Rollout trained policy with human intervention in DAgger fashion
cd third_party/diffusion_policy
python rollout_dagger.py

# Rollout trained policy with human intervention and Sirius-style reweighing mechanism
cd third_party/diffusion_policy
python rollout_sirius.py

# Rollout trained policy with predicted trajectory visualized
cd third_party/diffusion_policy
python rollout_visualize.py

# Save the annotated predicted trajectory into a video
cd third_party/diffusion_policy
python visualize_traj.py
```

If you would like to verify the correctness of human teleoperation, please run the following command:

```
python replay_demo.py -p ${your demo path}
```