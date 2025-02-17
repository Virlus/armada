# Human-in-the-loop learning with Diffusion Policy

```
# Teleoperation w/ sigma7
python record.py 

# Training Diffusion Policy
cd third_party/diffusion_policy
bash mytest.sh

# Evaluating Diffusion Policy in real world
python my_eval_real.py
```

If you would like to verify the correctness of human teleoperation, please run the following command:

```
python replay_demo.py -p ${your demo path}
```