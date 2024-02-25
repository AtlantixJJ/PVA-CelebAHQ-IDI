#

# Training

We support multi-GPU parallelization with huggingface accelerate.

```bash
# Train the PVA pathway: stage 1
python script/pva_train.py --config config/SDI2_PVA_stage1.json 
# stage 2
python script/pva_train.py --config config/SDI2_PVA_stage2.json --resume expr/celebahq/PVA/stage1/

# Reproducing Textual Inversion baselines on CelebAHQ-IDI-5
python script/finetune.py

# Reproducing Custom Diffusion baselines on CelebAHQ-IDI-5
python script/submit.py --func cdi_train --gpu <0/1>

# Reproducing PVA with 20 steps of finetuning
python script/finetune.py --config config/SDI2_PVA_FT.json --resume expr/celebahq/PVA/stage2/
```

# Acknowledgement

The `Lion` implementation is provided by [Google](https://github.com/google/automl/tree/master/lion).
