## Overview
- This directory goes over how to instrument model training / evaluation with `Weights & Biases` (`W&B` or `wandb`) to help monitoring/visualization & speed-up debug. 
- Its based on https://learn.deeplearning.ai/courses/evaluating-debugging-generative-ai

## Training Workflow:
- Modify training code. Add hooks to:
    - Initiate `wandb` with appropriate project name & training configs. 
    - Log key metrics (train/val loss, val outputs/images, etc.)
    - Save deisred checkpoints & add them into Artifacts
    - Log the artifacts
- Run training experiments with different configs.
- Use W&B UI to track progress, view metrics, debug, etc.
- Select the desired artifact from the metrics
- Link it to the model registry (a central system to record/share models)

## Evaluation:
- Download the desired artifact
- Get the config and updated
- Use W&B `Tables` to visualize results
    - Text comparison (LLM prompt/output, etc.)
    - Image comparison (Diffusion DDPM/DDIM images, etc.)
- For LLM evaluation:
    - Can use wandb `Tables`
    - Can use wandb`Trace` or Tracing features for more complex chains.

## Images:
- Instrument W&B
    - <img src="images/WB1.png" alt="drawing" width="400"/>   
- Model registry
    - <img src="images/WB2.png" alt="drawing" width="400"/>   
- Table
    - <img src="images/WB3.png" alt="drawing" width="400"/>   
- LLM Train/Finetune monitoring
    - <img src="images/WB4.png" alt="drawing" width="400"/>   
- LLM Eval
    - <img src="images/WB5.png" alt="drawing" width="400"/>   
    - <img src="images/WB6.png" alt="drawing" width="400"/>   
    - <img src="images/WB7.png" alt="drawing" width="400"/>   
    - <img src="images/WB8.png" alt="drawing" width="400"/>   

## References:
- https://learn.deeplearning.ai/courses/evaluating-debugging-generative-ai/
- https://learn.deeplearning.ai/courses/diffusion-models/