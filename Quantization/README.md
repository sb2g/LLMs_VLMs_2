### Overview
- In this directory, we'll go over some of the quantization techniques for LLMs/VLMs.
- What is Quantization?
    - We can reduce the model size & compute requirements by converting weights & activations into lower precision formats.

### High level concept:
- Numeric representations:
    - Integer
        - Unsigned integer: Range: $[0, 2^{n}-1]$
        - Signed integer:
            - Sign magnitude rep: Range: $[-2^{n-1}+1, 2^{n-1}-1]$  
                - 1st bit is only for sign. So, 00..00, 10..00 both represent 0
            - Two's Complement rep: Range: $[-2^{n-1}, 2^{n-1}-1]$  
                - 1st bit is $-2^{n-1}$. So, 00..00 represents 0; 10..00 both represent $-2^{n-1}$
    - Real numbers
        - Fixed point
            - Fixed number of digits are reserved to represent Integer, Fraction.
            - Format is: Signbit,Integer.Fraction
        - Floating point
            - Can represent bigger range of values. Number of digits for Integer, Fraction can vary.
            - Given by the formula
                - For normal numbers, $Exponent$ $\ne$ 0
                ```math
                (-1)^{Sign} * (1 + {Mantissa}) * (2^{Exponent-bias})
                ```
                - For subnormal numbers, $Exponent$ $=$ 0
                ```math
                (-1)^{Sign} * ({Mantissa}) * (2^{1-bias})
                ``` 
                - For $inf$ numbers, $Exponent$ $=$ 255, $Mantissa$ $=$ 0
                - For $NaN$ numbers, $Exponent$ $=$ 255, $Mantissa$ $\ne$ 0
                where $bias$ is usually half of max representation  
            - Here $Exponent$ width => Range; $Mantissa$ width => Precision
            - Popular floating-point precision formats:

                | Format | #Bytes | #Bits | #SignBits | #ExponentBits | #MantissaBits | Notes |
                | --- | --- | --- | --- | --- | --- | --- |
                | FP32/float32 (E8M23) | 4 |  32 | 1 | 8 | 23 | Single precision|
                | FP16/float16 (E5M10) | 2 |  16 | 1 | 5 | 10 | Half precision|
                | BFP16/bfloat16 (E8M7) | 2 |  16 | 1 | 8 | 23 | Half precision |
                | FP8/float8 (E4M3) | 1 |  8 | 1 | 4 | 3 | |
                | FP8/float8 (E5M2) | 1 |  8 | 1 | 5 | 2 | |
                | FP4/float4 (E1M2) | 0.5 |  4 | 1 | 1 | 2 | |
                | FP4/float4 (E2M1) | 0.5 |  4 | 1 | 2 | 1 | |
                | FP4/float4 (E3M0) | 0.5 |  4 | 1 | 3 | 0 | |
                | INT8/int8 (E0M7) | 1 |  8 | 1 | - | 7 | |
                | INT4/int4 (E0M3) | 1 |  0.5 | 1 | - | 3 | |

- Main Computations in deep learning:
    - Add: Linear savings w.r.t number of bits
    - Multiply: Quadratic savings w.r.t number of bits
- Which part of FP is important?
    - Dynamic range (Exponent): Important for training, esp. during initial epochs
    - Precision (Mantessa): Important for inference
     


### Quantization & Calibration concepts:
- Mapping function / Quantization scheme:
    - Maps a value $r$ to its quantized value $r_q$. 
    - Affine quantization scheme:
        ```math
        r_q = round(\frac{r}{S} + Z) \\
        ```
        where $S$, $Z$ are quantization parameters.
        - $S$ is scaling factor. It is usually float32.   
        - $Z$ is zero point. It is the quantized value corresponding to 0 (in non-quantized float32).  
        - $r$ input range is $[a,b]$, output range is $[a_q,b_q]$ (usually the max range of data type). If $r$ is outside the range of $[a,b]$, its usually clipped. 
        ```math
        S = \frac{b - a}{b_q - a_q} ;  
        Z = a_q - \frac{a}{S}
        ```
    - Symmetric affine quantization scheme:
        - $r$ input range is $[-a, a]$. This allows Z=0 & skips additional operations & provides speed-up. E.g. $[-127, 127]$ & drop -128.  
- What components of the model can be quantized
    - Weights, Activations, KV-Cache, sometimes Gradients.
- Granularity of quantization:
    - Per-tensor quantization: Compute $(S, Z)$ per tensor.
    - Per-channel quantization: Compute $(S, Z)$ per each channel of the tensor. More memory, better accuracy.
- Calibration:
    - We need to determine $[a,b]$ range for the model's components. This step is called Calibration.
    - Some common calibration techniques are:
        - Use observed Min-max: Works well for weights.
        - Use observed moving average Min-Max: Works well with activations
        - Plot histogram with min, max: Then minimize error between input & quantized distributions using Entropy, MeanSquareError, KL divergence, Percentiles, etc.
    - We may have to remove outliers so that important parameter ranges get enough bins. 

### Types of Quantization techniques:

- Post-Training Quantization (PTQ)
    - Here we load a trained model & perform quantizaton after training.
    - Easy to calibrate weights ahead of time since we know the range. But less clear for activations.
    - 2 types of PTQ:
        - Dynamic PTQ
            - Weights are quantized before inference ahead of time.  
            - Activation ranges are determined & activations are rescaled on the fly during runtime, just before the computation.
            - Pros
                - Simpler. No calibraton dataset is required. 
                - Good for models with varying activation distributions. Also, if model latency is dominated by memory bandwidth.
            - Cons
                - Slower inference. Runtime overhead. 
                - May not be available on all hardware.
        - Static PTQ
            - Weights and activations are quantized offline before inference ahead of time.              
            - Pros
                - Faster inference. No runtime overhead to quantize activations during runtime.
                - Good for models where both memory bandwight & compute savings are important.
            - Cons
                - More effort. Requires calibraton dataset for activation statistics.
                - Accuracy degrades if input dataset distribution shifts.
- Quantization-Aware Training (QAT)
    - Here, the training is done in full precision. But, weights & activations are "fake quantized" during training.
    - Pros:
        - Higher accuracy/quality models.
    - Cons:
        - Expensive to retrain. (e.g. LLMs)
- Low-precision fine-tuning, like QLoRA
    - Here, Quantization is combined with fine-tuning.
    - In the case of QLoRA: Quantization is combined with Parameter efficient fine-tuning (PEFT) technique LoRA (Low-rank Adaptation). The base model is usually quantized to lower precision (NF4, details are shown below) and LoRA parameters are finetuned.


### Steps to perform Quantization:
1. Try Dynamic PTQ. If inference speed is OK, go to step 3.
2. Try Static PTQ. Choose a calibration technique & perform it.
3. Check accuracy. If OK, stop.
4. Try QAT. Choose a calibration technique & perform it.
5. Check accuracy. If OK, stop.

### Some popular Quantization techniques:
- Dynamic PTQ with Pytorch
- ZeroQuant (for LLMs). Dynamic PTQ.
- bitsandbytes. Dynamic PTQ.
- SmoothQuant. Dynamic/Static PTQ.
    - 8-bit quantization (W8A8).
    - Addresses activation outliers. Redistribute this quantization difficulty from activations to weights.
- Layer-wise quantization.
    - Different bit precisions for different layers based on their importance. (Critical layers use higher precision)
- Generative Pre-trained Transformer Quantization (GPTQ). Static PTQ. 
    - Reduces the size of the models significantly with minimal accuracy drop. Can compresses to 3-bit or 4-bit.
    - Uses a form of layer-wise quantization and quantizes layers in batches to minimize MSE between layer & its quantized layer outputs. Uses calibration dataset for this.
    - Uses approximate second order information of loss function (Hessian) to identify critical weights.
    - Uses mixed precision: low precision for weights & higher for activations (e.g. INT4, FP16). Model's weights are dequantized during inference computation. 
    - Other techniques/observations like: Order of quantizing weights does not matter, Processes columns of weights in batches to speed-up quantization.
- Activation-aware Weight Quantization (AWQ). Static PTQ.
    - Not all weights are equally important. So, it preserves a small set of salient weights(~1%) , & quantizes the rest.
    - Also, quantize the weights based on data. These salient weights are selected based on activation statistics on a calibration dataset.
- Half-quadratic Quantization (HQQ). Dynamic PTQ.
- QLoRA, NormalFloat (NF4) & Double quantization (DQ):
    - NF4: A special 4-bit format to store weights. Normalizes each weight to [-1,1] and then quantize
    - DQ: Quantize the scaling factors of each quantization block.
    - Uses bfloat16 for computations in forward/back prop. 
- GGML (Georgi Gerganov Machine learning) / GGUF (GPT-Generated Unified format) formats:
    - GGML: C-based ML library to quantize & save models in GGML binary format to run Llama models on CPU. Most important weights are quantized to higher precision & rest to lower precision. 
    - GGUF: Successor to GGML. Enables quantization for non Llama models.
    - Needs llama.cpp library to run .GGML/.GGUF models.

- 

### References:
1. https://huggingface.co/docs/optimum/concept_guides/quantization
2. https://hanlab.mit.edu/courses/2024-fall-65940
3. https://www.youtube.com/watch?v=kw7S-3s50uk
