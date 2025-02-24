# Psych LLM

## Setup
```bash
# from root of repository 
# make virtual environment 
python3 -m venv venv
source venv/bin/activate

# install requirements
pip install -r requirements.txt
```


## Framing

Evaluate and explore psychological latent variables from large psychological assays using LLMs.

**Goal**: We have large transcripts for individuals $y_1, \dots, y_N$.

- We can estimate "multi-task classifications" on questions and answers conditioned on $y_i$ as $a_{ij} \sim P_{LLM}(a_{ij} | y_i + q_j)$
- Where $q_1, \dots, q_M$ is a list of questions.
- We want to find some $\hat x_i = \hat x(y_i)$ s.t. 
```
example of \hat x: 

"""
Your name is ____ and 
1. You are ___ percentile in openness. This is exemplified by: 
[FILL IN]
3. You are ___ percentile in conscientiousness. This is exemplified by: 
[FILL IN]
4. You are ___ percentile in extraversion. This is exemplified by: 
[FILL IN]
5. You are ___ percentile in agreeableness. This is exemplified by: 
[FILL IN]
6. You are ___ percentile in neuroticism. This is exemplified by: 
[FILL IN]
"""
```

- For a given template $\hat x$, we fill it in for each participant $\hat x_i = \hat x (y_i)$. Then, we compute 
$$
x^* = \arg \min_{\hat x}
D_{KL}\big(P_{LLM}(a_{ij} | y_i + q_j) \| P_{LLM}(a_{ij} | \hat x(y_i) + q_j\big)
$$

High-level components: 
1. `get_answers()`: function that generates answers from full transcripts $a_{ij} \sim P_{LLM}(a_{ij} | y_i + q_j)$. 
	1. Question template: string stored in some file, has a space for the transcript and the question. 
	2. Dictionaries, lists, JSON are your friends. 
2. `fill_in_latents()`: function that fills in $\hat x_i = \hat x(y_i)$. 
	1. Define a initial $\hat x$ sorta thing that contains the info needed to fill in the prompt s.t. $P_{LLM}(\hat x_i | y_i + \hat x)$. 
3. `get_distance(y_i, hat_x_i, [q_1, ..., q_M], a_ij=...)`: Computes the KL divergence between the model conditioned on the full transcript versus the model conditioned on the extracted latent values `hat_x_i`.
4. `evolution_loop()`: stochastic local search over potential hat_x to optimize `get_distance()`. 
5. `generate_new_template()`: Given the current template and all the questions, answers, log probs, outputs, ask the LLM 

## Sketch Code

 - `sketch/compute_loglik_api.py` -- together.ai API.
 - `sketch/compute_kl_local.py` -- with huggingface transformers (llama-3.2 1b or 3b or smth). 