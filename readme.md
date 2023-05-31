## Abstract
Recent continual learning (CL) has become an interesting and critical topic in the AI community, with the main aim of learning a sequence of tasks without losing previously acquired knowledge. Early efforts focused on solving the catastrophic forgetting (CF) problem behind CL and correspondingly have achieved promising results in overcoming such a thorny. More recent studies are exploring whether learning a sequence of tasks can be facilitated from the perspective of knowledge transfer or knowledge consolidation. However, existing solutions either still confront severe negative knowledge transfer issues or share narrow knowledge between the new and previous tasks. Motivated by human cognitive processes, this paper presents a novel knowledge transfer solution for Continual Learning via Adaptive Neuron Selection (CLANS), which treats the used neurons in earlier tasks as a knowledge pool and makes it scalable via reinforcement learning with a small margin. Subsequently, the adaptive neuron selection enables knowledge transfer for both old and new tasks in addition to overcoming the CF problem. The experimental results conducted on four datasets widely used in CL evaluations demonstrate that CLANS outperforms the state-of-the-art baselines, e.g., up to 2.98\% improvements compared to the best baseline on TinyImageNet regarding ACC.

## Installing

1. Create a python 3 conda environment (check the requirements.txt file)

2. The following folder structure is expected at runtime. From the git folder:
   
   * src/ : Where all the scripts lie (already produced by the repo)
   * dat/ : Place to put/download all data sets
tips：To run experiment with TinyImageNet dataset, you need to download it in http://cs231n.stanford.edu/tiny-imagenet-200.zip and put it in the folder 'src'. 

3. The main script is src/run.py. 


