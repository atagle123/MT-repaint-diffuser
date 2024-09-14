cosas que faltan:
renderer (Mujoco renderer)
dataset (sequence datasets valuedatasets etc)

clase trainer esta buena, probar dataset y ver el renderer


Dataset:
quizas probar añadir un metodo set state a half chetah v4

- instalar gymnasium 1.0.0 a2 y cambiar cosa de minari
- realizar datasets v5 obtenidos de: 
- obtener: https://huggingface.co/datasets/im-Kitsch/minari_d4rl
- pasar v2 to v5 con script

to combine datasets with minari:
minari combine halfcheetah-expert-v0 halfcheetah-medium-expert-v0 halfcheetah-medium-replay-v0 halfcheetah-medium-v0 halfcheetah-random-v0 --dataset-id halfcheetah-all-v0
minari combine walker2d-expert-v0 walker2d-medium-expert-v0 walker2d-medium-replay-v0 walker2d-medium-v0 walker2d-random-v0 --dataset-id walker2d-all-v0