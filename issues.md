~~se3 + ddpm = Non-positive determinant in test~~

~~se3 diffusion type 1 predict 0 in tranlation~~


TODO
Block Lifting
Block Stacking

ddpm scaling factor test(unet_film)?
~~diff vs no-diff~~
~~abs xyz as feat for pc encoder~~
diffusion steps: 30 10
no noise diffusion?
6 vs 5 no lr
rot + trans augmentation!

local cond? k in eef enc?


no unetdiffusion w/ se3 data
no unetdiffusion w/ rotated_se3 data: se3: yes, but translation/offset is also affected by rel. rotation?! solution: **separate** ori pos net!?

sep2 w/ k=1 global_cond=1 local_cond=1: k was set to 3, **bad** result
sep2 w/ k=1

sep2 w/ ddpm predict noise  k=1 global_cond=1 local_cond=1
sep2 w/ ddpm predict target  k=1 global_cond=1 local_cond=1


unet_equivariance=True
    no unetdiffusion w/ se3 data
    no unetdiffusion w/ rotated_se3 data

    sep2 w/ k=1 global_cond=1 local_cond=1
    sep2 w/ k=1

    sep2 w/ ddpm predict noise
    sep2 w/ ddpm predict target
