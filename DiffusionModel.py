from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

class diffusionModel(object):
    def __init__(self) -> None:
        pass

    def trainBaseDDM(self, train_path):


        model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8)
        ).cuda()

        diffusion = GaussianDiffusion(
            model,
            image_size = 128,
            timesteps = 1000,   # number of steps
            loss_type = 'l1'    # L1 or L2
        ).cuda()

        trainer = Trainer(
            diffusion,
            train_path,
            train_batch_size = 32,
            train_lr = 1e-4,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True                        # turn on mixed precision
        )

        trainer.train()
        return None

    def trainVariationDDM(self, train_path):
        return None

    def predictGaussian(self, dimension: tuple):
        return None
