import hydra
from omegaconf import DictConfig
from trainer import Trainer


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    print(f'--main--cfg:{cfg}')
    trainer = Trainer(cfg)



if __name__ == '__main__':
    main()
