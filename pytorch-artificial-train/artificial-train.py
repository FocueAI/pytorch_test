import hydra
from omegaconf import DictConfig


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    print(f'cfg:{cfg}')


if __name__ == '__main__':
    main()
