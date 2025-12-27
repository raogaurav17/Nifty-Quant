import hydra
from omegaconf import DictConfig, OmegaConf

from nifty_quant.bootstrap.config_schema import AppConfig

# from nifty_quant.application.run_backtest import run_backtest


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Application entry point.
    - Loads Hydra config
    - Validates it against typed schema
    - Delegates execution to application layer
    """

    # Convert Hydra config to plain dict
    cfg_dict = OmegaConf.to_object(cfg)

    # Typed, validated config (fails fast if invalid)
    app_cfg = AppConfig(**cfg_dict)

    # Run application use case
    # run_backtest(app_cfg)


if __name__ == "__main__":
    main()
