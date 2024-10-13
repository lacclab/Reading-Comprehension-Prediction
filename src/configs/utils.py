from hydra.core.config_store import ConfigStore

from src.configs.constants import ConfigName


def register_config(group: ConfigName):
    def decorator(cls):
        cs = ConfigStore.instance()
        cs.store(
            group=group.value,
            name=cls.__name__,
            node=cls,
        )
        print(f"Registered {group.value}: {cls.__name__}")
        return cls

    return decorator
