from gymnasium.envs.registration import (
    registry,
    register,
    make,
    spec,
)

register(
    id="Biped-custom",
    entry_point="envs.biped_np:Biped",
)
