from .base import Model, Part, assemble
from .builders import eiffel_tower, great_pyramid, burj_khalifa, heart, dna_helix as dna_old, inline_four as engine_old

MODELS: dict[str, object] = {
    "eiffel": eiffel_tower,
    "pyramid": great_pyramid,
    "burj": burj_khalifa,
    "heart": heart,
    "dna": dna_old,
    "engine": engine_old,
}


def register_all_models():
    """Register all 33 models from the full builders. Import heavy, call once."""
    from .wonders import WONDERS as W
    from .anatomy import ANATOMY as A
    from .anatomy2 import ANATOMY2 as A2, BODY as BD
    from .biology import BIOLOGY as B
    from .engines import ENGINES as E
    from .vehicle_machines import VEHICLES as V, MACHINES as M
    for group in [W, A, A2, BD, B, E, V, M]:
        for m in group:
            key = m["name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            if key not in MODELS:
                MODELS[key] = m["build"]