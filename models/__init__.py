from .base import Model, Part, assemble
from .builders import (
    eiffel_tower,
    great_pyramid,
    burj_khalifa,
    heart,
    dna_helix,
    inline_four,
)

MODELS = {
    "eiffel": eiffel_tower,
    "pyramid": great_pyramid,
    "burj": burj_khalifa,
    "heart": heart,
    "dna": dna_helix,
    "engine": inline_four,
}
