from .latents import LatentRecord, Neighbour


def set_neighbours(
    record: LatentRecord,
    neighbours: dict[str, list[tuple[float, int]]],
    threshold: float,
):
    """
    Set the neighbours for the latent record.
    """

    latent_neighbours = neighbours[str(record.latent.latent_index)]

    # Each element in neighbours is a tuple of (distance,feature_index)
    # We want to keep only the ones with a distance less than the threshold
    latent_neighbours = [
        neighbour for neighbour in latent_neighbours if neighbour[0] > threshold
    ]

    record.neighbours = [
        Neighbour(distance=neighbour[0], latent_index=neighbour[1])
        for neighbour in latent_neighbours
    ]
