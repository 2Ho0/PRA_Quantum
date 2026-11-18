import h5py
import numpy as np

from qerr.fermidata import load_shots, number_histogram, species_masks


def _write_shots(path, shots_by_time, qubit_order, onsite=4.0):
    """Write a file in the documented format: packed bits per time, attrs original_shape; metadata attrs."""
    with h5py.File(path, "w") as h5:
        g = h5.create_group("data")
        for t, arr in shots_by_time.items():
            d = g.create_dataset(str(t), data=np.packbits(arr.astype(np.uint8), axis=-1))
            d.attrs["original_shape"] = np.array(arr.shape)
        m = h5.create_group("metadata")
        m.attrs["qubit_order"] = np.array(qubit_order, dtype=h5py.string_dtype())
        m.attrs["onsite_term"] = onsite


def test_load_shots_unpacks_bits_and_sorts_times(tmp_path):
    rng = np.random.default_rng(0)
    order = ["(0,0,up)", "(0,0,down)", "(0,1,up)", "(0,1,down)", "(1,0,up)", "(1,0,down)",
             "(1,1,up)", "(1,1,down)", "(2,0,up)", "(2,0,down)"]  # 10 qubits: not a multiple of 8
    a = rng.integers(0, 2, size=(7, 10))
    b = rng.integers(0, 2, size=(7, 10))
    path = tmp_path / "shots.h5"
    _write_shots(path, {0.2: b, 0.1: a}, order)
    times, shots, meta = load_shots(path)
    assert times.tolist() == [0.1, 0.2]
    assert shots.shape == (2, 7, 10) and shots.dtype == np.uint8
    assert np.array_equal(shots[0], a) and np.array_equal(shots[1], b)
    assert meta["qubit_order"] == order and meta["onsite_term"] == 4.0


def test_species_masks_follow_qubit_labels():
    order = ["(0,0,up)", "(0,0,down)", "(1,0,down)", "(1,0,up)"]
    up, down = species_masks(order)
    assert up.tolist() == [True, False, False, True] and down.tolist() == [False, True, True, False]


def test_number_histogram_counts_particles_of_one_species():
    shots = np.array([[1, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.uint8)
    mask = np.array([True, False, True, True])
    assert number_histogram(shots, mask).tolist() == [0, 1, 0, 2]
