import numpy as np

from app.noise_map import GridPoint, build_noise_map
from app.quiet_zone import check_feasibility, zone_of_quiet_diameter
from app.room_model import NoiseSource, QuietZone, RoomModel


def test_build_noise_map_corners():
    points = [GridPoint(0, 0, 60), GridPoint(2, 0, 50), GridPoint(0, 2, 55), GridPoint(2, 2, 45)]
    xi, yi, zi = build_noise_map(points, resolution=0.5)
    assert xi.size > 1 and yi.size > 1
    assert np.all(np.isfinite(zi[~np.isnan(zi)]))


def test_zone_of_quiet_scales_with_frequency():
    assert zone_of_quiet_diameter(100) > zone_of_quiet_diameter(1000)


def test_check_feasibility():
    res = check_feasibility((1.0, 1.0, 0.0), (1.5, 1.0, 0.0), 200.0)
    assert res["verdict"] in {"good", "marginal", "poor"}
    assert res["distance_m"] == 0.5


def test_room_model_roundtrip(tmp_path):
    room = RoomModel(name="lab")
    room.add_source(NoiseSource(id="printer", name="打印机", position_m=(1.0, 2.0, 0.0)))
    room.add_quiet_zone(QuietZone(id="qz1", position_m=(1.0, 1.0, 0.0)))
    path = tmp_path / "room.json"
    room.save(path)
    loaded = RoomModel.load(path)
    assert loaded.name == "lab"
    assert loaded.sources[0].position_m == (1.0, 2.0, 0.0)
