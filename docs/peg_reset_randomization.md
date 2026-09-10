# Peg insertion reset distribution

The shared `PegInsertionGeometry` used by the digital twin and co-training real
environment samples X and Y independently and uniformly within ±5 cm in the
insertion frame, at a fixed 10 cm offset above the fully inserted target. Rotation
is sampled uniformly within ±10 degrees about the insertion axis; there is no
additional roll/pitch randomization. `random_yaw` still accepts radians internally;
the default is expressed as `np.deg2rad(10.0)`.

The standalone geometry's default XY safety bound is ±5 cm so that its default
reset distribution is valid. Existing peg experiment YAMLs explicitly use ±15 cm
and retain that bound. YAML `peg_config` overrides take precedence over defaults.
The board, target pose, controller gains and action scales are unchanged.

CPU regression test (from the repository with its Python environment):

```bash
python -m unittest discover -s tests/unit_tests -p test_peg_insertion_geometry.py
```

The test checks default construction, fixed height, rotation axis, sampling bounds,
coverage and safety clipping for 1,024 reset poses, without creating a simulator or
connecting to a robot.
