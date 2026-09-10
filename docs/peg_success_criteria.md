# Peg insertion success criteria

The co-training real environment and digital twin share `PegInsertionGeometry`.
By default, success requires all of the following for three consecutive steps:

- Actual TCP horizontal distance from the target, measured in the insertion
  frame, is at most 10 mm (Euclidean XY distance, not 10 mm per axis).
- Absolute insertion-axis position error is at most 10 mm.
- Overall relative rotation angle is at most 5 degrees, not 5 degrees per axis.

`success_angle` uses radians internally. Explicit `peg_config` values override
these defaults. This changes the task success label and terminal reward timing;
it does not change the dense reward formula, target pose, controller, workspace
limits or force/torque protections. Pose proximity is a proxy for insertion,
not an independent measurement of physical engagement.

CPU-only regression checks, without simulator or robot initialization:

```bash
python -m unittest discover -s tests/unit_tests -p test_peg_insertion_geometry.py
```
