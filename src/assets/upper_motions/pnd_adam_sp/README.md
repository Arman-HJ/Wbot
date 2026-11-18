# Adam SP upper-body motion pack (FALCON tracking)

Absolute joint angles in **radians**, 14 arm DoF (no waist).

Mapped from FALCON X2 `motions_upper` (ACCAD lift / put-down / gesture clips).
Joint names and hinge sense already match Adam SP; values are clipped to
Adam SP XML limits.

| file | shape | use |
|------|-------|-----|
| `*_upper14.npy` | `(T, 14)` | standing `ref_upper_dof_pos` |
| `playlist.yaml` | — | clip list for `UpperBodyMotionCommand` |
