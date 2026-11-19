# Test results — run `smoke40_T8w7` (8 views, 7x7 window)

40 test samples (28 with glare on the target view), 1859 object squares, 475 of them under glare in the target view.

Models are trained; `single + ... vote` rows reuse the trained `single` model on every view (no training).

| model | object IoU (all) | mIoU (all) | F1 (all) | AP (all) | object IoU clean | object IoU glare-on-t | recall under glare |
|---|---|---|---|---|---|---|---|
| single | 0.384 | 0.669 | 0.555 | 0.556 | 0.350 | 0.390 | 0.606 |
| single + OR vote | 0.375 | 0.649 | 0.545 | 0.653 | 0.229 | 0.414 | 0.918 |
| single + 2-of-N vote | 0.526 | 0.746 | 0.689 | 0.731 | 0.444 | 0.541 | 0.760 |
| single + majority vote | 0.467 | 0.717 | 0.636 | 0.715 | 0.423 | 0.473 | 0.564 |
| single + mean vote | 0.484 | 0.726 | 0.652 | 0.734 | 0.427 | 0.493 | 0.589 |
| avg | 0.381 | 0.671 | 0.552 | 0.600 | 0.310 | 0.394 | 0.486 |
| median | 0.411 | 0.688 | 0.583 | 0.665 | 0.407 | 0.412 | 0.463 |
| glareskip | 0.507 | 0.738 | 0.672 | 0.728 | 0.466 | 0.513 | 0.646 |
| noalign | 0.321 | 0.637 | 0.485 | 0.508 | 0.364 | 0.313 | 0.383 |
| beltpulse | 0.315 | 0.637 | 0.480 | 0.542 | 0.356 | 0.309 | 0.505 |

## Per test video
`iron_ore_jaspilito_4` has real glare and motion blur. `iron_ore_4` is ore only, so its object IoU is 0 by definition:
look at its false-alarm squares instead.

| model (object IoU / false-alarm squares) | iron_ore_4 | iron_ore_7_wood_3 | iron_ore_canga3 | iron_ore_things_3 |
|---|---|---|---|---|
| single | 0.000 / 348 | 0.428 / 86 | 0.490 / 179 | 0.391 / 52 |
| single + OR vote | 0.000 / 1173 | 0.561 / 242 | 0.505 / 646 | 0.529 / 201 |
| single + 2-of-N vote | 0.000 / 294 | 0.654 / 61 | 0.636 / 219 | 0.494 / 52 |
| single + majority vote | 0.000 / 107 | 0.496 / 14 | 0.565 / 55 | 0.387 / 12 |
| single + mean vote | 0.000 / 114 | 0.483 / 11 | 0.600 / 55 | 0.407 / 8 |
| avg | 0.000 / 193 | 0.436 / 22 | 0.470 / 68 | 0.333 / 10 |
| median | 0.000 / 101 | 0.307 / 17 | 0.533 / 62 | 0.374 / 3 |
| glareskip | 0.000 / 119 | 0.402 / 38 | 0.673 / 99 | 0.429 / 13 |
| noalign | 0.000 / 239 | 0.267 / 38 | 0.409 / 135 | 0.343 / 44 |
| beltpulse | 0.000 / 166 | 0.220 / 38 | 0.408 / 63 | 0.334 / 11 |

## Does BeltPulse distrust glared views?
Average attention weight a view receives on object squares (uniform = 1/8 = 0.125):

| where the view is | avg weight | n |
|---|---|---|
| glared | 0.146 | 1314 |
| clean | 0.123 | 13558 |
