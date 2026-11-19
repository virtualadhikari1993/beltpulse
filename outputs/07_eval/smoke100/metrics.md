# Test results — run `smoke100` (3 views, 3x3 window)

50 test samples (28 with glare on the target view), 3040 object squares, 368 of them under glare in the target view.

Models are trained; `single + ... vote` rows reuse the trained `single` model on every view (no training).

| model | object IoU (all) | mIoU (all) | F1 (all) | AP (all) | object IoU clean | object IoU glare-on-t | recall under glare |
|---|---|---|---|---|---|---|---|
| single | 0.526 | 0.741 | 0.690 | 0.767 | 0.562 | 0.469 | 0.429 |
| single + OR vote | 0.533 | 0.739 | 0.695 | 0.821 | 0.530 | 0.537 | 0.840 |
| single + majority vote | 0.563 | 0.762 | 0.720 | 0.816 | 0.592 | 0.519 | 0.595 |
| single + mean vote | 0.572 | 0.767 | 0.728 | 0.822 | 0.598 | 0.533 | 0.636 |
| avg | 0.529 | 0.739 | 0.692 | 0.799 | 0.526 | 0.533 | 0.807 |
| median | 0.593 | 0.779 | 0.745 | 0.817 | 0.601 | 0.583 | 0.753 |
| glareskip | 0.580 | 0.772 | 0.734 | 0.810 | 0.592 | 0.563 | 0.753 |
| noalign | 0.560 | 0.759 | 0.718 | 0.805 | 0.554 | 0.570 | 0.796 |
| beltpulse | 0.605 | 0.785 | 0.754 | 0.825 | 0.604 | 0.607 | 0.812 |

## Per test video
`iron_ore_jaspilito_4` has real glare and motion blur. `iron_ore_4` is ore only, so its object IoU is 0 by definition:
look at its false-alarm squares instead.

| model (object IoU / false-alarm squares) | iron_ore_4 | iron_ore_7_wood_3 | iron_ore_canga3 | iron_ore_jaspilito_4 | iron_ore_things_3 |
|---|---|---|---|---|---|
| single | 0.000 / 11 | 0.523 / 124 | 0.565 / 178 | 0.445 / 14 | 0.530 / 514 |
| single + OR vote | 0.000 / 15 | 0.583 / 371 | 0.538 / 296 | 0.743 / 59 | 0.479 / 1005 |
| single + majority vote | 0.000 / 0 | 0.560 / 123 | 0.583 / 147 | 0.639 / 14 | 0.544 / 439 |
| single + mean vote | 0.000 / 0 | 0.581 / 145 | 0.591 / 151 | 0.634 / 13 | 0.550 / 429 |
| avg | 0.000 / 1 | 0.545 / 297 | 0.465 / 381 | 0.717 / 22 | 0.518 / 699 |
| median | 0.000 / 0 | 0.569 / 149 | 0.529 / 302 | 0.789 / 29 | 0.600 / 152 |
| glareskip | 0.000 / 9 | 0.561 / 142 | 0.562 / 168 | 0.690 / 48 | 0.578 / 265 |
| noalign | 0.000 / 4 | 0.567 / 195 | 0.469 / 401 | 0.673 / 66 | 0.581 / 340 |
| beltpulse | 0.000 / 5 | 0.559 / 152 | 0.563 / 249 | 0.817 / 26 | 0.611 / 161 |

## Does BeltPulse distrust glared views?
Average attention weight a view receives on object squares (uniform = 1/3 = 0.333):

| where the view is | avg weight | n |
|---|---|---|
| glared | 0.352 | 368 |
| clean | 0.333 | 8752 |
