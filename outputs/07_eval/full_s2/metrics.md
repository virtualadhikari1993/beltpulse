# Test results — run `full_s2` (3 views, 3x3 window)

508 test samples (263 with glare on the target view), 38526 object squares, 4388 of them under glare in the target view.

Models are trained; `single + ... vote` rows reuse the trained `single` model on every view (no training).

| model | object IoU (all) | mIoU (all) | F1 (all) | AP (all) | object IoU clean | object IoU glare-on-t | recall under glare |
|---|---|---|---|---|---|---|---|
| avg | 0.604 | 0.778 | 0.753 | 0.835 | 0.608 | 0.600 | 0.869 |
| beltpulse | 0.614 | 0.784 | 0.761 | 0.841 | 0.614 | 0.615 | 0.832 |

## Per test video
`iron_ore_jaspilito_4` has real glare and motion blur. `iron_ore_4` is ore only, so its object IoU is 0 by definition:
look at its false-alarm squares instead.

| model (object IoU / false-alarm squares) | iron_ore_4 | iron_ore_7_wood_3 | iron_ore_canga3 | iron_ore_jaspilito_4 | iron_ore_things_3 |
|---|---|---|---|---|---|
| avg | 0.000 / 29 | 0.569 / 3272 | 0.710 / 2238 | 0.756 / 617 | 0.539 / 6269 |
| beltpulse | 0.000 / 108 | 0.557 / 3939 | 0.637 / 3582 | 0.738 / 539 | 0.637 / 2843 |

## Does BeltPulse distrust glared views?
Average attention weight a view receives on object squares (uniform = 1/3 = 0.333):

| where the view is | avg weight | n |
|---|---|---|
| glared | 0.194 | 4388 |
| clean | 0.339 | 111190 |
