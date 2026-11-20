# Test results — run `full_s1` (3 views, 3x3 window)

508 test samples (263 with glare on the target view), 38526 object squares, 4388 of them under glare in the target view.

Models are trained; `single + ... vote` rows reuse the trained `single` model on every view (no training).

| model | object IoU (all) | mIoU (all) | F1 (all) | AP (all) | object IoU clean | object IoU glare-on-t | recall under glare |
|---|---|---|---|---|---|---|---|
| avg | 0.599 | 0.774 | 0.749 | 0.838 | 0.609 | 0.591 | 0.864 |
| beltpulse | 0.629 | 0.794 | 0.773 | 0.842 | 0.618 | 0.639 | 0.801 |

## Per test video
`iron_ore_jaspilito_4` has real glare and motion blur. `iron_ore_4` is ore only, so its object IoU is 0 by definition:
look at its false-alarm squares instead.

| model (object IoU / false-alarm squares) | iron_ore_4 | iron_ore_7_wood_3 | iron_ore_canga3 | iron_ore_jaspilito_4 | iron_ore_things_3 |
|---|---|---|---|---|---|
| avg | 0.000 / 27 | 0.548 / 5182 | 0.718 / 1794 | 0.701 / 1021 | 0.562 / 5363 |
| beltpulse | 0.000 / 102 | 0.563 / 2745 | 0.746 / 1183 | 0.643 / 473 | 0.637 / 2979 |

## Does BeltPulse distrust glared views?
Average attention weight a view receives on object squares (uniform = 1/3 = 0.333):

| where the view is | avg weight | n |
|---|---|---|
| glared | 0.126 | 4388 |
| clean | 0.342 | 111190 |
