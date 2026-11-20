# Test results — run `full` (3 views, 3x3 window)

508 test samples (263 with glare on the target view), 38526 object squares, 4388 of them under glare in the target view.

Models are trained; `single + ... vote` rows reuse the trained `single` model on every view (no training).

| model | object IoU (all) | mIoU (all) | F1 (all) | AP (all) | object IoU clean | object IoU glare-on-t | recall under glare |
|---|---|---|---|---|---|---|---|
| single | 0.583 | 0.767 | 0.737 | 0.813 | 0.609 | 0.562 | 0.637 |
| single + OR vote | 0.585 | 0.764 | 0.738 | 0.856 | 0.598 | 0.574 | 0.911 |
| single + majority vote | 0.615 | 0.786 | 0.762 | 0.841 | 0.623 | 0.609 | 0.750 |
| single + mean vote | 0.620 | 0.788 | 0.765 | 0.847 | 0.626 | 0.614 | 0.778 |
| avg | 0.658 | 0.810 | 0.794 | 0.868 | 0.657 | 0.659 | 0.850 |
| median | 0.621 | 0.788 | 0.766 | 0.855 | 0.628 | 0.616 | 0.873 |
| glareskip | 0.630 | 0.794 | 0.773 | 0.839 | 0.621 | 0.637 | 0.814 |
| noalign | 0.615 | 0.784 | 0.762 | 0.834 | 0.618 | 0.612 | 0.789 |
| beltpulse | 0.635 | 0.798 | 0.777 | 0.849 | 0.634 | 0.636 | 0.769 |

## Per test video
`iron_ore_jaspilito_4` has real glare and motion blur. `iron_ore_4` is ore only, so its object IoU is 0 by definition:
look at its false-alarm squares instead.

| model (object IoU / false-alarm squares) | iron_ore_4 | iron_ore_7_wood_3 | iron_ore_canga3 | iron_ore_jaspilito_4 | iron_ore_things_3 |
|---|---|---|---|---|---|
| single | 0.000 / 45 | 0.550 / 3458 | 0.697 / 1533 | 0.628 / 441 | 0.544 / 4907 |
| single + OR vote | 0.000 / 70 | 0.554 / 6004 | 0.702 / 2422 | 0.742 / 793 | 0.520 / 8159 |
| single + majority vote | 0.000 / 24 | 0.581 / 2786 | 0.740 / 1066 | 0.654 / 325 | 0.573 / 4434 |
| single + mean vote | 0.000 / 30 | 0.587 / 2970 | 0.745 / 1121 | 0.678 / 343 | 0.572 / 4741 |
| avg | 0.000 / 164 | 0.608 / 2357 | 0.726 / 1724 | 0.725 / 518 | 0.658 / 2698 |
| median | 0.000 / 5 | 0.550 / 5252 | 0.710 / 2310 | 0.735 / 832 | 0.620 / 3209 |
| glareskip | 0.000 / 198 | 0.559 / 2283 | 0.729 / 1802 | 0.716 / 559 | 0.629 / 3457 |
| noalign | 0.000 / 15 | 0.590 / 3456 | 0.612 / 4098 | 0.704 / 855 | 0.620 / 3035 |
| beltpulse | 0.000 / 91 | 0.572 / 3407 | 0.730 / 1469 | 0.699 / 448 | 0.638 / 2084 |

## Does BeltPulse distrust glared views?
Average attention weight a view receives on object squares (uniform = 1/3 = 0.333):

| where the view is | avg weight | n |
|---|---|---|
| glared | 0.160 | 4388 |
| clean | 0.340 | 111190 |
