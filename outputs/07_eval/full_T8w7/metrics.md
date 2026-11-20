# Test results — run `full_T8w7` (8 views, 7x7 window)

490 test samples (319 with glare on the target view), 37984 object squares, 4933 of them under glare in the target view.

Models are trained; `single + ... vote` rows reuse the trained `single` model on every view (no training).

| model | object IoU (all) | mIoU (all) | F1 (all) | AP (all) | object IoU clean | object IoU glare-on-t | recall under glare |
|---|---|---|---|---|---|---|---|
| single | 0.622 | 0.788 | 0.767 | 0.838 | 0.644 | 0.611 | 0.808 |
| single + OR vote | 0.579 | 0.758 | 0.734 | 0.867 | 0.600 | 0.569 | 0.945 |
| single + 2-of-N vote | 0.643 | 0.799 | 0.783 | 0.878 | 0.651 | 0.639 | 0.910 |
| single + majority vote | 0.666 | 0.815 | 0.800 | 0.871 | 0.659 | 0.671 | 0.839 |
| single + mean vote | 0.671 | 0.817 | 0.803 | 0.877 | 0.664 | 0.674 | 0.850 |
| avg | 0.682 | 0.823 | 0.811 | 0.892 | 0.678 | 0.684 | 0.881 |
| median | 0.647 | 0.804 | 0.786 | 0.871 | 0.640 | 0.651 | 0.820 |
| glareskip | 0.640 | 0.800 | 0.781 | 0.847 | 0.632 | 0.645 | 0.829 |
| noalign | 0.626 | 0.791 | 0.770 | 0.842 | 0.637 | 0.620 | 0.833 |
| beltpulse | 0.666 | 0.814 | 0.799 | 0.871 | 0.672 | 0.662 | 0.850 |

## Per test video
`iron_ore_jaspilito_4` has real glare and motion blur. `iron_ore_4` is ore only, so its object IoU is 0 by definition:
look at its false-alarm squares instead.

| model (object IoU / false-alarm squares) | iron_ore_4 | iron_ore_7_wood_3 | iron_ore_canga3 | iron_ore_jaspilito_4 | iron_ore_things_3 |
|---|---|---|---|---|---|
| single | 0.000 / 220 | 0.555 / 3997 | 0.679 / 1775 | 0.705 / 1001 | 0.654 / 2757 |
| single + OR vote | 0.000 / 731 | 0.541 / 7711 | 0.632 / 3870 | 0.638 / 2142 | 0.598 / 6568 |
| single + 2-of-N vote | 0.000 / 416 | 0.580 / 5068 | 0.713 / 2268 | 0.699 / 1469 | 0.680 / 3580 |
| single + majority vote | 0.000 / 41 | 0.582 / 3062 | 0.757 / 946 | 0.750 / 695 | 0.697 / 1643 |
| single + mean vote | 0.000 / 53 | 0.587 / 3234 | 0.759 / 1050 | 0.756 / 773 | 0.702 / 1797 |
| avg | 0.000 / 25 | 0.595 / 4000 | 0.769 / 1224 | 0.747 / 529 | 0.731 / 1050 |
| median | 0.000 / 29 | 0.564 / 2665 | 0.717 / 1729 | 0.730 / 442 | 0.685 / 1470 |
| glareskip | 0.000 / 21 | 0.581 / 3403 | 0.723 / 1410 | 0.538 / 682 | 0.702 / 2192 |
| noalign | 0.000 / 0 | 0.614 / 2905 | 0.607 / 3842 | 0.678 / 953 | 0.638 / 1680 |
| beltpulse | 0.000 / 518 | 0.601 / 3046 | 0.726 / 1584 | 0.751 / 850 | 0.707 / 1865 |

## Does BeltPulse distrust glared views?
Average attention weight a view receives on object squares (uniform = 1/8 = 0.125):

| where the view is | avg weight | n |
|---|---|---|
| glared | 0.088 | 20933 |
| clean | 0.128 | 282939 |
