# Per-object decisions — run `smoke`

Test tracks: 14 objects + 8 ore regions (test videos never seen in training). Up to 24 frames per track, every 8th frame.

| method | glare per frame | impurities found | false alarms on ore | correct type (impurities) | correct (all) |
|---|---|---|---|---|---|
| SeqTransformer (ours) | 0% | 14/14 | 0/8 | 0.93 | 0.95 |
| MeanPool (average) | 0% | 14/14 | 0/8 | 0.93 | 0.95 |
| FrameMLP + OR | 0% | 14/14 | 0/8 | 1.00 | 1.00 |
| FrameMLP + k-of-N (30%) | 0% | 14/14 | 0/8 | 1.00 | 1.00 |
| FrameMLP + mean | 0% | 14/14 | 0/8 | 1.00 | 1.00 |
| SeqTransformer (ours) | 50% | 14/14 | 0/8 | 0.93 | 0.95 |
| MeanPool (average) | 50% | 13/14 | 0/8 | 0.93 | 0.95 |
| FrameMLP + OR | 50% | 14/14 | 3/8 | 0.93 | 0.82 |
| FrameMLP + k-of-N (30%) | 50% | 14/14 | 0/8 | 0.93 | 0.95 |
| FrameMLP + mean | 50% | 14/14 | 0/8 | 0.93 | 0.95 |

## Counting on the test videos

Every 6th frame of each test video, detector = `smoke100/single`. **Lamp** = a fixed glare spot painted on every frame (does not move with the belt).

| video | lamp | true objects | raw detections (frames x blobs) | tracks (>= 3 detections) | counted impurities (transformer) | counted by type |
|---|---|---|---|---|---|---|
| iron_ore_7_wood_3 | no | 3 (wood 3) | 119 | 6 | 4 | plastic 1, wood 3 |
| iron_ore_7_wood_3 | yes | 3 (wood 3) | 143 | 9 | 4 | wood 4 |
| iron_ore_canga3 | no | 6 (canga 4, plastic 2) | 61 | 6 | 5 | tool 1, canga 3, plastic 1 |
| iron_ore_canga3 | yes | 6 (canga 4, plastic 2) | 77 | 5 | 5 | tool 1, plastic 1, canga 3 |
| iron_ore_jaspilito_3 | no | 2 (jaspilito 2) | 60 | 6 | 2 | canga 1, jaspilito 1 |
| iron_ore_jaspilito_3 | yes | 2 (jaspilito 2) | 88 | 9 | 4 | canga 1, jaspilito 3 |
| iron_ore_things_3 | no | 3 (wood 1, tool 2) | 51 | 3 | 3 | wood 1, plastic 1, tool 1 |
| iron_ore_things_3 | yes | 3 (wood 1, tool 2) | 76 | 6 | 4 | wood 2, plastic 1, tool 1 |
| iron_ore_4 | no | 0 (none) | 11 | 1 | 1 | canga 1 |
| iron_ore_4 | yes | 0 (none) | 22 | 1 | 0 | - |
