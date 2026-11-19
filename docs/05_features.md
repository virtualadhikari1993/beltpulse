# 05 — CNN features (frozen ResNet-18)

**Script:** `src/05_extract_features.py [--limit N]`
**Output:** `data/features/<tag>/*.npy`, `data/samples/subset_<tag>.json`, `outputs/05_features/<tag>/`
(`<tag>` = `smoke100` for the 100-sample test run, `full` for everything)

## In plain words
A photo is just colours of pixels. That is a poor description: the same piece of wood gives totally different
pixel values in glare, in shadow or when blurred, and 640 × 352 × 3 ≈ 675,000 numbers per photo is too much for a small model.

ResNet-18 converts the photo into a **description per grid square**: 40 × 22 squares, each a list of **256 numbers**
saying what the patch looks like (colour, edges, texture). Similar-looking patches get similar lists.

```
photo 640×352 ──ResNet-18 (frozen)──► 256 × 22 × 40 numbers
```

## Why a pre-trained ResNet ("transfer learning")
- It was trained by others on 1.2 million everyday photos (ImageNet), so it already knows general visual building blocks: edges, textures, blobs, colours.
- We have only ~1,000 training examples with objects, far too few to learn those basics from scratch.
- We **do not change** ResNet ("frozen"). We only use the first part of it (up to `layer3`), which shrinks the photo 16× and outputs 256 numbers per square.

## Why save to disk ("feature caching")
ResNet is frozen, so a photo always gives the same numbers. Training (step 06) loops over the data ~30 times.
Running ResNet once and saving the result makes training fast and lets it fit in the 4 GB GPU.
A photo with fake glare is a different photo, so it gets its own file (the name ends in `__g<cx>_<cy>_<r>_<a>`).

## Smoke test vs full run
`--limit 100` picks 100 train samples (half with objects, half ore-only), 25 val and 50 test → 520 photos,
234 MB, about 1 minute. Without `--limit`, all 3,694 samples are used (~9,000 photos, ~4 GB, ~15 min).

## Picture: what does ResNet "see"? (`what_resnet_sees_XX.jpg`)
256 numbers can't be drawn. PCA squeezes them to the 3 most important directions, shown as red/green/blue.
**Same colour = ResNet thinks those squares look alike.**
In `what_resnet_sees_02.jpg` the wooden block shows up as its own patch (green outline, pink inside), the ore is mostly purple/blue,
and the steel frame has its own colours. ResNet already separates "different stuff" before we train anything.
