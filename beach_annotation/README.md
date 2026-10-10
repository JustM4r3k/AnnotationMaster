# Beach litter annotation pipeline

This package creates **draft instance annotations for CVAT** from photographs.
It runs Grounding DINO and SAM 2.1 locally on your VM. Qwen2.5-VL is an optional
second opinion on the class. It can also preserve and supplement existing BePLi
COCO annotations.

**Start with 5 photographs to check the installation, then a 50-image pilot.**
All output needs human review. The script does not train the Jetson model or
automatically approve images as training data.

## 1. Folder structure

Extract the ZIP anywhere on the VM. The top folder is `beach_annotation/`.
Run the commands below from that folder. Paths to models and the default config
are resolved relative to the script, so moving the whole project is fine.

| Path inside `beach_annotation/` | Purpose |
| --- | --- |
| `annotate.py` | Main pipeline, with explanatory comments. |
| `download_models.py` | Downloads pretrained model files from their official repositories. |
| `config.json` | Model choices, detection prompts, thresholds and crop settings. |
| `bepli_category_map.json` | Mapping of BePLi v2's 13 source names to your two plastic classes. |
| `requirements.txt` | Pinned Python dependencies. |
| `tests/test_pipeline.py` | Data integrity tests that do not need model weights. |
| `models/grounding-dino-base/` | Created by the model downloader. |
| `models/sam2.1-hiera-large/` | Created by the model downloader. |
| `models/Qwen2.5-VL-7B-Instruct/` | Created only with `--with-qwen`. |
| `models/model_versions.json` | Exact downloaded model revisions. |
| `data/beachlitter/images/` | Put BeachLitter photographs here. Create this folder yourself. |
| `data/bepli/` | Optional: unpack BePLi here, keeping its original structure. |
| `output/` | Separate subfolder for each run; created by the script. |

Keep BeachLitter's old semantic-mask PNGs **outside the input images folder**.
The script processes supported images recursively and cannot distinguish a
photograph from a coloured label PNG by filename alone. Keep public train,
validation and test partitions separate.

## 2. Install on the VM

These instructions target a Linux x86-64 VM with an NVIDIA GPU and Python
3.11 or 3.12. This is the annotation VM setup, not a Jetson installation.

Check the VM first:

```bash
cd beach_annotation
python3 --version
nvidia-smi
```

Create a separate Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The PyTorch command installs the CUDA 12.6 build. Your VM's NVIDIA driver must
support it. If your team's VM already provides a managed GPU environment, check
with whoever maintains it before replacing packages in that environment; the
separate `.venv` above avoids changing it. A separate system CUDA compiler is
not required for this implementation of Grounding DINO.

For a GPU that requires a newer CUDA build, PyTorch also provides this matching
2.7.1/0.22.1 pair using `--index-url https://download.pytorch.org/whl/cu128`.
Use the build appropriate to the VM rather than upgrading packages randomly.

Check that Python can see the GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

`CUDA available` should be `True`. Every new terminal session must reactivate
the environment with `source .venv/bin/activate`.

## 3. Get the pretrained models

You do not train these annotation models yourself. The downloader retrieves
their pretrained weights, tokenizer files and image-processing settings.

| Role | Official Hugging Face repository |
| --- | --- |
| Detector | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) |
| Segmenter | [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large) |
| Optional classifier | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

Download the detector and segmenter:

```bash
python download_models.py
```

To download Qwen as well:

```bash
python download_models.py --with-qwen
```

These are large downloads; Qwen adds several model shards and substantially
more disk and GPU memory use. Internet access is needed during download.
Inference subsequently loads only local files. No paid inference API is used.

The downloader records exact model revisions. Running it again reuses those
revisions and the download cache. Keep the model folders and
`model_versions.json` together when moving the project.

This implementation uses Transformers directly. You do not also need to clone
Grounded-SAM-2, install Meta's SAM repository, or run three separate servers.

## 4. Put your photos in place and check the paths

Create `data/beachlitter/images/` and copy the photographs into it. Keep their
original resolution. Subdirectories and duplicate basenames in different
subdirectories are supported; their relative paths are preserved.

Run a check without model inference:

```bash
python annotate.py run --images data/beachlitter/images --output output/check --limit 5 --dry-run
```

The dry run reads images and checks existing annotations if supplied. It does
not check the GPU or pretrained inference. It does not write output files.

Images carrying an EXIF rotation other than normal are rejected so their masks
cannot silently become misaligned. Normalize those photographs first; existing
annotations must undergo the corresponding coordinate transform as well.

## 5. Run the first batch

Start with five photographs:

```bash
python annotate.py run \
  --images data/beachlitter/images \
  --output output/beachlitter_first5 \
  --limit 5 \
  --pack-images
```

The default device is `cuda`. If your VM has multiple GPUs, select one with
`--device cuda:1`, for example. `--device cpu` exists for debugging but will be
slow with these large models.

Once the five-image run works, use a **new output folder** for the pilot:

```bash
python annotate.py run \
  --images data/beachlitter/images \
  --output output/beachlitter_pilot50 \
  --limit 50 \
  --pack-images
```

`--limit` takes the first N images sorted by relative filename. It is not a
random or representative sampling method. For the real pilot, put 50 varied
photos in a dedicated input directory, including tiny objects, clutter and
apparently clean scenes.

| Output file/folder | What you do with it |
| --- | --- |
| `annotations_draft.json` | Import into CVAT as COCO annotations. |
| `images_for_cvat.zip` | Upload as the task's images. Contains exactly the processed originals; created with `--pack-images`. |
| `previews/` | Quick visual overlays for the first 20 photos. These are not training images. |
| `review.csv` | Every annotation's provisional class and review flags; also lists images with no annotations. |
| `summary.json` | Counts and any images left pending after an interrupted run. |
| `checkpoints/` | Completed results per image, used by `--resume`. |
| `run.json` / `runtime_versions.json` | Settings, model revisions, source hashes and installed versions. |

For the whole collection, omit `--limit` and choose another output folder.
Large collections can be processed in separate input batches to keep CVAT
tasks and local JSON files manageable. The script retains completed annotation
records in memory while assembling a batch's COCO export.

## 6. Optional Qwen classification

After downloading with `--with-qwen`, enable classification like this:

```bash
python annotate.py run \
  --images data/beachlitter/images \
  --output output/beachlitter_qwen5 \
  --limit 5 \
  --qwen \
  --pack-images
```

Qwen receives the original crop, the target box coordinates inside it, and the
seven class definitions. A valid suggestion becomes the draft class. The
detector's original class and disagreements are retained as metadata.

`uncertain`, `not_trash`, or malformed responses do **not** delete the proposal.
It keeps its detector-assigned provisional class and receives a review flag.
There is no eighth training class and no automatic conversion of uncertainty
to `other_trash`. Qwen receives objects found by the detector; it cannot rescue
objects that were never proposed.

All three models share the selected GPU in this implementation. If Qwen causes
an out-of-memory error, start with the baseline command without `--qwen` in a
new output folder. Grounding DINO and SAM run in float32 for compatibility;
Qwen uses bfloat16 or float16 on supported CUDA devices. Actual VRAM use and
throughput must be measured on your VM.

## 7. BePLi: keep existing masks and propose missing objects

Use BePLi **v2**, whose category names match `bepli_category_map.json`.
The mapping assigns `pet_bottle` and `other_bottle` to `plastic_bottle` and
the other 11 plastic categories to `plastic_other`. Unexpected source category
names cause an explicit error, so they cannot be silently mislabelled.

Here is an example after unpacking BePLi. Replace the two input paths with the
actual image root and COCO JSON path in your download:

```bash
python annotate.py run \
  --images data/bepli/plastic_coco/train \
  --existing-coco data/bepli/plastic_coco/annotations/instances_train.json \
  --category-map bepli_category_map.json \
  --output output/bepli_pilot50 \
  --classes can,glass,metal_other,paper_cardboard,other_trash \
  --limit 50 \
  --pack-images
```

**Those BePLi paths are illustrative; the script does not assume them.**
Check one `file_name` value in the COCO JSON. If it is `photo.jpg`, `--images`
must be the folder containing that file. If it is `train/photo.jpg`,
`--images` must be the parent of `train/`. Exact relative paths and image
dimensions must match. Add `--dry-run` to the command to check this first.

Existing mask geometry is preserved. Original category names and annotation
IDs are retained in attributes, while output image and annotation IDs are
regenerated uniquely. Overlapping new proposals are kept and flagged for you
to resolve; a new AI proposal never automatically replaces a source mask.

The five-class prompt selection above focuses on missing non-plastic litter.
Omit `--classes` to search all seven classes, including potentially missed
plastics; this will produce more overlap with existing masks. In either mode,
inspect the whole image for missed objects before marking it complete.

Already remapped seven-class COCO can be supplied without `--category-map`.
Other source taxonomies require an explicit JSON mapping by source class name.
This package includes the BePLi mapping; it does not invent a complete TACO
mapping for ambiguous source categories.

## 8. Review in CVAT

Use [CVAT Online](https://app.cvat.ai/) or your team's own CVAT installation.
CVAT is separate from this Python environment. Self-hosting instructions are
in the [official CVAT documentation](https://docs.cvat.ai/docs/administration/basics/installation/).

1. Create a project with exactly the seven label names below.
2. Create a task within it and upload this run's `images_for_cvat.zip`.
3. Use the task's annotation upload/import action, select **COCO 1.0**, and
   upload `annotations_draft.json`.
4. Open the task's annotation job. Check masks, classes, duplicates, natural
   objects wrongly labelled as trash, and objects the AI missed.
5. Export the corrected dataset as COCO instance annotations. Save that as a
   separate reviewed file; do not overwrite the pipeline's checkpoints.

COCO supports per-instance RLE masks with `iscrowd=0`; these masks preserve
holes and disconnected visible parts. A failed SAM mask is exported as a
flagged box with empty segmentation. Replace it with a valid object mask in
CVAT before segmentation training.

Custom review attributes are included in COCO. Their visibility and round-trip
depend on your CVAT label schema. `review.csv` and the original draft JSON
remain the independent record; do not rely on unknown attributes surviving
import/export. You can add text attributes such as `source`, `review_status`,
`flags`, `qwen_status`, and `qwen_reason` to every CVAT label if you want them
in the editor.

| Your training ID | COCO ID used here | Label |
| ---: | ---: | --- |
| 0 | 1 | `can` |
| 1 | 2 | `plastic_bottle` |
| 2 | 3 | `glass` |
| 3 | 4 | `metal_other` |
| 4 | 5 | `paper_cardboard` |
| 5 | 6 | `plastic_other` |
| 6 | 7 | `other_trash` |

CVAT can regenerate numeric category IDs. Before later YOLO conversion, check
the exported `categories` and normalize by name to the table above. The
validator bundled here expects this canonical mapping and will reject other
ID assignments. This package intentionally stops at draft COCO; only reviewed,
complete images should proceed to your final training export.

An image with no detections remains `needs_review`. It becomes a negative
training image only after someone verifies that it has no target litter.

## 9. Resume, tune and validate

After an interruption, rerun the **same command** with `--resume`. Completed
images are loaded from checkpoints. The unfinished image starts again.
Changed settings, input lists, model revisions or source COCO require a new
output folder. Changed image bytes also invalidate their cached labels.

For example:

```bash
python annotate.py run --images data/beachlitter/images --output output/beachlitter_pilot50 --limit 50 --pack-images --resume
```

Useful settings in `config.json`:

| Setting | Default | Effect |
| --- | ---: | --- |
| `box_threshold` | 0.25 | Lowering it proposes more candidates, generally with more false detections. |
| `text_threshold` | 0.20 | Controls the words retained in detector descriptions. |
| `tile_size` | 1024 | Pixel width/height of additional detector crops; 0 disables tiled detection. |
| `tile_overlap` | 0.25 | Overlap between detector crops. |
| `box_merge_iou` / `mask_merge_iou` | 0.85 | Merge near-identical proposals, retaining competing classes for review. |
| `minimum_crop_size` | 256 | Minimum side length of the context crop passed to SAM. |
| `context_margin` | 0.35 | Extra context around the detector box for SAM. |

These are starting settings, not measured optimal values. The full photograph
is scanned in addition to the tiles. SAM receives a context crop per candidate
so tiny objects are not always reduced with the entire beach image. This is
slower than encoding each full image only once. Cropping cannot recover detail
that was not captured in the photograph.

Detection scores, SAM quality estimates and Qwen responses are not calibrated
probabilities that an annotation is correct. Competing prompts can refer to
the same object. Duplicate merging is a heuristic; inspect crowded scenes.
`other_trash` prompts cover specific examples, not every possible manufactured
object. Expand them when review reveals missing object types.

Check the output's IDs, references, dimensions and generated mask geometry:

```bash
python annotate.py validate --coco output/beachlitter_pilot50/annotations_draft.json
python -m unittest discover -s tests -v
```

The supplied tests cover mask round-trips, coordinate coverage, ambiguous
labels, existing-mask preservation, output integrity and resuming. They use
synthetic images and mocked inference. **The pretrained GPU inference and a
live CVAT import have not been run in the authoring environment.** Your first
five-image run is the installation/integration check; it is also where you
start assessing actual prediction quality.

## References

- [Grounding DINO model and usage](https://huggingface.co/IDEA-Research/grounding-dino-base)
- [SAM2 Transformers 4.57.1 interface](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/sam2)
- [Qwen2.5-VL model and usage](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [PyTorch version-specific installation commands](https://pytorch.org/get-started/previous-versions/)
- [CVAT COCO import/export](https://docs.cvat.ai/docs/dataset_management/formats/format-coco/)
- [BePLi v2 category definitions](https://pubmed.ncbi.nlm.nih.gov/40761540/)
