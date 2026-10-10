# Beach litter annotation pipeline

**Updated after the first five-image BeachLitter pilot.** For an existing
installation, start with [RERUN.md](RERUN.md). Reuse your downloaded models and
Python environment; run the updated code into a new output folder.

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
| `tests/test_regressions.py` / `tests/fixtures/` | Regressions for the observed pilot failures; saved predictions are not ground truth. |
| `RERUN.md` | Upgrade steps and the five-photo rerun command. |
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
| `annotations_draft.json` | Complete review master for CVAT, including uncertain and suspected-background proposals. |
| `images_for_cvat.zip` | Upload as the task's images. Contains exactly the processed originals; created with `--pack-images`. |
| `previews/` | Three aligned panels per photo: candidates, uncertain/existing, and background to review. Background masks use outlines only. |
| `review.csv` | All proposals, review groups, flags, Qwen reasons and scores; also lists images with no annotations. `image_object_id` matches preview numbers. |
| `review_candidates.csv`, `review_uncertain.csv`, `review_background_review.csv`, `review_existing.csv` | Separate review queues; they are not training datasets. |
| `review_groups.json` | The annotation IDs belonging to each queue in the complete COCO master. |
| `summary.json` | Counts by review group and Qwen status, plus any pending images. |
| `qwen_error.json` | Written if Qwen still produces invalid output after retries; contains the image, target box and raw answers. |
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

Qwen receives a tight unmarked target crop, a second context crop with a red
outline identifying the same target, and the seven class definitions. The
outline avoids relying on pixel coordinates that can become misleading after
processor resizing. A valid suggestion becomes the draft class. The detector's
original class and disagreements are retained as metadata.

The prompt requests one JSON object with an explicit `label` and a `reason`,
and shows concrete valid examples. The parser accepts only the seven class
names, `uncertain`, or `not_trash`; it never infers a class from prose. Valid
legacy status/class responses and singleton JSON lists are also supported.
Malformed output is retried once with corrective instructions. If both attempts
fail, the run **stops**, writes `qwen_error.json`, saves previously completed
images and lists unfinished ones in `summary.json`. It cannot silently finish
a Qwen run using only detector labels. Inspect the raw error before retrying;
unchanged deterministic generation may repeat it. Changed code/config needs a
new output directory.

`uncertain` routes a proposal to the uncertain queue; `not_trash` routes it to
background review. Both remain in the complete COCO master under the detector's
provisional class until a person resolves them. There is no eighth training
class and no automatic conversion of uncertainty to `other_trash`. Qwen receives
objects found by the detector; it cannot rescue objects that were never proposed.

Oversized masks are flagged as possible scene/background regions, regardless
of SAM's quality score. Size is only a review heuristic: a large real piece of
litter can also trigger it. A valid litter suggestion for a large region stays
uncertain for inspection. No size rule automatically deletes an object.

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
`flags`, `review_group`, `qwen_status`, and `qwen_reason` to every CVAT label if you want them
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
| `tile_size` | 384 | Pixel width/height of additional detector crops; 0 disables tiled detection. This activates tiles on the 765x575 pilot photos. |
| `tile_overlap` | 0.25 | Overlap between detector crops. |
| `box_merge_iou` / `mask_merge_iou` | 0.85 / 0.75 | Merge similar proposals, retaining competing classes and merged evidence for review. |
| `minimum_crop_size` | 256 | Minimum side length of the context crop passed to SAM. |
| `context_margin` | 0.35 | Extra context around the detector box for SAM. |
| `review.large_mask_fraction` | 0.20 | Flag masks covering at least 20% of the photo for scene/background review. |
| `review.scene_span_fraction` | 0.85 | Also flag masks spanning at least 85% of image width or height, if their area reaches the minimum below. |
| `review.scene_min_area_fraction` | 0.02 | Minimum image-area fraction for the span heuristic; avoids treating every long thin object as background. |
| `review.containment_overlap` | 0.90 | Flag a retained smaller mask mostly contained in another as a possible duplicate or part; keep both. |
| `classification.max_attempts` | 2 | One initial Qwen answer plus one corrective retry. |

These are starting settings, not measured optimal values. The full photograph
is scanned in addition to the tiles. SAM receives a context crop per candidate
so tiny objects are not always reduced with the entire beach image. This is
slower than encoding each full image only once. Cropping cannot recover detail
that was not captured in the photograph.

Each pilot photo now has one full view plus six tiles, with seven detector
prompt groups per view. Expect the detector stage to take longer. For much
larger photos, tune tile size in a separate pilot before processing thousands
of images. The script prints progress for each detector view and candidate.

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

The 20 supplied tests cover mask round-trips, crop coverage, ambiguous labels,
Qwen retries and failure reporting, review queues, existing-mask preservation,
output integrity and resuming. Tests use synthetic images, mocked inference,
and saved predictions from the first five-image pilot. They confirm that four
observed oversized sea/sand masks move to background review, three near-duplicate
masks merge, and the bottle/part overlap is retained for review. Saved predictions
are not ground truth, so these checks do not establish model accuracy.

**The updated pretrained GPU inference and a live CVAT import have not been run
in the authoring environment.** Rerun five photos with the updated script before
expanding the batch. In particular, natural objects can still be misclassified;
review the actual new Qwen responses and all seven categories.

## References

- [Grounding DINO model and usage](https://huggingface.co/IDEA-Research/grounding-dino-base)
- [SAM2 Transformers 4.57.1 interface](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/sam2)
- [Qwen2.5-VL model and usage](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [PyTorch version-specific installation commands](https://pytorch.org/get-started/previous-versions/)
- [CVAT COCO import/export](https://docs.cvat.ai/docs/dataset_management/formats/format-coco/)
- [BePLi v2 category definitions](https://pubmed.ncbi.nlm.nih.gov/40761540/)
