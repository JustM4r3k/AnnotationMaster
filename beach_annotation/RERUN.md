# Rerun the five BeachLitter photos

This update addresses the failures in `beachlitter_first5.zip` and
`beachlitter_qwen5.zip`. It uses the same pretrained models and dependencies.

## Update your existing installation

1. Extract this ZIP to a temporary directory on the VM.
2. Copy the new `annotate.py`, `config.json`, `README.md`, `RERUN.md`, and
   `tests/` into your existing `beach_annotation/` folder, replacing those files.
3. Keep your existing `models/`, `data/`, `.venv/`, and old `output/` runs.
   You do not need to download the model weights again.
4. In a terminal, enter your existing `beach_annotation/` folder and activate
   the environment you used for the successful first run.

For the environment created in the main README:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

All 20 tests should pass. They do not load the models.

## Run the updated pipeline

Use the same photos folder as before. The example below uses the original
suggested path; substitute your actual path if different.

```bash
python annotate.py run \
  --images data/beachlitter/images \
  --output output/beachlitter_fixed_qwen5 \
  --limit 5 \
  --qwen \
  --pack-images
```

Use a **new output folder**. Do not use `--resume` on the old results: the
script and settings changed. The same sorted input folder selects the same
first five photos. Existing seven-class labels and model folders are unchanged.

The smaller crops increase detection work: each of these photos now receives
one full view plus six tiles. Progress is printed for every view and object.

## Check these outputs

- `previews/`: green candidates, amber uncertain/existing objects, and gray
  suspected background in separate panels. Numbers match `image_object_id`
  in `review.csv`. These are provisional groups, not approval decisions.
- `summary.json`: Qwen status counts, group counts, and pending images.
- `review.csv`: class disagreements, Qwen reasons, and mask warnings.
- `annotations_draft.json`: complete review master. It intentionally retains
  suspected background and uncertain objects for correction/deletion in CVAT.

Qwen uses a tight crop and a marked context crop. Its new JSON prompt uses
concrete labels instead of the literal enum string it copied in the old run.
It retries invalid output once, then stops and saves `qwen_error.json` if still
invalid. Send that error file if this happens; do not treat a stopped run as a
successful Qwen run.

To package the new results for checking, run this from `beach_annotation/`:

```bash
python -c "import shutil; shutil.make_archive('output/beachlitter_fixed_qwen5', 'zip', 'output', 'beachlitter_fixed_qwen5')"
```

Upload `output/beachlitter_fixed_qwen5.zip` for review. If the run stopped, this
ZIP also includes the error and partial-run information.

The export and failure-handling tests passed during authoring. Fresh GPU model
inference still needs this rerun; the update does not guarantee correct labels
for every object or recover detail missing from these low-resolution photos.
