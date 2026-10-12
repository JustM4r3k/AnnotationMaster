# Run the Qwen visual check

This add-on tests Qwen alone using five selected regions from your uploaded
photos: sea, sand, driftwood, a dark slipper, and a small green fragment.
The required photos and selected coordinates are included.

## Put the files in place

Copy the contents of this ZIP's `beach_annotation/` folder into your existing
`beach_annotation/` folder on the VM. This adds:

- `diagnose_qwen.py`
- `diagnostic_cases/` (photos, selected coordinates and comparison prompts)
- `diagnostic_tests/`
- `QWEN_DIAGNOSTIC.md`

Use your existing Python environment and downloaded Qwen model. The diagnostic
loads only Qwen and writes a separate output directory. There are no new Python
dependencies, paid API calls or model downloads.

## Run this command

From your existing `beach_annotation/` directory, with its environment active:

```bash
python diagnose_qwen.py
```

If you followed the original environment instructions, activate it first with:

```bash
source .venv/bin/activate
```

The default model path is `models/Qwen2.5-VL-7B-Instruct/`, resolved relative to
the diagnostic script. If your model is somewhere else:

```bash
python diagnose_qwen.py --model-dir /your/path/to/Qwen2.5-VL-7B-Instruct
```

The script makes **20 generations: five regions x four checks**. It prints each
answer as it runs and saves progress after every answer. The first model load
can take time. Run it after the full annotation process has exited so Qwen can
use the GPU memory. The default device is `cuda`; `--device cuda:1` selects a
different GPU if needed.

## Send back this file

The script automatically creates:

```text
output/qwen_diagnostic.zip
```

Upload that ZIP. It includes the raw answers, prompts, input crops, processed
pixel previews, processor checks, runtime/model information and an HTML report.
To read it yourself, open `output/qwen_diagnostic/report.html` in a browser.

The script also packages partial results when inference fails. In that case,
send the ZIP containing the error. If it fails before making an output folder,
send the terminal error text.

To run again, choose a new output directory:

```bash
python diagnose_qwen.py --output output/qwen_diagnostic_second
```

## What the four checks isolate

| Check | Images Qwen receives | Prompt |
| --- | --- | --- |
| A | Exact rectangular crop | Neutral description in ordinary sentences |
| B | Pipeline's square crop | Exactly the same neutral description prompt |
| C | Square crop plus marked context | Current classification prompt with example answers removed |
| D | The same two images as C | Current classification prompt including its example answers |

A versus B tests the effect of crop context. C versus D tests the effect of
prompt examples. Each call starts a new conversation: previous answers cannot
influence later answers. Reviewer notes and filenames are never sent to Qwen.
There are no retries, so malformed or unexpected answers stay visible.

The sea case demonstrates why this matters: its exact rectangle is 765x105
pixels, but the pipeline's square crop becomes 765x574 pixels, almost the entire
765x575 photograph.
A scene description of the square crop can legitimately mention the beach;
check the actual input before treating that as a recognition error.

These are manually selected diagnostic regions. They reproduce the crop policy
and original prompt, but they are **not exact replays of the earlier detector
inputs**: the annotation outputs did not save the original detector boxes.

## Read the results carefully

- If A accurately describes sea/wood/footwear but B drifts to another object,
  the larger square crop is a likely contributor.
- If C behaves reasonably but D repeats the example answer on the same images,
  that supports the prompt-bias explanation.
- If even A confidently describes water as a bottle, inspect the processed
  images and processor checks before trusting any classifier changes.
- If descriptions are sensible but class choices are wrong, focus on the
  classification instructions and class definitions.

The script verifies that image pixels are present, finite, and represented by
the expected number of image tokens. It reconstructs previews from the actual
processed pixel patches, undoing normalization and patch packing. These show
what resizing/normalization produced; they do not visualize internal model
features or prove the model uses the image correctly. A reconstruction failure
is recorded without hiding the model's raw answer.

Material is not always visible in these photos. In particular, the slipper's
plastic/rubber composition and the green fragment's original object are not
established. The notes are for reviewing the test, not training ground truth.
Valid JSON and the final `COMPLETE_REQUIRES_REVIEW` status mean the test ran;
they do not mean its predictions are correct. This small selected set is a
diagnostic, not an accuracy benchmark.

## Optional checks without GPU inference

```bash
python -m unittest discover -s diagnostic_tests -v
python diagnose_qwen.py --dry-run --output output/qwen_diagnostic_inputs
```

The seven offline tests check crop handling, processed-image reconstruction,
controlled prompt comparisons, report escaping and saving partial results.
They passed in the authoring environment. Model inference could not run there;
the command above performs that check on your VM.

The dependency remains `transformers==4.57.1`, matching your successful runs.
The script defaults to the same automatic bfloat16/float16 choice on CUDA and
uses local files only. An optional `--dtype float32` is available for a later
numerical comparison, but uses substantially more GPU memory; start with the
default run.
