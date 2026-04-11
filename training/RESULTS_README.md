# Evaluation Results and Analysis

## Quick Results

### Final Metrics

| Split | Pipeline | Loss | Recall@1 | MRR | nDCG@10 | Score@1 |
|---|---|---:|---:|---:|---:|---:|
| Val | Text | 0.6797 | 0.2500 | 0.3891 | 0.7440 | 0.4392 |
| Val | Image | 0.6924 | 0.2708 | 0.4276 | 0.7557 | 0.4426 |
| Val | Multimodal | 0.6923 | 0.1979 | 0.3570 | 0.7195 | 0.3652 |
| Test | Text | 0.6687 | 0.2143 | 0.3853 | 0.7457 | 0.4189 |
| Test | Image | 0.6926 | 0.2143 | 0.3886 | 0.7315 | 0.3804 |
| Test | Multimodal | 0.6926 | 0.1837 | 0.3193 | 0.7101 | 0.3745 |

### Quick Comparison

- **Text:** the most stable model overall. It does not always win top-1, but it
  is the safest model on test and its mistakes are usually closer to the
  correct meme.
- **Image:** the best validation model. Its main strength is on short, vague,
  or weakly specified posts where visual template cues matter more than the
  tweet text.
- **Multimodal:** the weakest standalone model in this setup. It adds some
  complementary signal, but it does not beat the simpler text-only or image-only
  pipelines in its current training setup.

### In Simple Words, Why

- The **text** model is strong because it does not only read meme titles. It
  also uses captions and metaphor-style descriptions, so it sees a rich
  semantic summary of each meme.
- The **image** model matters because some posts do not say enough in words.
  In those cases, the meme image carries useful information that text alone
  cannot recover.
- The **multimodal** model underperforms because the current way of combining
  image and candidate text does not seem to fuse both signals well. It often
  falls back to generic, high-frequency meme semantics instead.

This does not mean multimodal information is unhelpful. Since text and image
help on different kinds of tasks, a better-trained multimodal model should
still be promising. In particular, LoRA or partial unfreezing, lighter
candidate-text prompts, or a stronger learned fusion module are likely better
fits than the current mostly frozen prompt-level setup.

### Conclusion

The clearest overall result is not that one model dominates everywhere. Instead,
the text and image models help on different kinds of tasks: text is stronger
when candidate metadata is rich, while image is stronger when the post text is
too weak to identify the intended meme on its own. This means multimodal
learning should be useful in principle. However, in the current setup, the
multimodal model does not combine these strengths effectively enough to surpass
the unimodal baselines.

## Scope

This document summarizes the final validation and test results for the three
training pipelines in this repository and explains what the results likely mean
in terms of model behavior.

It also incorporates a deeper analysis from the downloaded
`training/analysis/final_analysis` artifact, which records task-level agreement
between the text, image, and multimodal models. That analysis is used below to
explain not just which model scored better, but where each model helps, where
it fails, and how wrong those failures are.

The evaluated checkpoints were:

- `training/modal_checkpoints/text_hf_best.pt`
- `training/checkpoints/manual_eval/image_qwen_best.pt` on Modal
- `training/checkpoints/multimodal_qwen_clean/best.pt` on Modal

The ranking metrics are computed in `training/metrics.py`:

- `recall_at_1`
- `mrr`
- `ndcg_10`
- `score_at_1`

Lower `loss` is better. Higher values are better for the ranking metrics.

## Final Results

| Split | Pipeline | Loss | Recall@1 | MRR | nDCG@10 | Score@1 |
|---|---|---:|---:|---:|---:|---:|
| Val | Text | 0.6797 | 0.2500 | 0.3891 | 0.7440 | 0.4392 |
| Val | Image | 0.6924 | 0.2708 | 0.4276 | 0.7557 | 0.4426 |
| Val | Multimodal | 0.6923 | 0.1979 | 0.3570 | 0.7195 | 0.3652 |
| Test | Text | 0.6687 | 0.2143 | 0.3853 | 0.7457 | 0.4189 |
| Test | Image | 0.6926 | 0.2143 | 0.3886 | 0.7315 | 0.3804 |
| Test | Multimodal | 0.6926 | 0.1837 | 0.3193 | 0.7101 | 0.3745 |

## High-Level Takeaways

1. The image pipeline is the strongest validation model, but not the safest
   overall model.
2. The text pipeline is the most stable model from validation to test, and its
   errors are usually less severe.
3. The text and image pipelines help on different slices of the data:
   image helps more when the tweet is short or underspecified, while text helps
   more when the candidate metadata is rich.
4. This is the clearest argument for using image features in this task: visual
   information is most useful exactly when the post text is too sparse, vague,
   or noisy to identify the intended meme on its own.
5. The multimodal pipeline is the weakest standalone model, but it still adds a
   small amount of complementary signal on some tasks.
6. The expected "multimodal should be best" outcome did not happen in this
   setup because the current fusion strategy appears to default too often to
   generic, high-frequency meme semantics.
7. Even so, the text/image split suggests multimodal learning is still a strong
   direction if trained with better adaptation and fusion.

## What The Models Are Actually Learning

### Text pipeline

The text pipeline is stronger than it may look from the name alone.

Its candidate text is not just the meme title. It concatenates:

- `meme_title`
- `img_captions`
- `meme_captions`
- `metaphors`

This means the text model sees a dense semantic summary of each meme, including
image description and inferred meaning, without having to interpret the raw
image itself.

On this dataset, the candidate text is information-rich:

- train: average candidate text length `300.1` characters
- val: average candidate text length `296.1` characters
- test: average candidate text length `302.9` characters

Field coverage is also very high:

- `meme_title`: `100%`
- `img_captions`: `100%`
- `meme_captions`: `100%`
- `metaphors`: about `92%` to `94%`

So the text model is operating over high-quality, already-structured semantic
features, not a weak textual baseline.

### Image pipeline

The image pipeline uses `Qwen/Qwen2.5-VL-3B-Instruct` as a cross-encoder over:

- the tweet text
- the candidate image

This model is competitive, and it wins clearly on validation. That means the
raw visual signal is useful for the task.

However, the image model is not obviously stronger than the text model on test.
It ties the text model on `Recall@1`, is only slightly higher on `MRR`, and is
worse on `nDCG@10` and `Score@1`.

### Multimodal pipeline

The multimodal pipeline also uses Qwen, but it adds the candidate text into the
prompt alongside the image and tweet.

In theory this should be the strongest pipeline, because it has access to the
most information. In practice it is the weakest result here. That strongly
suggests the extra modality is not being used effectively.

At the same time, these results should not be read as evidence that multimodal
modeling is a bad idea. Since text and image help on different slices of the
data, a better-trained multimodal system remains a plausible next step. The
current result is better interpreted as a limitation of the training setup than
as evidence against multimodality itself.

## Why Text Is So Competitive

The main reason is that the "text" pipeline is really a semantic metadata
pipeline.

The repository builds candidate text from multiple fields that summarize the
meme's content and intended meaning. Example candidate text often looks like:

- meme title
- image description
- explanation of meme meaning
- metaphor or situation mapping

That gives the text model a compressed description of what the image is doing.
For a task like meme reply ranking, that can be almost as useful as looking at
the image directly.

The text model also has an optimization advantage:

- text runs with `batch_size=16`
- Qwen runs with `batch_size=1`
- the text encoder is trainable
- the default Qwen training flow freezes the encoder

So the text model is both cheaper to optimize and more adaptable to the task.

## Why Multimodal Underperformed

Several factors point in the same direction.

### 1. The added text may be noisy rather than helpful

For Qwen multimodal ranking, the prompt becomes:

- tweet text
- candidate meme text
- candidate image

But the candidate meme text is already long and highly interpreted. It may
contain explanations that are imperfect, redundant, or overly specific. Instead
of helping the VLM, it may distract it from the image.

### 2. The Qwen encoder is mostly frozen

The default training flow passes `--freeze_encoder` to the Qwen pipelines.

That means the large VLM is not fully adapting to the task. The trainable part
is mainly the lightweight projection and scoring head on top of the frozen
Qwen representation. That is a much smaller adaptation budget than the text
pipeline gets.

This matters more for multimodal than image-only:

- image-only can rely on Qwen's pretrained visual-text alignment
- multimodal needs the model to learn how to use your specific candidate text
  representation well

Freezing makes that harder.

This is also why the current multimodal result should not be treated as the
final word on multimodal learning. If the model were trained with LoRA,
partial unfreezing of later Qwen layers, or a more task-specific fusion head,
it would have a much better chance of learning when to trust image cues versus
candidate text.

### 3. The Qwen losses remain near random-pair BPR scale

The training loss is BPR. A random pairwise model tends to sit near `0.693`.

Both Qwen pipelines finish evaluation around `0.692x`, while the text pipeline
is lower. This does not mean the Qwen models are useless, but it does suggest
their pairwise ranking separation remains weak relative to the text model.

### 4. The fusion strategy is prompt-level, not task-specific learned fusion

The multimodal setup does not build a dedicated learned fusion block over image
and candidate text for Qwen. Instead, it relies on a frozen VLM prompt
construction plus a shallow scorer head. That is a practical baseline, but not
necessarily the best way to extract complementary signal from both modalities.

## Validation vs Test Behavior

### Text

Text is the most stable pipeline:

- `MRR`: `0.3891 -> 0.3853`
- `nDCG@10`: `0.7440 -> 0.7457`

It is also the safest model once we look beyond exact top-1 hits. On test,
text and image tie on `Recall@1`, but text's mistakes are usually closer to the
correct answer:

- test mean gold rank of the chosen top prediction: `4.22` for text vs `4.95`
  for image
- when wrong on test, the gold meme still appears at rank `2` or `3` in
  `36.4%` of text errors vs only `19.5%` of image errors

So text is not just stable in aggregate metrics. It is also less likely to fail
catastrophically.

### Image

Image wins on validation but drops more on test:

- `Recall@1`: `0.2708 -> 0.2143`
- `MRR`: `0.4276 -> 0.3886`
- `Score@1`: `0.4426 -> 0.3804`

The deeper analysis shows that image is still important, but it is more
specialized than text. It tends to help most on short or weakly specified
tweets, where the post text alone does not strongly identify the intended meme.
This is the strongest evidence for why the image pipeline matters at all:
it contributes visual template information precisely in the cases where the
tweet text is not informative enough to recover the meme from metadata alone.

### Multimodal

Multimodal is weak on both splits, so this is not just a validation/test
generalization issue. The main problem looks structural rather than incidental.

At the same time, multimodal is not completely redundant:

- it is better than both text and image on `10 / 96` validation tasks
- it is better than both text and image on `19 / 98` test tasks

However, only `4` of those cases on each split become actual top-1 wins. Most
of the time, multimodal improves rank slightly without producing enough
separation to beat the unimodal baselines at the top.

## Dataset Effects That Matter

The split sizes are modest:

- validation tasks: `96`
- test tasks: `98`

Each task has a fixed candidate set of roughly `8` to `9` memes on average.

That means:

- small metric differences should not be overclaimed
- the tie between text and image on test `Recall@1` is real enough to matter
- the consistent gap between multimodal and the other two models is more
  meaningful than tiny text-vs-image differences

## Likely Domain Mismatch

There is also a plausible cross-platform mismatch in the data construction.

- the query side comes from `HSDSLab/TwitterMemes`
- the candidate meme bank and metadata come from `MemeCap`
- in the clean split, `7143 / 7745` candidate image URLs resolve to `i.redd.it`
- the tweet side is strongly Twitter-shaped, with hashtags in essentially all
  clean rows

This matters because meme usage norms are platform-dependent.

A meme that is common on Reddit can still be visually recognizable on Twitter,
but its typical tone, caption style, and implied audience may differ. That
could affect:

- which memes are considered natural replies
- how useful MemeCap captions are for reply ranking
- whether multimodal prompts overfit to platform-specific textual phrasing

This does not by itself explain every result, but it is a meaningful hypothesis
for why the text and multimodal pipelines may behave differently from what we
would expect in a purely in-domain setup.

## Deep Analysis from `final_analysis`

The downloaded `final_analysis` artifact gives a more useful view of model
behavior than the aggregate metric table alone. In particular, it shows where
the models agree, where they disagree, and whether an incorrect prediction was
still close to the correct meme.

### 1. Most tasks are hard for all three models

The models are simultaneously wrong on many tasks:

- validation: all three are wrong on `56 / 96` tasks
- test: all three are wrong on `67 / 98` tasks

This matters because it means the task is not just a matter of choosing the
right encoder. A large portion of the benchmark appears genuinely ambiguous or
underspecified under the current training formulation.

### 2. Agreement often means agreement on the wrong meme

The models predict the exact same top meme on:

- `26` validation tasks
- `25` test tasks

But many of those are shared failures:

- `15` validation tasks where all three chose the same wrong meme
- `15` test tasks where all three chose the same wrong meme

This is important because it points to common dataset-wide shortcuts rather than
independent model errors.

### 3. The text model is more forgiving when it misses

Top-1 accuracy alone hides an important difference between text and image.

On test, both text and image get `21 / 98` tasks exactly correct, but the text
model is much more likely to place the correct meme near the top even when it
misses:

- text test errors that are near-misses (`rank 2` or `3`): `36.4%`
- image test errors that are near-misses (`rank 2` or `3`): `19.5%`
- multimodal test errors that are near-misses (`rank 2` or `3`): `31.3%`

The average gold rank of the chosen top meme when the model is wrong also shows
the same pattern:

- validation: text `5.39`, image `5.71`, multimodal `6.08`
- test: text `5.10`, image `6.03`, multimodal `5.53`

So even though image ties text on test `Recall@1`, text is still the more
reliable ranking model overall.

### 4. Image and text are solving different kinds of tasks

The slice analysis is where the strongest interpretation comes from.

#### Short or weakly specified tweets favor image

On short tweets, image consistently beats text:

- validation short tweets: image `25.0%` vs text `11.5%`
- test short tweets: image `18.9%` vs text `13.5%`

On tasks where candidate text is relatively weak, image also helps more:

- validation, average candidate text length `<= 260`: image `21.1%` vs text
  `5.3%`
- test, average candidate text length `<= 260`: image `12.5%` vs text `0.0%`

This suggests image is carrying useful template or scene information that is
not recoverable from the tweet alone. In other words, the image model is most
valuable when the text channel is weak. That is the clearest practical case for
including visual features in this project.

#### Rich candidate metadata favors text

On tasks where the candidate metadata is richer, text becomes stronger:

- validation, metaphor coverage `all`: text `35.7%` vs image `23.8%`
- test, average candidate text length `301+`: text `32.7%` vs image `22.4%`

This is consistent with the text model benefiting directly from the candidate
metadata bundle, which already contains a strong semantic summary of the meme.

### 5. The multimodal model appears to overuse generic fallback memes

The most repeated wrong predictions are highly generic meme titles. Across both
splits, the multimodal model most often fails by choosing:

- `This relates to current events!` (`37` wrong predictions)
- `you can still use the service without little blue stamp` (`13`)
- `Him buying and destroying Twitter and making a fool out of himself while doing so mightve done us all a favor.` (`8`)
- `Today on twitter` (`7`)
- `The state of Twitter` (`6`)

This is more extreme than the other models. Using a simple heuristic for
"Twitter-like fallback" titles, roughly:

- `15%` to `26%` of text/image errors look like these generic internet-commentary
  memes
- `35%` of multimodal validation errors do
- `45%` of multimodal test errors do

That is a strong sign that multimodal is not performing robust fusion. It is
often defaulting to a small set of broad, socially topical memes instead.

### 6. Some benchmark slices are much easier than others

`MeMe Live` or `meme.chat` tasks are much easier than the rest of the dataset:

- validation `MeMe Live`: text `64.3%`, image `50.0%`, multimodal `64.3%`
- validation non-`MeMe Live`: text `18.3%`, image `23.2%`, multimodal `12.2%`
- test `MeMe Live`: text `52.4%`, image `38.1%`, multimodal `52.4%`
- test non-`MeMe Live`: text `13.0%`, image `16.9%`, multimodal `9.1%`

This suggests that the benchmark contains some repetitive or easier subdomains,
while the remaining tasks are substantially harder.

### 7. Multimodal adds some complementary signal, but not enough

There is real complementarity between the models, especially between text and
image.

If an oracle could choose the better of the text and image predictions per task:

- validation: `36 / 96 = 37.5%`
- test: `27 / 98 = 27.6%`

If the oracle could choose the best of all three:

- validation: `40 / 96 = 41.7%`
- test: `31 / 98 = 31.6%`

So multimodal is not useless, but the gain over the text+image pair is only
`4` tasks on each split. That is too small to justify calling the current
multimodal design a successful integration of both modalities.

Still, that small gain matters. It shows that the multimodal model is capturing
some extra signal that neither unimodal model uses consistently. That makes a
better-trained multimodal system a realistic next step rather than a dead end.

## What We Can Say About "How Wrong" The Model Is

For this project, simple top-1 accuracy is not enough. The task is ranking, not
independent classification.

That is why the current evaluation already includes several metrics with
different meanings:

- `Recall@1`: did the model place a rank-1 meme at the top
- `MRR`: how early the first truly good meme appears
- `nDCG@10`: how good the whole ranked list is, not just the first item
- `Score@1`: how semantically good the top prediction is under the ground-truth
  rank labels

These metrics also help quantify how wrong a prediction is, rather than only
whether the exact top candidate was correct:

- `Recall@1` only captures exact top-hit behavior
- `MRR` captures whether the model was close even if not exactly right
- `nDCG@10` captures graded ranking quality across the whole candidate list
- `Score@1` captures whether the top prediction was near-correct or very wrong

So the current metric suite is already better aligned with ranking than a
multi-class classifier that treats classes independently.

## Qualitative and Error Analysis

The task-level artifact makes it possible to say more than just "model A beat
model B." It already reveals a few recurring qualitative patterns.

### Text-only wins

When text is uniquely correct, the winning cases tend to be semantically
specific caption-like tasks where the metadata clearly matches the intended
situation. Two representative test examples are:

- `2018_12-10677`: text gets `Ah yes just an innocent crocodile`, while image
  and multimodal miss badly
- `2018_12-10244`: text gets `When your day is done, and you want Nirvana`,
  while image and multimodal both fail on the same alternative

### Image-only wins

When image is uniquely correct, the tweet text is often vague, short, or weakly
descriptive. The image model appears to recover the meme template directly from
visual cues. Representative test examples include:

- `2018_12-1072`: image gets `My entire feed these days`, while text and
  multimodal both fall back to `This relates to current events!`
- `2018_12-10146`: image gets `Menaces of the Ocean.`, while the other models
  miss
- `2018_12-10610`: image gets `Welp here is my joke meme for your
  entertainment`, while text and multimodal choose weaker semantic matches

### Multimodal-only wins

When multimodal is uniquely correct, the task often has a social-media or
internet-commentary flavor. However, this is also the same region where
multimodal overpredicts generic fallback memes. So the extra signal is real,
but inconsistently used.

Representative test examples include:

- `2018_12-10020`: multimodal gets `A useful tool`
- `2018_12-10090`: multimodal gets `Today on twitter`
- `2018_12-1021`: multimodal gets `Online really messed our minds`
- `2018_12-10698`: multimodal gets `The state of Twitter`

### Shared confident failures

Some of the clearest shared failures occur when all three models collapse to
the same generic meme:

- `This relates to current events!`
- `you can still use the service without little blue stamp`

On test, `12` of the `15` tasks where all three models made the same wrong
prediction collapsed to `This relates to current events!` alone. This is one of
the strongest signals in the analysis, because it shows a recurring shortcut
that is shared across pipelines rather than one model-specific bug.

## Confidence Analysis

The analysis artifact uses the top-1 vs top-2 score gap as a practical
confidence signal.

The confidence patterns differ sharply across models.

### Text

Text produces much larger margins than the Qwen-based models:

- validation mean margin: `0.1735`
- test mean margin: `0.2068`

That larger separation is present for both correct and incorrect predictions.
For example, on test:

- text correct mean margin: `0.3451`
- text bad-miss mean margin: `0.1444`

So text is the most decisive model, but it is not always well-calibrated. Some
of its most confident failures are exactly the generic fallback cases mentioned
above.

### Image

Image margins are extremely small across the board:

- validation mean margin: `0.0053`
- test mean margin: `0.0053`

This suggests the image model often has only weak separation between the top few
candidates. That is consistent with the earlier observation that image is
useful, but more fragile and less robust as a standalone scorer.

### Multimodal

Multimodal margins are also small:

- validation mean margin: `0.0120`
- test mean margin: `0.0119`

They are slightly larger than image margins, but still far below text. This is
consistent with multimodal improving some ranks without producing enough
confidence to become the best top-1 model.

## Best Way To Present The Results

The most defensible summary is:

1. The image pipeline gives the best validation ranking performance.
2. The text pipeline is the most stable and competitive model on test, and its
   mistakes are usually less severe.
3. The image pipeline is not weaker than text in general; it is strongest on
   short or underspecified posts where visual template recognition matters.
4. This short-text setting is the clearest justification for the image model:
   it helps precisely when text-only reasoning is least reliable.
5. The multimodal pipeline adds some complementary signal, but it does not
   improve over text-only or image-only in the current formulation.

That is a better summary than saying:

- "image is definitely best" or
- "multimodal should have been best but mysteriously failed"

The stronger interpretation is that the current multimodal design is not making
good use of the extra candidate-text signal, while the text baseline is much
stronger than a simple title-only baseline would be. The main comparison is not
between "text vs image" in the abstract, but between two different ways of
recovering meme meaning:

- text from rich metadata
- image from raw visual template cues

Since those two sources of signal are clearly complementary, the right
interpretation is not "multimodality does not help." It is "the current
multimodal setup is not trained effectively enough yet."

## Alternative Training Pipelines Worth Trying

The present experiments mostly compare encoders. A stronger follow-up would also
compare training formulations.

### 1. Text ablation pipeline

Train separate text models on:

- title only
- title + `img_captions`
- title + `img_captions` + `meme_captions`
- full text including `metaphors`

This would isolate how much of the text model's strength comes from rich
metadata rather than general language understanding.

### 2. Lightweight multimodal fusion pipeline

Instead of putting long candidate text directly into the Qwen prompt, build:

- a text encoder for the tweet and candidate text
- an image encoder for the meme image
- a learned fusion head over both representations

This would test whether the current prompt-level fusion is the problem.

### 3. Retrieval-plus-reranking pipeline

Right now each task reranks a fixed candidate set. A more realistic pipeline is:

1. retrieve top `N` candidates
2. rerank them

That would help answer the professor's question about semantic similarity
between "classes" and how wrong a model is when it misses the exact top meme but
retrieves something nearby.

### 4. LoRA or partial-unfreeze Qwen pipeline

The current Qwen setup is mostly frozen. A stronger multimodal experiment would
add:

- LoRA adapters
- partial unfreezing of later layers
- a comparison against the fully frozen baseline

This is likely the most direct way to test whether the current multimodal result
is limited by insufficient adaptation capacity.

### 5. Confidence-aware reranking pipeline

A useful variant is to train or evaluate with explicit uncertainty signals:

- top-score margin
- agreement between text and image models
- fallback from multimodal to text when margins are small

This would make the system easier to analyze and more defensible in a report.

## Recommended Next Experiments

If the goal is to make image or multimodal clearly beat text, the next
experiments should target the actual bottlenecks:

1. Ablate candidate text fields in the text baseline.
   Compare full text against:
   - title only
   - title + captions
   - title + captions + no metaphors

2. Simplify multimodal candidate text.
   Instead of feeding all joined fields to Qwen, try:
   - title only
   - title + short caption
   - no candidate text at all, image-only prompt

3. Unfreeze more of Qwen or use LoRA/adapters.
   Multimodal likely needs more than a shallow scorer head to learn how to use
   your candidate text representation.

4. Inspect prompt quality.
   The current prompt asks how suitable the meme image is as a reply. That may
   not be the best phrasing once long candidate text is also injected.

5. Add qualitative and confidence-based error analysis.
   The final report should not stop at metrics. It should show where the model
   is confidently correct, confidently wrong, and only weakly separated.

6. Report both validation and test, not just one split.
   Validation alone would make image look decisively best. Test shows the
   picture is more mixed.

## If We Had Three More Months

The strongest next phase would be:

1. build an error-analysis set with manually reviewed successes, near-misses,
   and failures
2. run text ablations to measure the contribution of each metadata field
3. train a better multimodal model with LoRA or partial unfreezing
4. test a fusion architecture that combines image and text embeddings outside a
   frozen prompt-only VLM
5. evaluate retrieval plus reranking instead of only reranking fixed candidates
6. analyze domain shift between the Twitter query distribution and the
   Reddit-heavy meme bank more explicitly

That would move the project from "which checkpoint scored higher" to "what the
model is actually using, where it fails, and how to improve it in a principled
way."

## Bottom Line

The text baseline is strong because it already contains structured semantic
descriptions of the memes. The image pipeline is useful and competitive, and
its main contribution is on short or weakly specified posts where visual cues
carry information that the tweet text does not. The multimodal pipeline is
currently underperforming because the extra textual signal is likely noisy and
the frozen Qwen setup does not have enough freedom to learn a better fusion.
Given the complementary strengths of text and image, a stronger multimodal
training strategy remains one of the most promising directions for improvement.
