# Our Approach

## The problem

Solar cells crack. Some cracks are hairline, some whole corners go dark, and
most of it is invisible in ordinary light. Shine current through the cell
instead and it glows — that is an electroluminescence (EL) photo. Faulty parts
glow less, so they show up as dark marks.

Checking those photos by hand is slow and inconsistent. We want a model that
takes an EL photo and marks which pixels are faulty.

Not just "is this cell bad, yes or no" — we want the location, because where
the fault is tells you what kind of fault it is.

## Why we label pixels instead of spotting odd ones

There are two ways to build this.

**Spot the odd one out.** Show the model thousands of good cells until it
knows what normal looks like, then flag anything that does not fit. No labels
needed.

**Label the pixels.** Take photos where a person has already painted over the
faults, and teach the model to copy that.

We went with the second, for three reasons.

**You need labels either way.** Even the first approach has to be tested. You
cannot tell whether "spot the odd one out" is working without photos where
somebody has marked the real faults. So the labels are needed regardless.

**We already have them.** PVEL-S comes with 1,200 photos and a hand-painted
mask for every one. Going label-free would mean throwing away work that was
already done for us.

**EL photos break the odd-one-out idea.** Solar wafers are grown from many
crystal grains, and the boundaries between them look like dark jagged lines —
much like cracks. A model with no labels has no way to tell "this dark line is
just how the wafer grew" from "this dark line is a fracture". And this is not a
guess: in a 2026 study, even a fully supervised model confused two similar
fault types, scoring 0.23 on one and 0.81 on the other despite them being the
same size. If labels barely separate them, no labels has no chance.

## How it works

Five steps.

**1. Pair up the files.** Each photo has a mask file next to it with `_mask`
added to the name. We match them on the filename.

**2. Split the images into three groups.** Training (the model learns from
these), validation (used to decide when to stop), and test (held back until
the very end). The dataset arrives pre-split 840 / 360, which is the same split
the paper we compare against used.

**3. Cut each photo into tiles.** The photos are 1024x1024, too big to feed in
whole. Each becomes nine overlapping 512x512 squares.

**4. Train.** Show the model tiles, measure how wrong it is, nudge it, repeat.
Each tile is randomly flipped or turned first, so the model sees variety rather
than memorising 5,818 specific pictures.

**5. Score it.** Predict every tile of a test photo, stitch the predictions
back into one full-size map, and compare it with the mask a human painted.

## The three decisions that matter most

### Keep full resolution

Most people shrink every image to 224x224. We do not.

A hairline crack is one to three pixels wide. Shrinking a 1024px photo to 224
makes that crack thinner than a single pixel — it is gone before the model ever
sees it. Cutting the photo into tiles keeps every pixel at its original size.

The tiles overlap by half. Without the overlap, a crack sitting on a tile edge
gets sliced in two, and each half is harder to recognise than the whole.

### Split by photo, and split first

This is the easiest way to get a great-looking result that means nothing.

Two traps. Tiles from the same photo overlap, so if you split the *tiles*, the
model gets tested on pixels it trained on. And training photos get flipped and
turned, so if you split *after* that, a mirrored copy of a training photo can
land in the test set.

Both are avoided by splitting the list of photos first, before any cutting or
flipping. Every tile from one photo stays on the same side. The code checks
this and crashes if it ever goes wrong.

### Make the rare pixels count

In a typical photo only about 1% of pixels are faulty. A lazy model could
answer "fine" for every pixel and be 99% correct — and completely useless.

Three things push against that:

- Mistakes on faulty pixels count about seven times more than mistakes on
  normal ones.
- Half of the loss comes from a measure that only looks at how well the faulty
  areas overlap, and ignores background entirely.
- Half of every batch is drawn from tiles that actually contain a fault,
  instead of picking tiles evenly.

And we never report overall accuracy, because 97% of pixels are background and
that number would always look good.

## What is hard

**Small faults.** The model finds big dark patches easily and misses hairline
cracks almost entirely. Our first throwaway run already showed it: everything
above 10,000 pixels found, everything under 2,000 missed. A published study
found the same cliff. This is the main thing left to solve.

**Two very different jobs in one.** Some faults cover half the wafer; others
are three pixels wide. Right now both are the same single "fault" label, and
the big ones dominate what the model learns. Splitting them apart may matter
more than any change to the network.

**No clean cells.** Every one of the 1,200 photos has a fault in it. So we can
measure *where* faults are, but we cannot yet measure how often the model
cries wolf on a perfectly good cell. That needs clean photos from somewhere
else.

## What we are deliberately not doing

Several recent papers propose custom network designs for this task. We use a
plain, well-tested one instead — a U-Net with a ResNet-34 encoder that already
knows about edges and textures from millions of everyday photos.

The reported gains from those custom designs are small, often around 1%, and
usually measured from a single run with no sense of how much the numbers wobble
between runs. One of those papers' own tables shows a standard off-the-shelf
model landing within 3 points of their custom one. Image resolution and the
handling of rare pixels move the result far more than the network shape does.

## Where this goes next

In rough order:

1. Run the full thing on a GPU and get a real baseline. Target: 95.5% IoU,
   which is what the reference paper reports on this same split.
2. Run all five folds, so we know how much the number wobbles. A 1% gain
   inside a 2% wobble is nothing.
3. Attack the small-fault problem — likely by separating big and small faults
   rather than by changing the network.
4. Try the label-free approach on the same test set, so we have a measured
   answer about it rather than a prediction.
5. Multi-class: PVEL-S has twelve fault types, and telling a crack from a dead
   cell is more useful to a plant operator than a plain yes or no.
