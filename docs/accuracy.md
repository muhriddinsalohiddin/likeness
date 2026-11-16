# Accuracy

What the presets can and cannot do, measured with `likeness eval` on
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/): 13 233
pictures of 5 749 people collected from news photographs, 250 pixels square,
with about 45 pixels between the eyes. They are harder than a person looking
into a webcam on purpose, and easier than a camera in a corridor.

```sh
likeness eval --preset default --gallery-size 1000,7000,10000,100000 lfw/
```

Version 0.1.0-dev, November 2026. The search policy turned down 35 pictures
with the default preset and 42 with `buffalo_s`, mostly for the pose of the
head.

## Verification: is this the person they say they are?

Every usable picture compared with every other one: 240 thousand pairs of
the same person, 87 million pairs of different people.

| | `default` (YuNet + SFace) | `buffalo_s` (SCRFD + ArcFace MobileFaceNet) |
|---|---|---|
| licence of the models | MIT, Apache-2.0 | non-commercial research only |
| same person, mean similarity | 0.696 | 0.619 |
| different people, mean similarity | 0.099 | 0.007 |
| equal error rate | 1.75% | 0.88% |
| threshold for 1 false match in 1 000 | 0.373, 1.85% of genuine pairs fail | 0.235, 0.91% fail |
| threshold for 1 false match in 10 000 | **0.445**, 2.16% fail | **0.308**, 1.00% fail |

The thresholds in bold, rounded, are the defaults for verification. A good
part of the genuine pairs that fail are pictures that fail for any model:
profiles, sunglasses, a face half hidden.

Below one false match in ten thousand the dataset stops being a measuring
instrument. It has a few people filed under two names and a few pictures
filed under the wrong one, and a single wrong label on someone with five
hundred pictures makes five hundred pairs of "different people" that look
alike. The thresholds `eval` prints for rarer false matches (0.60 for one in
a hundred thousand, 0.76 for one in a million with the default preset) are
set by those labels, not by the models, and are too strict.

## Identification: who is this?

One picture of each of 4 592 people in the gallery; 5 900 other pictures of
them as probes, and 2 700 pictures of 1 147 people who are not enrolled.

| | `default` at 0.60 | `buffalo_s` at 0.45 |
|---|---|---|
| the enrolled person is the most similar, at any score | 96.4% | 98.1% |
| searches for an enrolled person that do not name them | 17.5% | 7.4% |
| searches for a stranger that name someone | 1.15% | 3.57% |

A search is thousands of comparisons, and each is a chance for a stranger to
resemble somebody. With a false match rate of one in ten thousand per
comparison, a stranger is named in a gallery of 7 000 about every second time.
Raising the threshold until that stops costs the people who are enrolled:
the scores of genuine pairs average 0.6 to 0.7 with these models and leave
no room.

What follows from this:

- **Where people can say who they are, verify.** A login that asks for a
  name or a badge first and then looks at the face makes one comparison. It
  lets through one impostor in ten thousand attempts and turns away one or
  two genuine attempts in a hundred, who try again.
- **Search small galleries, or search and then confirm.** The same models
  that cannot be trusted to pick one of 7 000 are right 96 to 98 times in a
  hundred about who is the most similar. A search that proposes a person,
  followed by something that confirms it, uses what they are good at.
- **For unattended search among thousands, a larger embedder is needed.**
  The models here are the smallest of their kind, chosen to answer in tens
  of milliseconds on a processor. Finding a permissively licensed one that
  is stronger and still fast enough is the next piece of work.

## Measuring on your own pictures

These figures hold for pictures like the ones in the dataset. A camera at
eye level with even light does better; a ceiling camera does worse. Put a
few pictures per person in a directory each and run

```sh
likeness eval --gallery-size 7000 pictures/
```

It prints the tables above for your pictures and the thresholds to start the
server with. A few hundred people with three or four pictures each are
enough to measure one false match in ten thousand.
