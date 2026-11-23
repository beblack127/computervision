# Analysis report (auto-generated)

dump: `/content/cv_drive/runs/xpool_9k_s24/dump_model_best`

## Overall

| subset | n | R@1 | R@5 | R@10 | MdR | MnR |
|---|---|---|---|---|---|---|
| all | 1000 | 46.7 | 71.9 | 82.0 | 2.0 | 13.8 |

## By query length (words)

| subset | n | R@1 | R@5 | R@10 | MdR | MnR |
|---|---|---|---|---|---|---|
| <=6 | 273 | 33.7 | 59.3 | 74.4 | 3.0 | 26.9 |
| 7-9 | 317 | 41.0 | 68.5 | 77.3 | 2.0 | 12.0 |
| 10-12 | 220 | 54.1 | 82.7 | 90.5 | 1.0 | 8.0 |
| >=13 | 190 | 66.3 | 83.2 | 91.1 | 1.0 | 4.7 |

## Temporal words in the query

| subset | n | R@1 | R@5 | R@10 | MdR | MnR |
|---|---|---|---|---|---|---|
| no temporal word | 937 | 45.7 | 71.0 | 81.3 | 2.0 | 13.9 |
| temporal word | 63 | 61.9 | 85.7 | 92.1 | 1.0 | 11.5 |

## By category

| subset | n | R@1 | R@5 | R@10 | MdR | MnR |
|---|---|---|---|---|---|---|
| 0 | 74 | 36.5 | 66.2 | 79.7 | 2.0 | 16.1 |
| 1 | 18 | 27.8 | 38.9 | 55.6 | 8.0 | 49.9 |
| 2 | 53 | 26.4 | 52.8 | 66.0 | 5.0 | 29.6 |
| 3 | 95 | 40.0 | 74.7 | 90.5 | 2.0 | 7.1 |
| 4 | 58 | 43.1 | 67.2 | 77.6 | 2.0 | 15.9 |
| 5 | 26 | 53.8 | 88.5 | 96.2 | 1.0 | 3.4 |
| 6 | 30 | 40.0 | 66.7 | 76.7 | 2.0 | 9.0 |
| 7 | 122 | 41.8 | 66.4 | 76.2 | 2.0 | 19.6 |
| 8 | 38 | 55.3 | 76.3 | 78.9 | 1.0 | 11.3 |
| 9 | 76 | 50.0 | 72.4 | 84.2 | 1.5 | 8.1 |
| 10 | 46 | 50.0 | 73.9 | 82.6 | 1.5 | 18.6 |
| 11 | 29 | 62.1 | 96.6 | 100.0 | 1.0 | 1.7 |
| 12 | 43 | 55.8 | 69.8 | 76.7 | 1.0 | 35.3 |
| 13 | 58 | 62.1 | 86.2 | 89.7 | 1.0 | 6.3 |
| 14 | 47 | 44.7 | 72.3 | 89.4 | 2.0 | 4.0 |
| 15 | 13 | 69.2 | 84.6 | 84.6 | 1.0 | 5.2 |
| 16 | 64 | 51.6 | 76.6 | 82.8 | 1.0 | 10.7 |
| 17 | 33 | 48.5 | 69.7 | 87.9 | 2.0 | 5.5 |
| 18 | 53 | 50.9 | 73.6 | 79.2 | 1.0 | 12.7 |
| 19 | 24 | 62.5 | 79.2 | 87.5 | 1.0 | 6.5 |

## Frame-order probes (same checkpoint, frames permuted at test time)

| frames | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|
| original | 46.7 | 71.9 | 82.0 | 13.8 |
| reversed | 46.7 | 71.9 | 82.0 | 13.8 |
| shuffled | 46.7 | 71.9 | 82.0 | 13.8 |
| middle | 29.9 | 56.5 | 66.9 | 42.4 |

## Hubness

- N_10 skewness: 0.90; max times a single video is top-1: 8
- videos never retrieved at top-1: 338 / 1000; never in any top-10: 1
- DIAGNOSTIC (uses all test queries, not a valid method): dual-softmax R@1 = 48.8

## Failures

- failed queries (rank>1): 533
- cos(query, caption of wrongly retrieved video): mean 0.784; share ≥0.9: 15.8%

### Worst-ranked queries

| rank | query | GT | top-1 (its caption) |
|---|---|---|---|
| 882 | a man playing video games | video9808 | video9679 (the demonstration of mobile rack) |
| 789 | person playing a game | video7215 | video8459 (mario and friends play in a video game together) |
| 744 | a family is having coversation | video8912 | video8048 (george lopez s family sits down for dinner) |
| 473 | someone is playing a first person shooter game and is making jokes | video9872 | video7757 (a figure in camouflage clothing walks with bent knees through large cement rooms toward a door as people are being shot) |
| 440 | some peole are sitting in hall | video9878 | video7444 (a group of people watching a screen) |
| 371 | music is playing and advertisements was showing | video8667 | video8485 (foreign language music video) |
| 348 | a person is explaining something | video7166 | video8488 (a woman is talking about facial care products) |
| 340 | a man playing video games | video7461 | video9679 (the demonstration of mobile rack) |
| 312 | cartoons are talking to each otehr | video7619 | video7912 (peter is driving in the car) |
| 258 | some people are inside of a room | video9337 | video7825 (sitting and converstion 2 lady and 2 gents) |
| 182 | a man is singing | video7227 | video9351 (a man is singing on stage to a huge audience he is holding a microphone) |
| 173 | there is a man shooting other people in a corridor | video7110 | video7576 (men are being filmed in the darkness) |
| 164 | a man playing video games | video7710 | video9679 (the demonstration of mobile rack) |
| 158 | a woman is ripping off a man clothes | video7974 | video8825 (a woman holds up a leather jacket) |
| 156 | people are talking to each other | video8800 | video8302 (a man talks about kim chi to a camera) |

### Failures whose wrong top-1 has an almost identical caption (annotation ambiguity)

| rank | query | top-1 caption |
|---|---|---|
| 3 | a woman giving a photoshop tutorial | a woman giving a photoshop tutorial |
| 2 | a group of people are dancing in a room | a group of people are dancing in a room |
| 2 | a football video game is shown | a football video game is being played |
| 4 | a man is playing baseball | people are playing baseball |
| 145 | a women is doing craft and talking about that | a woman makes crafts |
| 10 | there is a wrestling match going on between two people | men are doing wrestling |
| 5 | a girl playing the game the sims | a lady is play the sims while talking about it |
| 47 | someone is playing a game | someone giving demo for some game |
| 26 | someone is playing a game | someone giving demo for some game |
| 3 | a video game character rides around on a motorcycle | a man drives a motorcycle in a video game |
| 4 | a band plays on a stage | a rock band performs on stage |
| 3 | a man is showing off a new vehicle | a man is giving a review on a vehicle |
| 2 | woman playing instruments in a field for a music video | a woman plays instruments in a field |
| 11 | two wrestlers are fighting on a mat | men are doing wrestling |
| 6 | there is no sound while the screen shows a person playing a computer game | there is someone playing a game in a computer |
