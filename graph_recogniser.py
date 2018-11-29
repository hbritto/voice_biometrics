import os
import pickle
from deep_speaker.audio_reader import AudioReader
from deep_speaker.constants import c
from deep_speaker.unseen_speakers import inference_embeddings
from infrastructure.inference import Inference

import matplotlib.pyplot as plt

if __name__ == "__main__":
    inference = Inference()
    base_dir = '/run/media/hbritto/Data/Datasets/deep-speaker-data'
    cache_dir = os.path.join(base_dir, 'cache', '')
    audio_dir = os.path.join(base_dir, 'VCTK-Corpus', '')
    reader = AudioReader(input_audio_dir=audio_dir, output_cache_dir=cache_dir,
                         sample_rate=c.AUDIO.SAMPLE_RATE, multi_threading=True)
    with open(os.path.join(cache_dir, 'embeddings.pkl'), 'rb') as pkl:
        all_embs = pickle.load(pkl)

    me = 'PhilippeRemy'
    me_emb = inference_embeddings(reader, me)
    all_embs.update({me: me_emb})
    inference.update_recogniser(all_embs)
    with open('p.txt', 'w') as per:
        print(inference.recogniser.persons, file=per, flush=True)
    iterations = range(1, 30)
    results = []
    for i in iterations:
        me_emb = inference_embeddings(reader, me)
        pers, dist = inference.identify_person(me_emb, True)
        print(pers)
        results.append((pers, dist))

    zipped = list(zip(*results))
    plt.figure()
    plt.scatter(zipped[0], zipped[1])
    plt.grid(True)
    plt.show()
