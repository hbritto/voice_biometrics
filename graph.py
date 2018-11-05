import matplotlib.pyplot as plt

import os
import pickle

from copy import deepcopy
from deep_speaker.audio_reader import AudioReader
from deep_speaker.constants import c
from deep_speaker.unseen_speakers import inference_unseen_speakers, inference_embeddings
from scipy.spatial.distance import cosine



if __name__ == "__main__":
    base_dir = '/run/media/hbritto/Data/Datasets/deep-speaker-data'
    cache_dir = os.path.join(base_dir, 'cache', '')
    audio_dir = os.path.join(base_dir, 'VCTK-Corpus', '')
    reader = AudioReader(input_audio_dir=audio_dir, output_cache_dir=cache_dir,
                         sample_rate=c.AUDIO.SAMPLE_RATE, multi_threading=True)
    ids = deepcopy(reader.all_speaker_ids)
    me = 'Henrique'
    ids.remove(me)
    ids.remove('PhilippeRemy')
    index = len(ids) // 2
    ids[index:index] = [me]
    print(ids)
    # ids = ['p225', 'p226']
    # me_emb1 = inference_embeddings(reader, me)
    # me_emb2 = inference_embeddings(reader, me)
    # me_res = cosine(me_emb1, me_emb2)
    me_res = inference_unseen_speakers(reader, me, me)
    # res_dict = {me: me_res}
    # res_dict = {}
    # for p_id in ids:
    #     print(p_id)
    #     res = inference_embeddings(reader, p_id)
    #     res = cosine(me_emb, res)
    #     print(res)
    #     res_dict.update({p_id: res})
    #     # break
    # print(res_dict)
    # with open('dist_to_me.pkl', 'wb') as pkl:
    #     pickle.dump(res_dict, pkl)

    with open('dist_to_me.pkl', 'rb') as pkl:
        res_dict = pickle.load(pkl)
    res_dict.update({me: me_res})
    names_to_plot = list(res_dict.keys())[:30]
    values_to_plot = list(res_dict.values())[:30]
    names_to_plot[0:0] = [me]
    values_to_plot[0:0] = [res_dict[me]]
    print(res_dict[me])
    plt.figure()
    plt.scatter(names_to_plot, values_to_plot)
    plt.ylim((0.0, 0.1))
    plt.grid(True)
    plt.show()
