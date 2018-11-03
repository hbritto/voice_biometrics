# -*- coding: utf-8 -*-
"""Módulo de registro de novos embeddings de pessoas."""

import pickle
import os
import sys
sys.path.append('..')
from voice_biometrics.deep_speaker.audio_reader import AudioReader
from voice_biometrics.deep_speaker.constants import c
from voice_biometrics.deep_speaker.unseen_speakers import inference_embeddings
from voice_biometrics.audio_utils.recorder import Recorder


def _update_cache(base_dir, person_folder):
    cache_dir = os.path.join(base_dir, 'cache', '')
    audio_reader = AudioReader(input_audio_dir=person_folder,
                               output_cache_dir=cache_dir,
                               sample_rate=c.AUDIO.SAMPLE_RATE,
                               multi_threading=True)
    audio_reader.build_cache()
    # Atualizando cache interno
    audio_reader = AudioReader(input_audio_dir=person_folder,
                               output_cache_dir=cache_dir,
                               sample_rate=c.AUDIO.SAMPLE_RATE,
                               multi_threading=True)
    return audio_reader


def _make_embedding(audio_reader, name):
    embed = inference_embeddings(audio_reader, name)
    return embed


def register(base_dir, name, n_audio=4):
    person_folder = os.path.join(base_dir, name, '')
    os.makedirs(person_folder, exist_ok=True)
    rec = Recorder()

    for i in range(n_audio):
        print('Áudio número {:>02d} de {:>02d}'
              .format(i + 1, n_audio))
        with rec.open(os.path.join(person_folder, '{}_{:>03d}.wav'.format(name, i))) as recfile:
            recfile.record(8)
    print('Áudio gravado')
    reader = _update_cache(base_dir, person_folder)
    print('Cache atualizado')
    embed = _make_embedding(reader, name)
    print('Embedding criado')
    person = {name: embed}
    with open(os.path.join(base_dir, name + '.pkl'), 'wb') as pkl:
        pickle.dump(person, pkl)

    return person
