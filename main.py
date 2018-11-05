import pickle
import os

from infrastructure.inference import Inference
from infrastructure.register_person import register

if __name__ == '__main__':
    leave = False
    inference = Inference()
    base_dir = '/run/media/hbritto/Data/Datasets/deep-speaker-data'
    cache_dir = os.path.join(base_dir, 'cache', '')
    with open(os.path.join(cache_dir, 'embeddings.pkl'), 'rb') as pkl:
        all_embs = pickle.load(pkl)
    inference.update_recogniser(all_embs)

    while not leave:
        ans = int(
            input('Pressione 1 para cadastrar uma nova pessoa, 2 para efetuar uma inferência ou 0 para sair: '))
        if ans == 1:
            name = str(input('Digite o nome da pessoa a ser cadastrada: '))
            person = register(base_dir, name)
            inference.update_recogniser(person)
        elif ans == 2:
            name = 'Ephemeral'
            person = register(base_dir, name, n_audio=1)
            print('Identificando')
            identified_person, distance = inference.identify_person(
                next(iter(person.values())), return_distance=True)
            if identified_person:
                print(f'Pessoa identificada: {identified_person}')
                print(f'Distância calculada do áudio enviado ao de {identified_person}: {distance}')
            else:
                print('Pessoa não identificada.')
        else:
            leave = True
