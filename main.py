import pickle
import os

from infrastructure.inference import Inference
from infrastructure.register_person import register

if __name__ == '__main__':
    leave = False
    inference = Inference()
    cache_dir = '/run/media/hbritto/Data/Datasets/deep-speaker-data/cache/'
    with open(os.path.join(cache_dir, 'embeddings.pkl'), 'rb') as pkl:
        all_embs = pickle.load(pkl)
    inference.update_recogniser(all_embs)

    while not leave:
        ans = int(
            input('Pressione 1 para cadastrar uma nova pessoa, 2 para efetuar uma inferência ou 0 para sair: '))
        if ans == 1:
            base_dir = str(
                input('Digite o caminho absoluto para o diretório base onde os áudios serão salvos: '))
            name = str(input('Digite o nome da pessoa a ser cadastrada: '))
            person = register(base_dir, name)
            inference.update_recogniser(person)
        elif ans == 2:
            base_dir = '/run/media/hbritto/Data/Datasets/deep-speaker-data'
            name = 'Ephemeral'
            person = register(base_dir, name)
            identified_person, distance = inference.identify_person(
                next(iter(person.values())), return_distance=True)
            if identified_person:
                person_name = next(iter(identified_person.keys()))
                print('Pessoa identificada: {}'.format(person_name))
                print('Distância calculada do áudio enviado ao de {}: {}'.format(person_name, distance))
            else:
                print('Pessoa não identificada.')
        else:
            leave = True
