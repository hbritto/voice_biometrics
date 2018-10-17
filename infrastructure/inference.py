import sys
sys.path.append('..')
from voice_biometrics.blocks.voice_recogniser import VoiceRecogniser

class Inference:
    def __init__(self):
        self.recogniser = VoiceRecogniser()
    
    def identify_person(self, embedding, return_distance=False):
        if not self.recogniser.persons:
            return None
        
        person, dist = self.recogniser.recognise(embedding)
        if return_distance:
            return person, distance
        return person

    def update_recogniser(self, person: dict):
        if self.recogniser.persons:
            all_persons = {**self.recogniser.persons, **person}
        else:
            all_persons = person
        self.recogniser.persons = all_persons
