from django.core.management.base import BaseCommand, CommandParser
from movie.models import Movie
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

def get_embedding(text, client, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Command(BaseCommand):
    help = 'Find the movie most similar to the given request'
    
    def add_arguments(self, parser):
        parser.add_argument('req', type=str, help='Request description to compare with movie embeddings')

    def handle(self, *args, **kwargs):

        #Se lee del archivo .env la api key de openai
        _ = load_dotenv('../api_key.env')
        client = OpenAI(
        # This is the default and can be omitted
            api_key=os.environ.get('openai_api_key'),
        )
        
        items = Movie.objects.all()
        req = kwargs['req']
        # req = 'película de la segunda guerra mundial'

        
        emb_req = get_embedding(req, client)

        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)
        # print(items[idx].title)
        return items[idx].title