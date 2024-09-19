#importar librerías
import os
from openai import OpenAI
import json
from dotenv import load_dotenv, find_dotenv

#Se lee del archivo .env la api key de openai
_ = load_dotenv('api_key.env')
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('openai_api_key'),
)

#Se carga la lista de películas de movie_titles.json
with open('movie_titles.json', 'r') as file:
    file_content = file.read()
    movies = json.loads(file_content)

print(movies[0])


#Se genera una función auxiliar que ayudará a la comunicación con la api de openai
#Esta función recibe el prompt y el modelo a utilizar (por defecto gpt-3.5-turbo)
#devuelve la consulta hecha a la api

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

#Definimos una instrucción general que le vamos a dar al modelo 

instruction = "Vas a actuar como un aficionado del cine que sabe describir de forma clara, concisa y precisa \
cualquier película en menos de 200 palabras. La descripción debe incluir el género de la película y cualquier \
información adicional que sirva para crear un sistema de recomendación."

instruction_genre = "Vas a calificar la película en un género específico"

instruction_year = "Si sabes, vas a decir el año de lanzamiento de la película. Por favor únicamente el año de forma que se pueda convertir en un dato numérico"

#Definimos el prompt
movie = movies[0]['title']
prompt = f"{instruction} Has una descripción de la película {movie}"

print(prompt)

#Utilizamos la función para comunicarnos con la api
response = get_completion(prompt)

print(response)


# Podemos iterar sobre todas las películas para generar la descripción. Dado que esto 
#puede tomar bastante tiempo, el archivo con las descripciones para todas las películas es movie_descriptions.json

# for i in range(len(movies)):
#     prompt =  f"{instruction} Has una descripción de la película {movies[i]['title']}"
#     response = get_completion(prompt)
#     movies[i]['description'] = response 
#     prompt = f"{instruction_genre} Género de la película {movies[i]['title']}"
#     response = get_completion(prompt)
#     movies[i]['genre'] = response   
#     prompt = f"{instruction_year} Año de lanzamiento de la película {movies[i]['title']}"
#     response = get_completion(prompt)
#     movies[i]['year'] = response
#     print(movies[i]['title'])
#     print(movies[i]['genre'])
#     print(movies[i]['year'])

#     print(f"pelicula {i} de {len(movies)}")

# file_path = "movie_descriptions.json"

# # Write the data to the JSON file
# with open(file_path, 'w') as json_file:
#     json.dump(movies, json_file, indent=4)  # The 'indent' parameter is optional for pretty formatting

# print(f"Data saved to {file_path}")
