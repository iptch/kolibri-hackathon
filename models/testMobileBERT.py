from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np

from memory_profiler import profile


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
model = AutoModel.from_pretrained('google/mobilebert-uncased')
embeddingSize = 512

print("tokenizer and model are instialized...")
#@profile
def inference(text):

    # Tokenize input (as a batch of one sentence)
    inputs = tokenizer(text, return_tensors='pt')

    #print("tokenize input text...")

    # Forward pass through the model
    outputs = model(**inputs)

    
    #print("inference success.")

    # Get the embeddings from the last layer
    last_hidden_state = outputs.last_hidden_state

    #print("Get thhe embedding.")

    # Average the embeddings over all tokens to get a sentence-level embedding
    sentence_embedding = torch.mean(last_hidden_state, dim=1)

    #print(sentence_embedding)

    return sentence_embedding

def match(searchTermEmbedding, contentEmbeddings):
    # Fit the KNN model to the data
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(contentEmbeddings)

    # Use the fitted model to find the nearest neighbors to the query
    distances, indices = knn.kneighbors(searchTermEmbedding)

    print("Indices of nearest neighbors:", indices)
    print("Distances to nearest neighbors:", distances)
    return indices, distances

if __name__ == "__main__":
    print("Let us test MobileBERT and do some profiling...")

    research_fields = [
        'Biologi', 'Biologe', 'Biolog' , 'Biiology', 'Bioology', 'Biolog',
    "Astrophysics",
    "Biochemistry",
    "Biophysics",
    "Botany",
    "Climatology",
    "Cytology",
    "Ecology",
    "Ethology",
    "Genetics",
    "Geology",
    "Histology",
    "Immunology",
    "Meteorology",
    "Microbiology",
    "Neuroscience",
    "Paleontology",
    "Pathology",
    "Pharmacology",
    "Physiology",
    "Psychology",
    "Sociology",
    "Toxicology",
    "Virology",
    "Zoology",
    "Astronomy",
    "Geography",
    "Nanotechnology",
    "Oncology",
    "Ornithology",
    "Seismology",
    "Anthropology",
    "Archaeology",
    "Oceanography",
    "Pedology",
    "Phonetics",
    "Primatology",
    "Quantum Physics",
    "Radiology",
    "Rheology",
    "Semiotics",
    "Thermodynamics",
    "Volcanology",
    "Xenobiology",
    "Optics",
    "Mycology",
    "Nephrology",
    "Neurology",
    "Histopathology",
    "Gastroenterology",
    "Endocrinology",
    "Dermatology",
    "Cardiology",
    "Biogeography",
    "Agrostology",
    "Vexillology",
    "Trichology",
    "Spectroscopy",
    "Sedimentology",
    "Rhinology",
    "Quantum Mechanics",
    "Pulmonology",
    "Proteomics",
    "Paleobotany",
    "Otolaryngology",
    "Ophthalmology",
    "Oncogenomics",
    "Nucleonics",
    "Neuroethology",
    "Nematology",
    "Molecular Biology",
    "Metallurgy",
    "Mechanics",
    "Mammalogy",
    "Lichenology",
    "Lepidopterology",
    "Kinesiology",
    "Ichthyology",
    "Hematology",
    "Geophysics",
    "Gemology",
    "Entomology",
    "Enology",
    "Embryology",
    "Electrochemistry",
    "Dendrology",
    "Crystallography",
    "Criminology",
    "Cosmology",
    "Conchology",
    "Computational Biology",
    "Climatology",
    "Chronobiology",
    "Chemistry",
    "Cetology",
    "Ceramics",
    "Cell Biology",
    "Carpology",
    "Capillaroscopy",
    "Biostatistics",
    "Bioinformatics",
    "Astrobiology",
    "Arachnology",
    "Anthrozoology",
    "Anesthesiology",
    "Algology",
    "Aerodynamics",
    "Aerobiology",
    "Acoustics"
]

    searchTerm = "Biology"
    #""deep learning"

    searchTermEmbedding = inference(searchTerm)

    titles=["Machine Learning Fundamentals: From Theory to Practice",
    "Natural Language Processing: Unleashing the Power of Text",
    "Computer Vision: Exploring the World through Images",
    "Reinforcement Learning: Building Intelligent Agents",
    "Generative Adversarial Networks: Creating Artificially Intelligent Creations"]


    inputText = research_fields

    contentEmbeddings = np.empty((len(inputText), embeddingSize))

    for i, iText in enumerate(inputText):
        ce = inference(iText)
        contentEmbeddings[i] = ce.detach().numpy()

    indices, distances = match(searchTermEmbedding.detach().numpy(), contentEmbeddings)

    print('nearest neighbor for ', searchTerm, ' is ',
    inputText[indices[0][0]], 
    inputText[indices[0][1]], 
    inputText[indices[0][2]],
    inputText[indices[0][3]],
    inputText[indices[0][4]])

    print(searchTermEmbedding)

    print(contentEmbeddings[indices[0][0]])


    

