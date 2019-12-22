from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

texts = [
    "Only two years later, all these friendly Sioux were suddenly plunged into new conditions, including starvation, martial law on all their reservations, and constant urging by their friends and relations to join in warfare against the treacherous government that had kept faith with neither friend nor foe.",
    'In ages which have no record these islands were the home of millions of "Contrast the condition into which all these friendly Indians are suddenly plunged now, with their condition only two years previous: martial law now in force on all their reservations; themselves in danger of starvation, and constantly exposed to the influence of emissaries from their friends and relations, urging them to join in fighting this treacherous government that had kept faith with nobody--neither with friend nor with foe.',
]
df = pd.DataFrame(texts)
# print(df)

cmatrix = CountVectorizer().fit_transform(texts)

sim = cosine_similarity(cmatrix)
print(sim[0][1] * 100, "%")
