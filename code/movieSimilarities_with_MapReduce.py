

from mrjob.job import MRJob
from mrjob.step import MRStep
from math import sqrt
import matplotlib.pyplot as plt
# Import all the Similarity Measures from the Similarity Metrics python file
from Similarity_Metrics import norm,cosine,jaccard,generalized_jaccard,pearson_correlation,normalized_pearson_correlation

'''
The below function expands the item,rating information. 
For all the (item,rating) pairs of all the users, the function gives a combination of 
(item,item),(rating,rating). This format is useful in extracting both the Movie Vectors and to compute the
similarity between them.

Eg:- For a Movie,Rating Pairs given by various users (Movie1,R1),(Movie2,R2) it generates 
(Movie1,Movie2) and (R1,R2) Pairs.
'''
def expand_ratings(l1):
    c = len(l1)
    i = 0;
    tot = list()
    for idx, elem in enumerate(l1):
        i = i+1
        j = i
        thiselem = elem
        while j < c:
            nextelem = l1[(j) % c]    
            a = (thiselem[0],nextelem[0])
            b = (thiselem[1],nextelem[1])
            tot.append((a,b))
            j = j+1
    return tot

'''
The Below Class implements the actual Map Reduce Algorithm.
The Job contains three steps, namely, Mapper , Combiner and Reducer.

Mapper Function:-
The Mapper reads the input from the user file which has schema user_name|item_name|rating
The Mapper groups the records by user name and emits a set of (item_name,rating) pairs

Combiner Function:-
The combiner reads the output from the Mapper partitioned by user_name. So all the user's set go into 
combiner. We extract all the movie,rating combinations from the expand_rating function.
From these combinations, we pair up the movie combinations along with the rating information and is emitted as 
Combiner output.

Reducer Function:-
The reducer takes the output from the Combiner partitioned by movie combinations. All the set of movie 
combinations rated by various users come into reducer. From these set, we form MovieA and MovieB vectors. 
We use these vectors to calculate the similarity between them.
We emit the values from the similarity measures along with their movie pairs.
'''
class MoviesSimilarities(MRJob):

    def steps(self):
        return [MRStep(mapper=self.group_by_user_rating,
                       combiner = self.get_pairwise_items,
                       reducer=self.pairwise_items_similarity)]

    def group_by_user_rating(self, key, line):

        user_id, item_id, rating = line.split('|')
        yield  user_id, (item_id, float(rating))

    def get_pairwise_items(self, user_id, values):
        item_count = 0
        item_sum = 0
        final = []
        movie_final = list()
        movie_ratings = list()
        for item_id, rating in values:
            item_count += 1
            item_sum += rating
            final.append((item_id, rating))
            
        rat = expand_ratings(final)
        for item1, item2 in rat:
            yield (item1[0], item1[1]), \
                    (item2[0], item2[1])

    def pairwise_items_similarity(self, user_id, values):
        movieA = list()
        movieB = list()
        item1,item2 = user_id
        for val in values:
            movieA.append(val[0])
            movieB.append(val[1])
        Cosine_Similarity = round(cosine(movieA,movieB), 5)
        Jaccard_Similarity = round(jaccard(movieA,movieB) , 5)
        Generalized_Jaccard_Similarity =round(generalized_jaccard(movieA,movieB),5)
        Pearson_Correlation = round(pearson_correlation(movieA,movieB) , 5)
        Normalized_Pearson_Correlation =round(normalized_pearson_correlation(movieA,movieB) , 5)
        yield (item1,item2),(Cosine_Similarity,Jaccard_Similarity,Generalized_Jaccard_Similarity,Pearson_Correlation,Normalized_Pearson_Correlation)
        
if __name__ == '__main__':
    MoviesSimilarities.run()