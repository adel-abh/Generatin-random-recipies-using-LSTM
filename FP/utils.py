import re
import matplotlib.pyplot as plt
from time import perf_counter
import itertools
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from pattern.en import pluralize, singularize
import enchant
from deep_translator import (GoogleTranslator,
                             MicrosoftTranslator,
                             PonsTranslator,
                             LingueeTranslator,
                             MyMemoryTranslator,
                             YandexTranslator,
                             PapagoTranslator,
                             DeeplTranslator,
                             QcriTranslator,
                             single_detection,
                             batch_detection)
#nltk.download('wordnet')
p=WordNetLemmatizer()
d = enchant.Dict("en_GB")

def word_cloud(col,w=800,h=600):
    """this function helps with creating a wordcloud
    Arguments:
        input : Dataframe column which contains the recipes
        w : width of your chart (default 800)
        h : height of your chart (default 600)
    Returns:
        wordcloud chart which is saved locally in the same directory 
    """
    col.to_list()
    flat_list = [item for sublist in col for item in sublist]
    text=" ".join(flat_list)
    wordcloud = WordCloud(background_color="white",width=w, height=h,).generate(text)

    wordcloud.to_file("output.png")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
def get_sentences(textfile,start,end):
    """returns chunk of text between two custom delimieters
Arguments:
        textfile: a String representing the path to the file
        start: a string representing the first delimiter
        end: a string representing the second delimiter
    returns:
        a list of recipes
 """

    ls=[] 

    with open(textfile,"r",encoding='UTF-8') as fp:
        for result in re.findall(f'{start}(.*?){end}',fp.read(),re.S):
            ls.append(result)
    return ls

def spellcheck(words,data):
    '''takes a list of words ,checks if the word exists in the molecular dataset , else turns it into a singular form
    if that fails too return the original list of words as a String
    Arguments:
        words: a list of strings representing ingredients
        data : Dataframe representing the molecular dataset
    returns:
        a string representing the properly spelled ingredient

    '''
    
    if lookup(" ".join(words),data) is not None: # check if the ingredient is in the databse without any modifications
        return " ".join(words)
    
    check_words=[d.check(singularize(word))==True for word in words] # else check if the sinular form of the word is in the database
    if all(check_words):
        return " ".join([singularize(word) for word in words])
    else:
        return " ".join(words)
    

                        
def lookup(ingridient,data):
    """takes an ingredient and tries to find the ID in flavordb ,returns None when it fails to find it
    Arguments:
        ingredient: a String representing the ingredient 
        data: Dataframe representing the molecular dataset
    returns:
        id : an int representing the id if it exists , else None
    """
    try:
        return int(data.loc[data['alias']==(ingridient)]['entity id'])

    except:
        try:
            return int(data.loc[data['alias']==(p.lemmatize(ingridient))]['entity id'])
            
        except:
            return None
        
 
def count_None(col):
    """to count how many None values are still left and the troubling ingredients
    Arguments:
        col: a dataframe column which contains the recipes
    returns :
        a print statement containing the total count of None values , and the ingredients that caused this issue
        trouble_ing :  a list containing the ingredients containing the None values
    """ 
    count=0
    trouble_ing=[]
    for element in col: #df['Ingredients_alpha']:
        count+=list(element.values()).count(None)
        for k,v in element.items():
            if element[k]==None:
                trouble_ing.append(k)
    print(f"there are {count} None values \nAnd the troubling ing list is {set(trouble_ing)}")
    return trouble_ing


def show_missing(trouble_ing,df):
    """this functions takes the ingredients list that is missing , counts how many times each missing ingredient was mentioned 
     and in which recipe
     Arguments:
        trouble_ing : the output of count_None()
        df: the dataframe containing the recipes list and names
    returns:

     """
    d=dict.fromkeys(trouble_ing, "") 
    count=0
    dic=dict()
    for element in set(trouble_ing):
        count=0
        print(element)
        
        for i,ls in df.iterrows():
            if element in ls['Ingredients_alpha']:
                print(df['recipe_name'].iloc[i][0])
                print(f'and the index is {i}')
                count=count+1
        d[element]=count
        print('----------------------------------------')
    print(d) 

def lookup_set(ingredient,data): 
    """Arguments:
        ingridient: String representing the ingrediant alias 
        data : Dataframe representing the molecular dataset 
    returns
        set of molecules of that ingredient"""
    return eval(data.query(f"alias.str.fullmatch('{ingredient}')")['molecules'].tolist()[0])

def avg_flavor(recipe):
    '''calculates the average flavor sharing of one recipe 
    Arguments
        recipe : dictionary which represents a recipe
    Returns
        score :a float representing average flavor sharing'''
    somme=0
    nR=len(recipe.keys())
    combs=list(itertools.combinations(recipe.keys(), 2))
    for element in combs:
        somme=somme+(len(recipe[element[0]].intersection(recipe[element[1]])))
    score=somme*2/(nR*(nR-1))
    return score
def calculate_IFW(col,recipe,occurence_dic):
    """takes a recipe and associate weights with each ingredient
    Arguments:
        col :dataframe column containing the recipes in english
        recipe: a dictionary representing a recipe
        occurence_dic: dictionary containing ingredients with their occurence
    returns:
        it modifies the recipe dictionary by assigning a float representing the fitness value to each ingredient
    
    """
    for ing,value in recipe.items():
        lengths=0
        for rec in col:
            rec=eval(rec)
            if ing in rec.keys():
                lengths=lengths+len(rec.keys())
        recipe[ing]=occurence_dic[ing]/lengths
    return recipe
def calculate_frequency(recipe,occurence_dic):
    """ takes a recipe and associate weights with each ingredient
    Arguments:
        recipe: dictionary representing a recipe
        occurence_dic: dictionary containing ingredients with their occurence
    returns:
        it modifies the recipe dictionary by assigning a float representing the fitness value to each ingredient"""

    for ing,value in recipe.items():

        recipe[ing]=occurence_dic[ing]/232
    return recipe


   
