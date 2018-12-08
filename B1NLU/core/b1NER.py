# implementing the named entity recognition (NER) needed to extract the new types of entities/slots
# we train a blank NER (using spacy library) on a data set of tagged utterances, save it to b1_NER_model_v1. the training code is not included here,
# instead we load and make use of the trained model to apply NER on the input sentence

# import required libraries
import spacy
import os


# load the intents file


# directory of B1 NER model
pwd = os.getcwd()
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, pwd+'/models/b1_NER_model/')

# load trained b1 NER model for new entities
# we name it nlp2 because we need to load our own nlp model
# we can use nlp to load the built-in spacy en model
nlp2 = spacy.load(output_dir)


# extract entities/slots using the trained b1 NER model
def getB1Entities(sentence):
    # build a document by applying our ner model on the input sentence
    document2 = nlp2((sentence))
    # a set of labels in the document
    labelset = set([w.label_ for w in document2.ents])
    lists = []
    for label2 in labelset:
        entities = [cleanup(e.string, lower=False) for e in document2.ents if label2 == e.label_]
        entities = list(set(entities))
        lists.append([label2, entities])


    # Entities
    destination = 'DESTINATION' # <named location>: B1 can move to any previously-named location under voice command (e.g., "Go to <the cancer center>")
    distance = "MoveDistance" # a specified distance B1 should move for.
    unit = "Unit" # the unit of distance/angle
    angle = "Angle" # a specified angle B1 should rotate by.
    direction = "Direction"  #  a specified dierction B1 should roatate by.

    restaurantName = "RESTAURANT.NAME"
    workingHours = "RESTAURANT.WORKING_HOURS"
    locationClose = "RESTAURANT.LOCATION.CLOSE"

    #food = "FOOD.NAME"
    size = "FOOD.SIZE"
    #location ="LOCATION.PLACE_TYPE"
    category = 'FOOD.CATEGORY' # category = food = food.name = dish = food.dish = course
    allergy = 'FOOD.ALLERGY'

    ingredient = "FOOD.INGREDIENT"  #
    ingredients = []
    categories = []

    # initiate a Key:Value list to save extracted slots/entities
    bdata = {}
    # loop for extracting found entities/slots: moveDistance, unit, direction, angle and destination
    for item in lists:
        key = item[0]
        value = item[1]

        if str(key) in restaurantName:
            for i in value:
                bdata['RESTAURANT.NAME'] = str(i)
        elif str(key) in workingHours:
            for i in value:
                bdata['RESTAURANT.WORKING_HOURS'] = str(i)

        elif str(key) in locationClose:
            for i in value:
                bdata['RESTAURANT.LOCATION.CLOSE'] = str(i)

        elif str(key) in size:
            for i in value:
                bdata['FOOD.SIZE'] = str(i)
        elif str(key) in category:
           for i in value:
               categories.append(str(i))
               bdata['FOOD.CATEGORY'] = categories
        elif str(key) in allergy:
            for i in value:
                bdata['FOOD.ALLERGY'] = str(i)
        elif str(key) in ingredient:
           for i in value:
                ingredients.append(str(i))
                bdata['FOOD.INGREDIENT'] = ingredients

    cat = []
    cat = getCategory(sentence)
    if 'FOOD.CATEGORY' not in bdata:
        if cat:
            bdata['FOOD.CATEGORY'] = cat
            bdata['intent'] = "OrderFood"
    rest = []
    rest = getRestaurant(sentence)
    if 'RESTAURANT.NAME' not in bdata:
        if rest:
            bdata['RESTAURANT.NAME'] = rest
            bdata['intent'] = "OrderFood"

    return bdata


# cleanup
def cleanup(token, lower=True):
    if lower:
        token = token.lower()
    return token.strip()



def getCategory(text):
    foodCategory = [
    "Afghan",
    "African",
    "Alcohol",
    "All Night Alcohol",
    "American",
    "Arabic",
    "Argentinian",
    "Asian",
    "Australian",
    "Authentic Pizza",
    "Azerbaijan",
    "Bagels",
    "Baguettes",
    "Balkan",
    "Balti",
    "Bangladeshi",
    "Basque",
    "BBQ",
    "Belgian Waffles",
    "Best Bites",
    "Biryani",
    "Brazilian Food",
    "Breakfast",
    "British",
    "Brunch",
    "Bubble Tea",
    "Bulgarian",
    "Burgers",
    "Burmese",
    "Burritos",
    "Business Lunch",
    "Café",
    "Cakes",
    "Cantonese",
    "Caribbean",
    "Chicken",
    "Chinese",
    "Colombian",
    "Continental",
    "Crepes",
    "Cuban",
    "Curry",
    "Danish",
    "Desserts",
    "Dim Sum",
    "Dinner",
    "Drinks",
    "Eastern European",
    "Egyptian",
    "English",
    "English Breakfast",
    "Ethiopian",
    "European",
    "Fast Food",
    "Filipino",
    "Fish",
    "Chips",
    "French",
    "Fusion",
    "Georgian",
    "German",
    "Ghanaian",
    "Gluten Free",
    "Gourmet",
    "Gourmet Burgers",
    "Greek",
    "Grill",
    "Healthy",
    "Hot Dogs",
    "Hungarian",
    "Ice Cream",
    "Indian",
    "Indo-Chinese Fusion",
    "Indonesian",
    "Iranian",
    "Iraqi",
    "Italian",
    "Italian Pizza",
    "Jamaican",
    "Japanese",
    "Jerk",
    "Kebabs",
    "Korean",
    "Kosher",
    "Kurdish",
    "Latin American",
    "Lebanese",
    "Low-Carb",
    "Lunch",
    "Malaysian",
    "Mauritian",
    "Mediterranean",
    "Mexican",
    "Middle Eastern",
    "Milkshakes",
    "Mongolian",
    "Moroccan",
    "Nepalese",
    "Nigerian",
    "Noodles",
    "North African",
    "Norwegian",
    "Organic",
    "Oriental",
    "Pakistani",
    "Pancakes",
    "Pancakes",
    "Panini",
    "Parmesans",
    "Pasta",
    "Peri Peri",
    "Persian",
    "Peruvian",
    "Pick n Mix",
    "Pizza",
    "Polish",
    "Portuguese",
    "Pub Food",
    "Punjabi",
    "Retro Sweets",
    "Roast Dinners",
    "Romanian",
    "Rotisserie",
    "Russian",
    "Salads",
    "Salt & Pepper",
    "Sandwiches",
    "Scottish",
    "Seafood",
    "Singapore",
    "Sizzlers",
    "Snacks"
    "Smoothies",
    "Soup",
    "South African",
    "South American",
    "South Indian",
    "Spanish",
    "Srilankan",
    "Steak",
    "Street Food",
    "Subways",
    "Sushi",
    "Sweets",
    "Syrian",
    "Tapas",
    "Tex-Mex",
    "Thai",
    "Trinidadian",
    "Turkish",
    "Ukrainian",
    "Vegan",
    "Vegetarian",
    "Vietnamese",
    "Waffles",
    "West African",
    "Wraps"
    ]
    cat = []
    cdata = {}
    for i in range(len(foodCategory)):
        if foodCategory[i].lower() in text or foodCategory[i] in text:
            cat.append(foodCategory[i].lower())
            cdata['FOOD.CATEGORY'] = cat


    return cat

def getRestaurant(text):
    restaurants = [
        "Shake Shack",
        "Plum Market",
        "Roasting Plant Coffee",
        "Roasting Plant"]
    rest = []
    crest = {}
    for i in range(len(restaurants)):
        if restaurants[i].lower() in text or restaurants[i] in text:
            rest.append(restaurants[i].lower())
            crest['FOOD.CATEGORY'] = rest

    return rest