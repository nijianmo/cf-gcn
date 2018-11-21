from ..sequence import WordSequence

class Beer(object):

    def __init__(self,
                 id,
                 name,
                 brewer,
                 style,
                 abv):
        self.id = id
        self.name = name
        self.brewer = brewer
        self.style = style
        self.abv = abv

    @staticmethod
    def from_json(obj):
        return Beer(
            int(obj['beer/beerId']),
            obj['beer/name'],
            int(obj['beer/brewerId']),
            obj['beer/style'],
            obj.get('beer/ABV', None)
        )

class Review(object):

    def __init__(self,
                 beer,
                 rating_overall,
                 rating_palate,
                 rating_taste,
                 rating_appearance,
                 rating_aroma,
                 text,
                 user
                 ):
        self.beer = beer
        self.rating_overall = rating_overall
        self.rating_palate = rating_palate
        self.rating_taste = rating_taste
        self.rating_appearance = rating_appearance
        self.rating_aroma = rating_aroma
        self.text = text
        self.user = user

    @property
    def ratings(self):
        return [self.rating_overall,
                self.rating_palate,
                self.rating_taste,
                self.rating_appearance,
                self.rating_aroma]

    @staticmethod
    def from_json(obj, beer):
        return Review(
            beer,
            obj['review/overall'],
            obj['review/palate'],
            obj['review/taste'],
            obj['review/appearance'],
            obj['review/aroma'],
            obj['review/text'],
            obj['user/profileName'],
        )
