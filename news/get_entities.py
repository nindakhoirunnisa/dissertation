from model.predict import PredictLabel

class GetEntities:
    def __init__(self):
        self.predictor = PredictLabel("./save/20230809204219")

    def categorize_entities(self, title):
        entities_result = self.predictor.get_entities_result(title)
        categories = {'organisation': [], 'address': [], 'person': []}
        print(entities_result)

        for entity in entities_result:
            if entity['type'] in categories:
                categories[entity['type']].append(entity['value'])

        return categories
