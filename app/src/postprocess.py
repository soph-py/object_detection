from typing import Dict, List, Union

GenericNumber = Union[int, float]
Key, Value = Union[str, int], Union[str, GenericNumber]
Prediction = Dict[Key, Value] # singular prediction
Detections = Dict[Key, List[Prediction]] # list of predicted objects detected per image
IndexMap = List[Union[str, int]]

class Postprocess:
    """
    Processes predictions resulting from POST request to flask api
    """
    def __init__(self) -> None:
        self._detection_map = None

    @property
    def detection_map(self):
        """ 
        setter method
        """
        return self._detection_map

    @detection_map.setter
    def detection_map(self, value) -> None: # value: Tuple[IndexMap, Detections]
        self._detection_map = value[0], list(value[1].values())

    @detection_map.deleter
    def detection_map(self) -> None:
        del self._detection_map

    def updated_detections(self): #-> Detections: Dict[str, List[Dict[str, Union[float, str, int]]]]:
        """ 
        returns predicted objects back in original nested dict form, filtering out
        observations which had no predictions
        """
        return {
            int(key): value for key, value in self._updated_detections().items() if len(value) > 0
            }

    def _updated_detections(self): #-> Detections:
        return dict(zip(self.detection_map[0], list(self.detection_map[-1].values())))

    def simple_detections(self, conf: bool = False):
        """ 
        returns a simple version of nested dict format as returned in Postprocess.updated_detections()
        returning a dict of object class name as keys and observation indicies in a list as values
        :params conf: [bool], default: False
                      if false, returns dict of object classes as keys and list of observation indicies as values
                      if true, returns dict of object classes as keys and list of index, max(confidence) tuple pair
                      max(confidence) if multiple instances of object is detected, we take the max confidence 
        """
        if not conf:
            return self._simple()
        else:
            return self._confidence()

    def _confidence(self):
        object_index_mapping = dict(solar=[], pool=[])
        for key, value in self.updated_detections().items():
            confidence = dict(solar=[], pool=[])
            for prediction in value:
                if prediction['class'] == 3:
                    confidence['solar'].append(prediction['confidence'])
                elif prediction['class'] == 2:
                    confidence['pool'].append(prediction['confidence'])
            if len(confidence['solar']) > 0:
                object_index_mapping['solar'].append((key, max(confidence['solar'])))
            if len(confidence['pool']) > 0:
                object_index_mapping['pool'].append((key, max(confidence['pool'])))
        return object_index_mapping

    def _simple(self):
        object_index_mapping = dict(solar=[], pool=[])
        for key, value in self.updated_detections().items():
            for prediction in value:
                if prediction['class'] == 3:
                    object_index_mapping['solar'].append(key)
                if prediction['class'] == 2:
                    object_index_mapping['pool'].append(key)
        return object_index_mapping

    def object_count(self):
        object_counts = {key: [] for key in self.updated_detections().keys()}
        for key, value in self.updated_detections.items():
            counts1, counts2 = 0, 0
            for prediction in value:
                if prediction['class'] == 3:
                    counts1 += 1
                elif prediction['class'] == 2:
                    counts2 += 1
            object_counts[key].append([counts1, counts2])
        return object_counts

#  updated_detections = dict(zip(r.json()['index'], r.json()['detections'].values()))
    def objects_predicted(self):
        """ 
        example key: '3988'
        example value with 1 predicted object/detected object: [{
            'class': 3, 
            'confidence': 0.9543327689170837, 
            'name': 'solar', 
            'xmax': 210.7845458984375, 
            'xmin': 166.87924194335938, 
            'ymax': 265.52911376953125, 
            'ymin': 232.11036682128906
            }]
        example value with no predictions: []

        :return: detections with mapped index from df observation, filtering out observations
        with empty/no predictions
        """
        objects = dict()
        for key, value in self.updated_detections.items():
            if len(value) > 0:
                for pred in value:
                    ## for now, only take unique objects, bc we only care if there is at least 1 solar panel detected
                    object_ = dict(
                        confidence=pred['confidence'],
                        name=pred['name'],
                        xmax=pred['xmax'],
                        xmin=pred['xmin'],
                        ymax=pred['ymax'],
                        ymin=pred['ymin']
                        )
                    objects[key] = object_
            return objects

    def prediction_summary(self, count: bool = False):
        """ 
        :params count: opts - [True, False], default: False 
                       True: returns dict with key: index of observation and value: dict(solar=0 or 1, pool = 0 or 1)
                       False: returns dict of keys: objects, values: class id (0,1,2,3), 
                       total count of each object detected, nested list of xmin_max, ymin_max
                       of each object, list of confidence of each prediction of that particular object, 

        """
        ## just simply returns yes or no/true or false for classes
        # doesn't return anything for indicies of observations that have no predictions

        pass
