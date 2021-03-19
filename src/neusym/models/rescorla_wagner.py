# -*- coding: utf-8 -*-


from .base import BaseModel


class RWModel(BaseModel):
    def __init__(self, lower, upper):
        super(RWModel, self).__init__(lower, upper)
        self.objs = dict()
    
    def train(self, context):
        for view in context:
            view_light_state = view["light_state"]
            view_objs = []
            for obj in view["objects"]:
                obj_id = "+".join([obj["shape"], obj["color"], obj["material"]])
                # in case of duplicates
                if obj_id in view_objs:
                    continue
                else:
                    view_objs.append(obj_id)
                if obj_id in self.objs:
                    self.objs[obj_id][view_light_state] += 1
                else:
                    self.objs[obj_id] = {"on": 0, "off": 0}
                    self.objs[obj_id][view_light_state] += 1
        for obj_id in self.objs:
            self.objs[obj_id]["blicketness"] = self.objs[obj_id]["on"] / (self.objs[obj_id]["on"] + self.objs[obj_id]["off"])
    
    def test(self, query):
        pred = []
        for view in query:
            probs = []
            for obj in view["objects"]:
                obj_id = "+".join([obj["shape"], obj["color"], obj["material"]])
                if obj_id in self.objs:
                    probs.append(self.objs[obj_id]["blicketness"])
                else:
                    probs.append(0.5)
            pred.append(self.predict(max(probs)))
        return pred
