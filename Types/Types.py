import json


class SessionData:
    apiKey=""
    initialized=False

    def __init__(self):
        pass

class ProjectData:
    csvFilepath=""
    modelFilepath=""
    name=""
    id=""

    def dumpInTemp(self):
        data = {
            "csvFilepath": self.csvFilepath,
            "modelFilepath": self.modelFilepath,
            "weightsFilepath": self.weightsFilepath,
            "name": self.name,
            "id":self.id
        }
        
        print("Dumping project data to temp:", data)
        with open("temp_project/projectInfo.json", "w") as f:
            json.dump(data, f)

    def __init__(self):
        self.csvFilepath = ""
        self.modelFilepath = ""
        self.weightsFilepath = ""
        self.name = ""
        self.id = ""

class OptimizationRequest:
    def __init__(self, encoding="", strategy="", inputFeatures=None, targetFeature="", epochs=0, problem_type="regression"):
        self.encoding = encoding
        self.strategy = strategy
        self.inputFeatures = list(inputFeatures or [])
        self.targetFeature = targetFeature
        self.epochs = epochs
        self.problem_type = problem_type