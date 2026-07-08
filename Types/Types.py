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
    encoding = ""
    strategy = ""
    inputFeatures = []
    targetFeature = []
    epochs = 0
    generations = 5
