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
    uploadedPt2Filepath=""
    uploadedOnnxFilepath=""
    optimizedPt2Filepath=""
    optimizedOnnxFilepath=""

    def dumpInTemp(self):
        data = {
            "csvFilepath": self.csvFilepath,
            "modelFilepath": self.modelFilepath,
            "weightsFilepath": self.weightsFilepath,
            "uploadedPt2Filepath": self.uploadedPt2Filepath,
            "uploadedOnnxFilepath": self.uploadedOnnxFilepath,
            "optimizedPt2Filepath": self.optimizedPt2Filepath,
            "optimizedOnnxFilepath": self.optimizedOnnxFilepath,
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
        self.uploadedPt2Filepath = ""
        self.uploadedOnnxFilepath = ""
        self.optimizedPt2Filepath = ""
        self.optimizedOnnxFilepath = ""
        self.name = ""
        self.id = ""

class OptimizationRequest:
    encoding = ""
    strategy = ""
    inputFeatures = []
    targetFeature = []
    epochs = 0
    generations = 5
    problemType = "regression"

    def __init__(self, encoding="", strategy="", inputFeatures=None, targetFeature=None, epochs=0, generations=5, problemType="regression"):
        self.encoding = encoding
        self.strategy = strategy
        self.inputFeatures = inputFeatures if inputFeatures is not None else []
        self.targetFeature = targetFeature if targetFeature is not None else []
        self.epochs = epochs
        self.generations = generations
        self.problemType = problemType

