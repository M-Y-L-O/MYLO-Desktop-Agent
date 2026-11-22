import json
import os


class SessionData:
    def __init__(self):
        self.apiKey = ""
        self.initialized = False


class ProjectData:
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
            "id": self.id,
        }

        os.makedirs("temp_project", exist_ok=True)
        with open("temp_project/projectInfo.json", "w", encoding="utf-8") as f:
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
    def __init__(
        self,
        encoding="",
        strategy="",
        inputFeatures=None,
        targetFeature=None,
        epochs=0,
        generations=5,
        problem_type="regression",
        problemType=None,
        validationSplit=0.2,
        testSplit=0.1,
        randomState=42,
    ):
        self.encoding = encoding
        self.strategy = strategy
        self.inputFeatures = inputFeatures if inputFeatures is not None else []
        self.targetFeature = targetFeature if targetFeature is not None else []
        self.epochs = epochs
        self.generations = generations
        self.problem_type = problemType or problem_type or "regression"
        self.validation_split = validationSplit
        self.test_split = testSplit
        self.random_state = randomState
