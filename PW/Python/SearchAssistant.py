import pickle
import AssistantTools as Tools

class Assistant:

    conversa = None
    allMatrixAllDataset = None
    tools = None

    def __init__(self, MATRIX_FRAC):
        self.conversa = pickle.load( open( "PW/Pickles/conversa.p", "rb" ) )
        if MATRIX_FRAC:
            self.allMatrixAllDataset = pickle.load( open( "PW/Pickles/allDatasetMatrix2.p", "rb" ) )
        else:
            self.allMatrixAllDataset = pickle.load( open( "PW/Pickles/allDatasetMatrix.p", "rb" ) )
        self.tools = Tools.getTools(self.conversa, self.allMatrixAllDataset)

    def getConversa(self):
        return self.conversa

    def getAllMatrixAllDataset(self):
        return self.allMatrixAllDataset

    # Version: Entity's weight between 0.5 - 1.0
    ## TIME: 9 minutes
    def computeEntityMatrix(self, MATRIX_FRAC):
        return self.tools.conversa
        #self.allMatrixAllDataset = self.tools.computeEntityMatrix(MATRIX_FRAC)


    def computePageRank(self, X):
        scores = self.tools.centrality_scores(X)
        return scores
        

# ------------------- Assistant Instanciation

def getAssistant(MATRIX_FRAC):
    a = Assistant(MATRIX_FRAC)
    return a



