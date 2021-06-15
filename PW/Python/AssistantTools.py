import urllib
import requests
import numpy as np
from time import time
from scipy.sparse import csr_matrix

class Tools:

    conversa = None
    allMatrixAllDataset = None

    def __init__(self, conv, matrix):
        self.conversa = conv
        self.allMatrixAllDataset = matrix
    

    # COMPUTING ENTITY MATRIX ---------------------------------------------------------------------------
    ## Version: Entity's weight between 0.5 - 1.0
    ### TIME: 9 minutes
    def computeEntityMatrix(self, MATRIX_FRAC):
        t0 = time()
        self.allMatrixAllDataset=[]
        headers = {'Accept': 'application/json'}
        for topicos in self.conversa:
            allMatrix = []
            for perguntas in topicos:
                hisdict= {}
                matrix = []
                valor = 1
                for k in range(len(perguntas)):
                    #Enconding
                    payload = urllib.parse.quote(perguntas[k])
                    #Request
                    r =requests.get("https://api.dbpedia-spotlight.org/en/annotate?text="+payload,headers=headers)
                    try:
                        for n in r.json()["Resources"] :
                            entity = n["@surfaceForm"].lower()
                            if entity not in hisdict:
                                hisdict[entity] = len(matrix)
                                matrix.append([entity,np.zeros(11,dtype=float)])
                            pos = hisdict[entity]
                            aux = matrix[pos][1]
                            aux[k]=valor
                            matrix[pos][1] = aux
                    except:
                        continue
                    if MATRIX_FRAC:
                        valor = valor - 0.05
                allMatrix.append(matrix)   
            self.allMatrixAllDataset.append(allMatrix)

        print("Done in %0.3fs" % (time() - t0))
        return self.allMatrixAllDataset


    # COMPUTING PAGE RANK ---------------------------------------------------------------------------
    def centrality_scores(self, X, alpha=0.85, max_iter=100, tol=1e-10):
        """Power iteration computation of the principal eigenvector

        This method is also known as Google PageRank and the implementation
        is based on the one from the NetworkX project (BSD licensed too)
        with copyrights by:

        Aric Hagberg <hagberg@lanl.gov>
        Dan Schult <dschult@colgate.edu>
        Pieter Swart <swart@lanl.gov>
        """
        n = X.shape[0]
        X = X.copy()
        incoming_counts = np.asarray(X.sum(axis=1)).ravel()

        #print("Normalizing the graph")
        for i in incoming_counts.nonzero()[0]:
            X.data[X.indptr[i]:X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
        dangle = np.asarray(np.where(np.isclose(X.sum(axis=1), 0),
                                    1.0 / n, 0)).ravel()

        scores = np.full(n, 1. / n, dtype=np.float32)  # initial guess
        for i in range(max_iter):
            #print("power iteration #%d" % i)
            prev_scores = scores
            scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                    + (1 - alpha) * prev_scores.sum() / n)
            # check convergence: normalized l_inf norm
            scores_max = np.abs(scores).max()
            if scores_max == 0.0:
                scores_max = 1.0
            err = np.abs(scores - prev_scores).max() / scores_max
            #print("error: %0.6f" % err)
            if err < n * tol:
                return scores

        return scores



# ------------------- Tools Instanciation

def getTools(conv, matrix):
    t = Tools(conv, matrix)
    return t