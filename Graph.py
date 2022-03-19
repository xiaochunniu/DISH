import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Generate a Random Graph (Erdos Renyi)
class GraphGen:
    def __init__(self, nclient, prob_edge):
        self.nclient = nclient
        self.prob_edge = prob_edge

    def run(self):
        # Use seed for reproducibility
        G = nx.erdos_renyi_graph(self.nclient, self.prob_edge)
        
        # A is the adjacency matrix
        A = nx.adjacency_matrix(G)
        A = A.todense()
        A =  np.squeeze(np.asarray(A))
        
        # D is the degree matrix
        d_max = 0
        for v in nx.nodes(G):
            d_max = max(d_max, nx.degree(G, v))    
        
        A = A / (d_max + 1)
        for i in range(self.nclient):
            A[i][i] = 1 - np.sum(A[i])
        # A is a consensus matrix
        #Z = (np.linalg.inv(D).dot(A) + np.identity(self.nclient))/2

        #adjlist = []
        #for line in nx.generate_adjlist(G):
        #    adjlist.append([int(s) for s in line.split(' ')])
        
        labels = {i:i+1 for i in range(self.nclient)}
        nx.draw(G, with_labels=True, labels=labels)
        #plt.show()
        plt.savefig("GraphER.pdf")
        
        return A#, adjlist

# Generate a Random Graph (Erdos Renyi)
class CompleteGraph:
    def __init__(self, nclient):
        self.nclient = nclient

    def run(self):
        # Use seed for reproducibility
        G = nx.complete_graph(self.nclient)
        
        # A is the adjacency matrix
        A = np.ones([self.nclient, self.nclient])
        A = A / self.nclient
        # A is a consensus matrix
        #Z = (np.linalg.inv(D).dot(A) + np.identity(self.nclient))/2

        #adjlist = []
        #for line in nx.generate_adjlist(G):
        #    adjlist.append([int(s) for s in line.split(' ')])
        
        labels = {i:i+1 for i in range(self.nclient)}
        nx.draw(G, with_labels=True, labels=labels)
        #plt.show()
        plt.savefig("CompleteGraph.pdf")
        
        return A#, adjlist