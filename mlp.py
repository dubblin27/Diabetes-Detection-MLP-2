import numpy as np

class mlp:
    
    def __init__(self,inputs,targets,nhidden,beta=.2,momentum=0.5,outtype='logistic'):
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

    def _computeError(self, inputs, targets):
        return .5*np.sum((targets-self.mlpfwd(inputs))**2)

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100, verbose=False):
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_error = np.inf
        new_error = self._computeError(valid, validtargets)
        
        count = 0
        while np.abs(old_error-new_error)/new_error > 0.01:
            count+=1
            if verbose: print count
            self.mlptrain(inputs,targets,eta,niterations)
            
            old_error = new_error
            new_error = self._computeError(valid, validtargets)
            
        print "Stopped", new_error,old_error
        return new_error
    	
    def mlptrain(self,inputs,targets,eta,niterations, verbose=False):
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
                      
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((targets-self.outputs)**2)
            if verbose and (np.mod(n,100)==0):
                print "Iteration: ",n, " Error: ",error    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (targets-self.outputs)/self.ndata
            elif self.outtype == 'logistic':
            	deltao = (targets-self.outputs)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (targets-self.outputs)/self.ndata
            else:
                print "Error: unknown neuron type"
            
            deltah = self.hidden*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))

            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1 += updatew1
            self.weights2 += updatew2
                
            np.random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]
            
    def mlpfwd(self,inputs):

        self.hidden = np.dot(inputs,self.weights1)
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        if self.outtype == 'linear':
       	    return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print "Error: unknown neuron type"

    def output(self, inputs):
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        return self.mlpfwd(inputs)

    def confmat(self,inputs,targets):
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print "Confusion matrix is:" 
        print cm
        print "Percentage Correct: ",np.trace(cm)/np.sum(cm)*100
