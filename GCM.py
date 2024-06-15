import time 
import torch 
from tqdm import tqdm 
import random 
import datetime
from torch import nn, sqrt, exp, optim, tensor, stack, moveaxis, argmax, no_grad
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
#from torchmetrics.classification import MultilabelPrecision

class ModuleAttention(nn.Module):

    def __init__(self,n_dim):
        super().__init__()
        self.layer_1 = nn.Linear(n_dim, 1)
    
    def forward(self, X):
        #out = self.layer_1(X)
        #with torch.no_grad():
        #    weight_norm = torch.abs(self.layer_1.weight)/(torch.abs(self.layer_1.weight).sum(dim=1, keepdim=True))
        #    self.layer_1.weight.copy_(weight_norm)

        return self.layer_1(X).reshape((X.shape[0],X.shape[1]))
        #return out.reshape((X.shape[0],X.shape[1]))

class ModulePrestige(nn.Module):

    def __init__(self, n_dim):
        super().__init__()
        pass 

class GCM(nn.Module):

    def __init__(self,
                 df,
                 fuzzy_param,
                 prieur_beta,
                 dim_attention,
                 date,
                 dim_meta,
                 class2text,
                 text2class,
                 meta_df):
        super().__init__()
        self.data = df 
        self.meta_df = meta_df 

        # Dictionnary with | key : categories | items : list of texts belonging to this category
        self.class2text = class2text 
        # Dictionary with  | key : text | items : list of categories to which it belongs
        self.text2class = text2class

        self.class2index = {k : e for e,k in enumerate(list(class2text.keys()))}

        #self.X, self.y = df.iloc[:,:-1], df.iloc[:,-1]
        self.classes = list(range(len(self.class2text.keys())))
        self.date = date 

        self.fuzzy_param = fuzzy_param
        self.prieur_beta = prieur_beta
        self.attention = nn.ModuleList([ModuleAttention(dim_attention) for c in range(len(self.classes))])
        self.attention.to("mps")
        #self.prestige = nn.ModuleList([ModulePrestige(dim_meta) for c in self.classes])

    

    def get_exemplars(self, n_exemplars=25):
        self.exemplars, self.non_exemplars = [], []
        self.train_eval, self.test = [], []


        working_df = self.data[self.data.index.get_level_values(0) <= self.date]
        
        for e,classe in enumerate(self.class2text.keys()):
            # Adding a security if there is not enough articles belonging to the class
            selected_texts = random.sample(self.class2text[classe], k=n_exemplars)
            self.exemplars.append(working_df[working_df.index.get_level_values(1).isin(selected_texts)].groupby(level=1).sample(10, replace=True))
            self.non_exemplars.append([x for x in self.class2text[classe] if not x in selected_texts])
            train, test = train_test_split(self.non_exemplars[e], train_size=0.7, random_state=14061021)
            self.train_eval.append(train)
            self.test.append(test)
        
        self.exemplars = [torch.from_numpy(exemplar.to_numpy()).to(torch.float32).to("mps") for exemplar in self.exemplars]


    def get_exemplars_old(self, strategie="etendue", n_exemplars = 10):
        """
            Three stategies :
                - random : pick random exemplars
                - qualité : pick random exemplars which are qualitatively interesting
                - etendu : pick random exemplars, wheigted by the time they existed as such * nb of contributors
        """
        working_df = self.data[self.data.index.get_level_values(0) <= self.date]
        working_meta_df = self.meta_df.loc[working_df.index,:]

        ## Il faut que pour chaque catégories l'ensemble du cop
        if strategie == "random":
            
            exemplars, categories =  working_df.iloc[:,:-1], working_df.iloc[:,-1]
            self.exemplars, self.exemplar_category, _, _ = train_test_split(exemplars, categories,
                                                                            train_size=260, stratify=categories)
        elif strategie == "qualité":
            working_df = working_df[~self.meta_df["qual"].isna()]
            exemplars, categories =  working_df.iloc[:,:-1], working_df.iloc[:,-1]

            self.exemplars, self.exemplar_category, _, _ = train_test_split(exemplars, categories,
                                                                            train_size=260, stratify=categories)
        else:
            date = working_df.index.get_level_values(0)
            working_df["date"] = date
            working_df["date"] = working_df.groupby(level=1).diff()["date"]

            self.exemplars = working_df.groupby(by=self.date.columns[-1]).sample(20, replace=True, weights="date")
            self.exemplar_category = self.data.loc[self.exemplar.index,:]
            self.exemplar_category = self.exemplar_category.iloc[:,-1]

    
    def fit(self, n_iter=200, tol_stop=5, batch_size=100, permutation=False):
        """
            Minimize Cross-Entropy 
        """
        tolerance_account = 0
        optimizer = optim.Adam(self.attention.parameters(), lr=1e-3)

        best = 10
        # y : (m, p) | m samples | p classes | value 1 ou 0
        # samples : (m, o) | m samples | o features
        if permutation:
            train_samples, eval_samples = train_test_split(self.X_train, train_size=10_000, test_size=2_000)
            y_train, y_eval = train_test_split(self.y_train, train_size=10_000, test_size=2_000)
        else:
            train_samples, eval_samples, y_train, y_eval = train_test_split(self.X_train, self.y_train, train_size=10_000, test_size=2_000)

        train_dataset = TensorDataset(tensor(train_samples),tensor(y_train)) # create your datset
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True) # create your dataloader

        eval_dataset = TensorDataset(tensor(eval_samples),tensor(y_eval)) # create your datset
        eval_dataloader = DataLoader(eval_dataset,batch_size=batch_size,shuffle=True) # create your dataloader

        for epoch in range(n_iter):
            
            self.torch_loop(train_dataloader, optimizer)
            
            precision = self.torch_loop(eval_dataloader, optimizer, mode="eval")
            print(epoch, precision)
            if precision < best:
                best = precision
                tolerance_account = 0
            elif tolerance_account == tol_stop:
                break
            else:
                tolerance_account += 1


    def torch_loop(self, dataloader, optimizer, mode="train"):
        if mode == "train":
            self.attention.train()
            for e, (X, y) in enumerate(dataloader):

                # ============ | Training Set | ============
                self.attention.train()
                
                #t = time.time()
                # scores : (m, p) | m samples | p classes
                scores = self.predict(X)
                #print(time.time()-t)

                # Loss : float ?
                L = nn.CrossEntropyLoss()
                loss = L(scores, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(loss)
        else:
            #accuracy = MultilabelPrecision(self.y_train.shape[-1])
            with no_grad():
                self.attention.eval()

                precision = []
                for e, (X,y) in enumerate(dataloader):
                    # ============ | Testing Set | ============
                    self.attention.eval()
                    scores = self.predict(X)
                    L = nn.CrossEntropyLoss()
                    loss = L(scores, y)

                    precision.append(loss.item())
            #print(precision)
            return np.mean(precision)


    def predict(self, X_in ):
        """
        Algorithme:
            For each classes:
                1. Compute the distance between y and a set X of other vectors
                2. Compute the value (s) giving more or less importance depending on the distance
                3. Compute the overall score (S) of the class
            4. Compute probability of each class through SoftMax
            5. Backpropagation to uptade the attention weights
        """

        #score : (p, m) | p classes | m samples 
        scores = []
        for e,(model_attention, classe) in enumerate(zip(self.attention, self.classes)):

            # X_in.shape = (m,o) | m samples | o features
            X= self.gen_data(classe, X_in)
            # X_in.shape = (m,n,o) | m samples | n exemplars | o features

            d_xy = model_attention(X)
            # d_xy.shape = (m,n) | m samples | n exemplars 


            # prestige.shape = (n) | m samples
            #prestige = model_prestige(X_meta)

            # s_r.shape = (m,n) | m samples | n exemplars
            s_r = self.prieur_beta[e] * exp(-self.fuzzy_param * d_xy)

            # S_R.shape = (m) | m samples
            S_R = s_r.sum(1)

            scores.append(S_R)
        #print(len(scores), scores[0].)
        # out : (m,p) | m samples | p classes -> moveaxis change la position des p classes
        #out = nn.functional.softmax(moveaxis(stack(scores),0,1).reshape((-1,len(self.attention))),dim=-1)
        out = moveaxis(stack(scores),0,1).reshape((-1,len(self.attention)))
        out = out/(out.sum(-1).reshape((-1,1))+0.0001)
        #print(out.shape)
        return out 
    


    def gen_data(self, classe, X_in):
        """
            From the exemplars in memory, generate the set of vectors 
        """

        return torch.abs(X_in.reshape((X_in.shape[0],1,-1))-self.exemplars[classe].reshape((1,-1,X_in.shape[-1])))



    def gen_samples(self,n_samples=10):
        """
            For the sample generation
            1. We pick N samples out of the one we can test
            2. We 
        """

        # We'll apply different filter for the dataframe here
        ### 1 - To keep only the relevant text which are indexed in the multi-index level 1
        ### 2 - To keep only the versions which have been produced
        ##### 2.a - In the six month period starting from the selected date
        ##### 2.b - If there is texts which were not treated at this time, we also take the last version, starting from 6 month ago
        
        self.X_train, self.y_train = self.get_Xy(n_samples, self.train_eval)
        self.X_test, self.y_test = self.get_Xy(n_samples, self.test)

        self.X_train = torch.from_numpy(np.array([element for classe in tqdm(self.X_train) for element in classe])).to(torch.float32).to("mps")
        self.y_train = torch.from_numpy(np.array([element for classe in tqdm(self.y_train) for element in classe])).to(torch.float32).to("mps")
        self.X_test = torch.from_numpy(np.array([element for classe in tqdm(self.X_test) for element in classe])).to(torch.float32).to("mps")
        self.y_test = torch.from_numpy(np.array([element for classe in tqdm(self.y_test) for element in classe])).to(torch.float32).to("mps")
    
    def get_Xy(self,n_samples, dataset):
        ### We compute the index for the rule (2.b first) 
        last_index = self.data[self.data.index.get_level_values(0) <= self.date+datetime.timedelta(180)].groupby(level=1).tail(1).index
        working_df = self.data[
        
            

            (
                ### Rule 2.A
                (
                    (self.data.index.get_level_values(0) >= self.date)
                    &
                    (self.data.index.get_level_values(0) <= self.date+datetime.timedelta(180))
                )
                ### Rule 2.B
                |
                (
                    self.data.index.isin(last_index)
                )

            )
        
            ]
        X_out, y_out = [], []
        list_classes = list(self.class2text.keys())

        for e, (k, class_members) in tqdm(enumerate(zip(list_classes, dataset))):
            
            samples = working_df[working_df.index.get_level_values(1).isin(class_members)]
            y = [[1 if f in [self.class2index[k] for k in self.text2class[x]] else 0 for f in range(len(dataset))] for x in samples.index.get_level_values(1)]
            y = [[x / sum(z) for x in z] for z in y]
            X_out.append(samples.to_numpy())
            y_out.append(y)
            
        return X_out, y_out
    
    def test_du_model(self):
        X_test, y_test = self.get_Xy(2_500,self.test)

        X_test = torch.from_numpy(np.array([element for classe in tqdm(X_test) for element in classe])).to(torch.float32).to("mps")
        y_test = torch.from_numpy(np.array([element for classe in tqdm(y_test) for element in classe])).to(torch.float32).to("mps")

        proba = self.predict(X_test)

        return proba, y_test
