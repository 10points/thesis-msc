import numpy as np
from surprise import AlgoBase
from math import sqrt
from utils import get_rng

class TrustSVD(AlgoBase):
    
    def __init__(self, n_factors=20,n_epochs=10,init_mean=0, init_std_dev=.1,
    max_clip_value=5, lr_all=0.007,reg_all=.02,lr_wv=None, reg=None, reg_t=None,
    lr_bu=None, lr_bj=None, lr_pu=None, lr_qj=None, lr_yi=None,
    random_state=None, verbose=False,
    ):
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.max_clip_value = max_clip_value
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.lr_wv = lr_wv if lr_wv is not None else lr_all
        self.reg = reg if reg is not None else reg_all
        self.reg_t = reg_t if reg_t is not None else reg_all
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bj = lr_bj if lr_bj is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qj = lr_qj if lr_qj is not None else lr_all
        self.lr_yi = lr_yi if lr_yi is not None else lr_all
        
        self.random_state = random_state
        self.verbose = verbose

        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        rng = get_rng(self.random_state)

        # user biases
        bu = np.zeros(trainset.n_users, dtype=np.float64)

        # item biases
        bj = np.zeros(trainset.n_items, dtype=np.float64)

        # user factors
        pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))

        # item factors
        qj = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

        # item implicit factors
        yi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))
        u_impl_fdb = np.zeros(self.n_factors, dtype=np.float64)

        # trust implicit factors
        wv = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
        trust_impl_fdb = np.zeros(self.n_factors, dtype=np.float64)

        global_mean = self.trainset.global_mean
        lr_bu = self.lr_bu
        lr_bj = self.lr_bj
        lr_pu = self.lr_pu
        lr_qj = self.lr_qj
        lr_yi = self.lr_yi
        lr_wv = self.lr_wv

        reg = self.reg
        reg_t = self.reg_t

        sum_euj_for_Iu = 0
        sum_euv =  0
        sum_euv_wv =  0
        sum_euj_for_Uj = 0
        sum_euj_qj= 0
        sum_euj_Iu_qj = 0
        sum_euj_Tu_qj = 0

        # sum error for updating pu
        sum_euj_pu = 0

        # sum error for updating qj
        sum_euj_user_impl = 0
        sum_euj_trust_impl = 0

        # Grad clipping
        max_clip_value = self.max_clip_value

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))

            for u in range(trainset.n_users):

                # Iu: Set of items were rated by user u
                Iu = []
                for i, _ in trainset.ur[u]:
                    Iu.append(i)
                Iu_length = len(Iu)
                sqrt_Iu = sqrt(Iu_length)


                # sigma_yi/(Iu**-1/2): user implicit feedback
                u_impl_fdb[:] = 0
                for k in range(Iu_length):
                    j = Iu[k]
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += yi[j, f] / sqrt_Iu

                # Tu: Set of users trusted by user u
                Tu = []
                try:
                    for uid, _ in trainset.ut[u]:
                        if u <= trainset.n_users:
                            Tu.append(uid)
                    Tu_length = len(Tu)
                    sqrt_Tu = sqrt(Tu_length)
                except KeyError:
                    Tu = []
                    Tu_length = len(Tu)
                    sqrt_Tu = sqrt(Tu_length)

            

                # sigma_wv/(Tu**-1/2): trust implicit feedback
                trust_impl_fdb[:] = 0
                if len(Tu) != 0: 
                    for k in range(Tu_length):
                        j = Tu[k]
                        for f in range(self.n_factors):
                            trust_impl_fdb[f] += wv[j, f] / sqrt_Tu
                else:
                    trust_impl_fdb[:] = 0

                # e_uj : rating error
                dot = 0
                for i, r in trainset.ur[u]:
                    for f in range(self.n_factors):
                        dot += qj[i, f] * (pu[u, f] + u_impl_fdb[f] + trust_impl_fdb[f])

                    e_uj = r - (bu[u] + bj[i] + trainset.global_mean + dot)

                    sum_euj_for_Iu += e_uj
                    for f in range(self.n_factors):
                        sum_euj_qj += e_uj*qj[i, f]
                        sum_euj_Iu_qj += e_uj*qj[i, f]/sqrt_Iu
                        if len(Tu) != 0:
                            sum_euj_Tu_qj += e_uj*qj[i, f]/sqrt_Tu
                        else:
                            sum_euj_Tu_qj = 0


                # e_uv: trust error
                try:
                    e_uv = 0
                    dot = 0
                    # sum_e_uv_wv = 0
                    for v, t in trainset.ut[u]:
                        for f in range(self.n_factors):
                            dot += wv[v, f] * pu[u, f]
                        e_uv = t - dot
                        for f in range(self.n_factors):
                            sum_euv_wv += e_uv*wv[v, f]
                            
                except KeyError:
                    pass
                    

                ############## Update Parameter ##############
                # bu
                for i in Iu:   
                    grad_bu = sum_euj_for_Iu + (reg*bu[u] / sqrt_Iu)
                    clip_grad_bu = np.clip(grad_bu, -max_clip_value, max_clip_value)
                    bu[u] +=  lr_bu * clip_grad_bu
            
                
                # pu
                if len(Tu) != 0:
                    for i in Iu:
                        for f in range(self.n_factors):
                            puf = pu[u, f]
                            grad_pu = (sum_euj_qj + reg_t*sum_euv_wv) + (reg/sqrt_Iu + reg_t/sqrt_Tu)*puf
                            clip_grad_pu = np.clip(grad_pu, -max_clip_value, max_clip_value)
                            pu[u, f] += lr_pu * (clip_grad_pu)
                else:
                    reg_t_sqrt_Tu = 0
                    for i in Iu:
                        for f in range(self.n_factors):
                            puf = pu[u, f]
                            grad_pu = (sum_euj_qj + reg_t*sum_euv_wv) + (reg/sqrt_Iu + reg_t_sqrt_Tu)*puf
                            clip_grad_pu = np.clip(grad_pu, -max_clip_value, max_clip_value)
                            pu[u, f] += lr_pu * (clip_grad_pu)

                # yi
                for i in Iu:
                    Ui_length = len(trainset.ir[i])
                    sqrt_Ui = sqrt(Ui_length)
                    for f in range(self.n_factors):
                        grad_yi = sum_euj_Iu_qj + reg*yi[i, f]/sqrt_Ui
                        clip_grad_yi = np.clip(grad_yi, -max_clip_value, max_clip_value)
                        yi[i, f] += lr_yi * (clip_grad_yi)


                # wv
                try:
                    e_uv = 0
                    Tv_length = len(trainset.vt[u])
                    sqrt_Tv = sqrt(Tv_length) 
                    for v, t in trainset.ut[u]: # same as for v from Tu
                            for f in range(self.n_factors):
                                dot += wv[v, f] * pu[u, f]
                            e_uv = t - dot
                            for f in range(self.n_factors):
                                grad_wv = sum_euj_Tu_qj + reg_t*e_uv*pu[u, f] + reg*wv[v, f]/sqrt_Tv
                                clip_grad_wv = np.clip(grad_wv, -max_clip_value, max_clip_value)
                                wv[v, f] += lr_wv * clip_grad_wv
                except KeyError:
                    pass
                    

            for i in range(trainset.n_items):
                # Uj
                Uj = []
                for u in trainset.ir[i]:
                    Uj.append(u)
                Uj_length = len(Uj)
                sqrt_Uj = sqrt(Uj_length)

                
                for u in Uj:
                    # Iu: Set of items were rated by user u
                    Iu = []
                    for item, _ in trainset.ur[u]:
                        Iu.append(item)
                    Iu_length = len(Iu)
                    sqrt_Iu = sqrt(Iu_length)

                    
                    # sigma_yi/(Iu**-1/2): user implicit feedback
                    u_impl_fdb[:] = 0
                    for k in range(Iu_length):
                        j = Iu[k]
                        for f in range(self.n_factors):
                            u_impl_fdb[f] += yi[j, f] / sqrt_Iu

                        # Tu: Set of users trusted by user u
                    Tu = []
                    try:
                        for uid, _ in trainset.ut[u]:
                            Tu.append(uid)
                        Tu_length = len(Tu)
                        sqrt_Tu = sqrt(Tu_length)
                    except KeyError:
                        Tu = []
                        Tu_length = len(Tu)
                        sqrt_Tu = sqrt(Tu_length)


                    # sigma_wv/(Tu**-1/2): trust implicit feedback
                    trust_impl_fdb[:] = 0
                    if len(Tu) != 0: 
                        for k in range(Tu_length):
                            j = Tu[k]
                            for f in range(self.n_factors):
                                trust_impl_fdb[f] += wv[j, f] / sqrt_Tu
                    else:
                        trust_impl_fdb[:] = 0

                    # e_uj : rating error
                    e_uj = 0 
                    dot = 0                          
                    for iid, r in trainset.ur[u]:
                        for f in range(self.n_factors):
                            dot += qj[iid, f] * (pu[u, f] + u_impl_fdb[f] + trust_impl_fdb[f])
            
                        e_uj = r - (bu[u] + bj[iid] + trainset.global_mean + dot)
                    
                        sum_euj_for_Uj += e_uj
                        for f in range(self.n_factors):
                            sum_euj_pu += e_uj*pu[u, f]              
                            sum_euj_user_impl += e_uj*u_impl_fdb[f]
                            sum_euj_trust_impl += e_uj*trust_impl_fdb[f]

                # bj
                grad_bj = sum_euj_for_Uj + (reg*bj[i] / sqrt_Uj)
                clip_grad_bj = np.clip(grad_bj, -max_clip_value, max_clip_value)
                bj += lr_bj * clip_grad_bj

                # qj
                for f in range(self.n_factors):
                    grad_qj = (sum_euj_pu + sum_euj_user_impl + sum_euj_trust_impl) + (reg*qj[i, f]/sqrt_Uj)
                    clip_grad_qj = np.clip(grad_qj, -max_clip_value, max_clip_value)
                    qj += lr_qj * clip_grad_bj

        self.bu = np.asarray(bu)
        self.bj = np.asarray(bj)
        self.pu = np.asarray(pu)
        self.qj = np.asarray(qj)
        self.yi = np.asarray(yi)
        self.wv = np.asarray(wv)

    def estimate(self, u, i):

        est = self.trainset.global_mean


        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bj[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            Iu = len(self.trainset.ur[u])
            u_impl_feedback = (sum(self.yi[j] for (j, _)
                            in self.trainset.ur[u]) / np.sqrt(Iu))
            try:
                Tu = len(self.trainset.ut[u])
                t_impl_feedback = (sum(self.wv[j] for (j, _)
                                in self.trainset.ut[u]) / np.sqrt(Tu))
                est += np.dot(self.qj[i], (self.pu[u] + u_impl_feedback + t_impl_feedback ))
            except KeyError:
                t_impl_feedback = 0
                est += np.dot(self.qj[i], (self.pu[u] + u_impl_feedback + t_impl_feedback ))

        return est