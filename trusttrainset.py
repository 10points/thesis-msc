from surprise import Trainset

class CustomTrainset(Trainset):

    def __init__(
        self,
        ut,
        vt,
        n_trustors,
        n_trustees,
        n_trusts,       
        *args, 
        **kwargs):
        
        super().__init__(*args, **kwargs)
        self.ut = ut
        self.vt = vt
        self.n_trustors = n_trustors
        self.n_trustees = n_trustees
        self.n_trusts = n_trusts
    

    def all_trusts(self):
        """Generator function to iterate over all ratings.
        Yields:
            A tuple ``(trustor_id, trustee_id, trust)`` where ids are inner ids (see
            :ref:`this note <raw_inner_note>`).
        """

        for u, u_trusts in self.ut.items():
            for v,t in u_trusts:            
                    yield (u, v, t)

    def all_trustors(self):
        """Generate function to iterate over all trustors

        Yields:
            Inner id of trustors.
        """
        return range(self.n_trustors)

    def all_trustees(self):
        """Generate function to iterate over all trustors

        Yields:
            Inner id of trustors.
        """
        return range(self.n_trustees)