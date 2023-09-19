from surprise import Dataset
from trusttrainset import CustomTrainset
import os
import itertools
from collections import defaultdict

class CustomDataset(Dataset):
    def __init__(self,reader,trust_reader):
        super().__init__(reader)
        self.trust_reader = trust_reader


    @classmethod
    def load_from_dataframe(cls, df, df_trust,reader, trust_reader):
        """Load a dataset from a pandas dataframe.
        Use this if you want to use a custom dataset that is stored in a pandas
        dataframe. See the :ref:`User Guide<load_from_df_example>` for an
        example.
        Args:
            df(`Dataframe`): The dataframe containing the ratings. It must have
                three columns, corresponding to the user (raw) ids, the item
                (raw) ids, and the ratings, in this order.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file. Only the ``rating_scale`` field needs to be
                specified.
            trust_reader(:obj:TrustReader): A reader to read the file. Only the 
                ``trust_scale`` field needs to be specified
        """

        return CustomDatasetAutoFolds(reader=reader, trust_reader=trust_reader, df=df, df_trust=df_trust)

    def read_trusts(self, file_name):
        """Return a list of trusts (trustor, trustee, trust)
        read from file name"""

        with open(os.path.expanduser(file_name)) as f:
            raw_trusts = [
                self.reader.parse_line(line)
                for line in itertools.islice(f, self.reader.skip_lines, None)
            ]

        return raw_trusts

    def custom_construct_trainset(self, raw_trainset, raw_trusts):

        # rating data in raw_trainset 
        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # raw_trust
        raw2inner_id_trustor = {}
        raw2inner_id_trustee = {}

        current_trustor_index = 0
        current_trustee_index = 0

        raw_ut = defaultdict(list) # {raw_trustor_id: (raw_trustee_id, trust)}
        raw_vt = defaultdict(list) # {raw_trustee_id: (raw_trustor_id, trust)}
        # ut = defaultdict(list)
        # vt = defaultdict(list)

        # combine userid rating and trust
        urt = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:
            try:
                # {rawid : uid}
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        # Create trust defaultdict {raw_trustor_id: (raw_trustee_id, trust)}
            # for utrid, vtrid, t in raw_trusts:
            #     if utrid == urid:
            #         if (vtrid,t) not in raw_ut[utrid]:
            #             raw_ut[utrid].append((vtrid,t))
            #     elif vtrid == urid:
            #         if (utrid,t) not in raw_vt[vtrid]:
            #             raw_vt[vtrid].append((utrid,t))
        for utrid, vtrid, t in raw_trusts:
            if utrid in raw2inner_id_users.keys() and vtrid in raw2inner_id_users:
                if (raw2inner_id_users[vtrid], t) not in raw_ut[utrid]:
                    raw_ut[utrid].append((raw2inner_id_users[vtrid],t))
                if (raw2inner_id_users[utrid], t) not in raw_vt[vtrid]:
                    raw_vt[vtrid].append((raw2inner_id_users[utrid],t))

        # Match raw user id and trustor id
        # to match the same inner id of user's rating and user's trust
        match_ut_id = set(raw2inner_id_users.keys()) & set(raw_ut.keys())
        ut = {raw2inner_id_users[key]:raw_ut[key] for key in match_ut_id}

        # Match raw user id and trustee id
        match_vt_id = set(raw2inner_id_users.keys()) & set(raw_vt.keys())
        vt = {raw2inner_id_users[key]:raw_vt[key] for key in match_vt_id}


        # user(trustor) raw id, user(trustee) raw id, translated trust
        # for utrid, vtrid, t in raw_trusts:
        #     try:
        #         utid = raw2inner_id_trustor[utrid]
        #     except KeyError:
        #         utid = current_trustor_index
        #         raw2inner_id_trustor[utrid] = current_trustor_index
        #         current_trustor_index += 1
        #     try:
        #         vtid = raw2inner_id_trustee[vtrid]
        #     except KeyError:
        #         vtid = current_trustee_index
        #         raw2inner_id_trustee[vtrid] = current_trustee_index
        #         current_trustee_index += 1

        #     ut[utid].append((vtid, t))
        #     vt[vtid].append((utid, t))

            # urt[uid].append((iid, r, t))


        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)
        n_trustors = len(ut)
        n_trustees = len(vt)
        n_trusts = len(raw_trusts)

        trainset = CustomTrainset(
            ut,
            vt,
            n_trustors,
            n_trustees,
            n_trusts,
            ur,
            ir,
            n_users,
            n_items,
            n_ratings,
            self.reader.rating_scale,
            raw2inner_id_users,
            raw2inner_id_items,
        )

        return trainset

class CustomDatasetAutoFolds(CustomDataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are not predefined. (Or for when there are no folds at
    all)."""

    def __init__(
        self, 
        ratings_file=None,
        trusts_file=None, 
        reader=None,
        trust_reader=None, 
        df=None,
        df_trust=None 
        ):

        super().__init__(trust_reader, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None and trusts_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
            self.trusts_file = trusts_file
            self.raw_trusts = self.read_trusts(self.trusts_file)
        elif df is not None and df_trust is not None:
            self.df = df
            self.raw_ratings = [
                (uid, iid, float(r), None)
                for (uid, iid, r) in self.df.itertuples(index=False)
            ]
            self.df_trust = df_trust
            self.raw_trusts = [
                (trustor_id, trustee_id, float(t))
                for (trustor_id, trustee_id, t) in self.df_trust.itertuples(index=False)
            ]
        # elif trusts_file is not None:
        #     self.trusts_file = trusts_file
        #     self.raw_trusts = self.read_trusts(self.trusts_file)
        # elif df_trust is not None:
        #     self.df_trust = df_trust
        #     self.raw_trusts = [
        #         (trustor_id, trustee, float(t))
        #         for (trustor_id, trustee_id, t) in self.df_trust.itertuples(index=False)
        #     ]
            
        else:
            raise ValueError("Must specify ratings file and trusts file or dataframe of trust and rating.")